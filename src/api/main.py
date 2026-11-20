import json
import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.graph import research_graph
from src.agents.state import ResearchGraphState
from src.mcp_server.research_tools import vector_store

app = FastAPI(
    title="Autonomous Multi-Agent Research & Synthesis Engine",
    version="1.0.0",
    description="LangGraph + MCP + Qdrant multi-agent research backend with real-time SSE streaming.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Research topic or question to investigate.")


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    final_report: str
    revision_count: int
    critic_passed: bool
    source_urls: List[str]
    latency_seconds: float


def _build_initial_state(query: str) -> ResearchGraphState:
    return {
        "session_id": f"session-{uuid.uuid4().hex[:8]}",
        "user_query": query,
        "prior_memories": [],
        "plan": [],
        "findings": [],
        "draft_report": "",
        "initial_draft": "",
        "critique": None,
        "revision_count": 0,
        "final_report": "",
        "all_source_urls": [],
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "multi-agent-research-engine"}


@app.get("/api/v1/memory/search")
async def search_long_term_memory(
    topic: str = Query(..., min_length=2, description="Topic to search in Qdrant verified memory"),
    top_k: int = Query(3, ge=1, le=10),
) -> Dict[str, Any]:
    hits = vector_store.recall_verified_memory(query=topic, score_threshold=0.55, top_k=top_k)
    return {"query": topic, "count": len(hits), "memories": hits}


@app.post("/api/v1/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    start_time = time.perf_counter()
    initial_state = _build_initial_state(request.query)

    try:
        final_state = await research_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {exc}")

    critique = final_state.get("critique")
    elapsed = round(time.perf_counter() - start_time, 2)

    return ResearchResponse(
        session_id=initial_state["session_id"],
        query=request.query,
        final_report=final_state.get("final_report", final_state.get("draft_report", "")),
        revision_count=final_state.get("revision_count", 0),
        critic_passed=bool(critique.passed) if critique else False,
        source_urls=final_state.get("all_source_urls", []),
        latency_seconds=elapsed,
    )


@app.post("/api/v1/research/stream")
async def stream_research(request: ResearchRequest) -> StreamingResponse:
    initial_state = _build_initial_state(request.query)

    async def event_generator():
        yield f"data: {json.dumps({'event': 'started', 'session_id': initial_state['session_id']})}\n\n"
        try:
            async for update in research_graph.astream(initial_state, stream_mode="updates"):
                for node_name, output in update.items():
                    payload: Dict[str, Any] = {"event": "node_completed", "node": node_name}

                    if node_name == "planner":
                        payload["sub_questions"] = [t.sub_question for t in output.get("plan", [])]
                    elif node_name == "critic":
                        c = output.get("critique")
                        payload["critic_passed"] = bool(c.passed) if c else False
                        payload["RevisionCount"] = output.get("revision_count", 0)
                    elif node_name == "memory_commit":
                        payload["final_report"] = output.get("final_report", "")

                    yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
