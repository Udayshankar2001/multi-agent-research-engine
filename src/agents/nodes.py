import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.client import Client
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import (
    CritiqueFeedback,
    ResearchGraphState,
    ResearchPlan,
    SubTopicFinding,
    WorkerTaskState,
)
from src.mcp_server.research_tools import mcp

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# Fast Mixture-of-Experts primary model (4B active params) + 31B/Flash-Lite fallbacks
# Pool A: Fast, high-quota models dedicated to the Multi-Agent Pipeline
m_31_lite = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    max_retries=1,
)

m_25_lite = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0.2,
    max_retries=1,
)

m_gemma_26b = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    temperature=0.2,
    max_retries=1,
)

m_gemma_31b = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=api_key,
    temperature=0.2,
    max_retries=1,
)

llm = m_31_lite.with_fallbacks([m_25_lite, m_gemma_26b, m_gemma_31b])



# Pydantic parsers so open-weights Gemma models output strict validated JSON
plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
critique_parser = PydanticOutputParser(pydantic_object=CritiqueFeedback)

_gemini_semaphore = asyncio.Semaphore(2)


def _extract_text(content: Any) -> str:
    """Normalize LangChain AIMessage.content (str or list of content blocks) into a clean string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return str(content)


def _extract_mcp_result(call_result: Any) -> Any:
    """Extract clean Python dict/list from MCP v2 CallToolResult."""
    sc = getattr(call_result, "structured_content", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=12))
async def _invoke_llm_safe(prompt: str) -> str:
    """Rate-limited and retry-protected LLM text generation."""
    async with _gemini_semaphore:
        response = await llm.ainvoke(prompt)
        return _extract_text(response.content)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=12))
async def _invoke_structured_safe(prompt: str, parser: PydanticOutputParser) -> Any:
    """Rate-limited structured JSON generation validated by Pydantic."""
    async with _gemini_semaphore:
        response = await llm.ainvoke(prompt)
        raw_text = _extract_text(response.content)
        return parser.parse(raw_text)


# --- NODE 1: Memory Check & Query Decomposition Planner ---
async def planner_node(state: ResearchGraphState) -> Dict[str, Any]:
    user_query = state["user_query"]

    async with Client(mcp) as client:
        mem_res = await client.call_tool(
            "recall_long_term_memory",
            {"topic": user_query},
        )
        prior_memories = _extract_mcp_result(mem_res) or []

    memory_context = ""
    if prior_memories:
        formatted = "\n".join(
            f"- {m.get('topic')}: {m.get('summary')}" for m in prior_memories
        )
        memory_context = f"\nPrior Verified Knowledge Found in Memory:\n{formatted}\n"

    prompt = f"""You are a Lead Research Architect. Break the user's research request into 3 focused, non-overlapping sub-questions and concise web search queries.
{memory_context}
User Research Request: {user_query}

{plan_parser.get_format_instructions()}
Return ONLY valid JSON matching the schema above.
"""
    plan_obj: ResearchPlan = await _invoke_structured_safe(prompt, plan_parser)

    return {
        "prior_memories": prior_memories,
        "plan": plan_obj.tasks,
        "revision_count": 0,
    }


# --- NODE 2: Parallel Map-Reduce Worker (Executes via MCP Tools) ---
async def worker_node(state: WorkerTaskState) -> Dict[str, Any]:
    session_id = state["session_id"]
    task = state["task"]

    async with Client(mcp) as client:
        search_res = await client.call_tool(
            "search_web",
            {"query": task.search_query, "max_results": 3},
        )
        search_hits: List[Dict[str, str]] = _extract_mcp_result(search_res) or []

                # Scrape top 2 URLs concurrently with an 8-second timeout per URL
        valid_urls = [h["url"] for h in search_hits if h.get("url")][:2]
        if valid_urls:
            scrape_tasks = [
                asyncio.wait_for(
                    client.call_tool(
                        "scrape_and_index_url",
                        {"url": url, "session_id": session_id},
                    ),
                    timeout=8.0,
                )
                for url in valid_urls
            ]
            await asyncio.gather(*scrape_tasks, return_exceptions=True)


        ev_res = await client.call_tool(
            "retrieve_evidence",
            {"query": task.sub_question, "session_id": session_id, "top_k": 4},
        )
        chunks: List[Dict[str, Any]] = _extract_mcp_result(ev_res) or []

    if not chunks and search_hits:
        chunks = [
            {
                "source_url": h.get("url", ""),
                "text": f"{h.get('title', '')}: {h.get('snippet', '')}",
            }
            for h in search_hits
            if h.get("snippet")
        ]

    evidence_text = "\n\n".join(
        f"[Source: {c.get('source_url')}]\n{c.get('text')}" for c in chunks
    )
    used_urls = list({c.get("source_url") for c in chunks if c.get("source_url")})
    snippets = [c.get("text", "") for c in chunks if c.get("text")]

    prompt = f"""You are a meticulous Research Analyst. Answer the sub-question using ONLY the retrieved evidence below. Do not invent facts outside the evidence. Include exact source URLs.

Sub-Question: {task.sub_question}

Retrieved Evidence:
{evidence_text if evidence_text else "No external evidence retrieved."}
"""
    summary_text = await _invoke_llm_safe(prompt)

    finding = SubTopicFinding(
        sub_question=task.sub_question,
        summary=summary_text,
        source_urls=used_urls,
        evidence_snippets=snippets,
    )

    return {"findings": [finding]}


# --- NODE 3: Synthesis Writer Agent ---
async def writer_node(state: ResearchGraphState) -> Dict[str, Any]:
    findings = state.get("findings", [])
    prior_memories = state.get("prior_memories", [])
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)

    all_urls = sorted(
        list({url for f in findings for url in f.source_urls if url})
    )

    findings_block = "\n\n".join(
        f"### Sub-Question: {f.sub_question}\nSources: {', '.join(f.source_urls)}\nNotes:\n{f.summary}"
        for f in findings
    )

    memory_block = ""
    if prior_memories:
        memory_block = "### Prior Verified Memory:\n" + "\n".join(
            f"- {m.get('topic')}: {m.get('summary')}" for m in prior_memories
        )

    revision_block = ""
    if critique and not critique.passed:
        revision_block = f"""
CRITICAL REVISION REQUIRED (Attempt {revision_count}):
The Critic flagged these issues in your previous draft:
{chr(10).join(f'- {issue}' for issue in critique.issues)}

Fix Instructions: {critique.revision_instructions}
"""

    prompt = f"""You are a Senior Research Report Writer. Synthesize a comprehensive, well-structured Markdown report answering the User Query based strictly on the verified findings below.
Rules:
1. Cite sources inline using exact URLs in brackets like [https://...].
2. Every factual claim must trace directly to the provided findings.
3. End with a 'References' section listing all cited URLs.
{revision_block}
User Query: {state['user_query']}

{memory_block}

{findings_block}
"""
    draft_text = await _invoke_llm_safe(prompt)

    updates: Dict[str, Any] = {
        "draft_report": draft_text,
        "all_source_urls": all_urls,
    }
    if revision_count == 0:
        updates["initial_draft"] = draft_text

    return updates


# --- NODE 4: Critic & Hallucination Reflection Agent ---
async def critic_node(state: ResearchGraphState) -> Dict[str, Any]:
    findings = state.get("findings", [])
    draft = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0)

    prior_mem_text = "\n".join(
        f"[Verified Prior Memory - {m.get('topic')}]: {m.get('summary')}"
        for m in state.get("prior_memories", [])
    )
    raw_evidence = prior_mem_text + "\n---\n" + "\n---\n".join(
        snippet for f in findings for snippet in f.evidence_snippets
    )

    sub_questions = "\n".join(f"- {t.sub_question}" for t in state.get("plan", []))

    prompt = f"""You are an adversarial Fact-Checking Critic. Audit the Draft Report against the Raw Retrieved Evidence and Sub-Questions.

Sub-Questions Required:
{sub_questions}

Raw Retrieved Evidence:
{raw_evidence[:12000]}

Draft Report to Audit:
{draft}

Check:
1. Are there major fabricated facts or numbers that contradict or have zero basis in the Raw Retrieved Evidence?
2. Are all sub-questions reasonably addressed?
3. If source URLs exist in the Raw Retrieved Evidence, are they cited inline? (If no external URLs were retrieved and only Verified Prior Memory was used, do NOT fail the draft for missing URLs.)

Note: Minor paraphrasing or synthesis across chunks is acceptable. Set passed=True unless there is a clear unsupported claim or missing sub-question.

{critique_parser.get_format_instructions()}
Return ONLY valid JSON matching the schema above.
"""
    critique_obj: CritiqueFeedback = await _invoke_structured_safe(prompt, critique_parser)

    return {
        "critique": critique_obj,
        "revision_count": revision_count + 1,
    }


# --- NODE 5: Long-Term Memory Commit Node ---
async def memory_commit_node(state: ResearchGraphState) -> Dict[str, Any]:
    final_report = state.get("draft_report", "")
    source_urls = state.get("all_source_urls", [])
    user_query = state["user_query"]

    summary_prompt = (
        f"Summarize the key verified conclusions of this report in 3 concise sentences "
        f"for long-term knowledge storage:\n\n{final_report[:6000]}"
    )

    summary_text = await _invoke_llm_safe(summary_prompt)

    async with Client(mcp) as client:
        await client.call_tool(
            "save_to_long_term_memory",
            {
                "topic": user_query,
                "summary": summary_text,
                "source_urls": source_urls,
            },
        )

    return {"final_report": final_report}
