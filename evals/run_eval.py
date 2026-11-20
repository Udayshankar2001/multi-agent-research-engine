import asyncio
import json
import os
import re
import time
import uuid
import pathlib

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.graph import research_graph
from src.agents.nodes import _extract_text
from src.agents.state import ResearchGraphState


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GeminiFreeJudge(DeepEvalBaseLLM):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        j1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, response_mime_type="application/json", max_output_tokens=8192, max_retries=1)
        j2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key, response_mime_type="application/json", max_output_tokens=8192, max_retries=1)
        j3 = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, response_mime_type="application/json", max_output_tokens=8192, max_retries=1)
        self.model = j1.with_fallbacks([j2, j3])

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None, **kwargs):
        if schema is not None:
            structured = self.model.with_structured_output(schema)
            return structured.invoke(prompt)
        res = self.model.invoke(prompt)
        return _clean_json(_extract_text(res.content))

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        if schema is not None:
            structured = self.model.with_structured_output(schema)
            return await structured.ainvoke(prompt)
        res = await self.model.ainvoke(prompt)
        return _clean_json(_extract_text(res.content))

    def get_model_name(self) -> str:
        return "gemini-pool-b-judge"


EVAL_DATASET = [
    {
        "query": "What is Model Context Protocol (MCP) and how does it standardize AI agent integrations?",
        "expected_output": "MCP is a JSON-RPC 2.0 open standard defining a uniform protocol for LLM applications to discover and execute external tools, resources, and prompts.",
    },
    {
        "query": "How does Qdrant vector database implement HNSW indexing and payload filtering?",
        "expected_output": "Qdrant uses HNSW graphs for approximate nearest neighbor search combined with filterable payload indexes for multi-tenant session isolation.",
    },
]


async def run_benchmark():
    judge = GeminiFreeJudge()
    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=False)
    precision = ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=False)

    results = []

    for idx, item in enumerate(EVAL_DATASET, start=1):
        query = item["query"]
        expected = item["expected_output"]
        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        print(f"\n[Eval {idx}/{len(EVAL_DATASET)}] Running query: {query}")

        state: ResearchGraphState = {
            "session_id": session_id,
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

        t0 = time.perf_counter()
        final_state = await research_graph.ainvoke(state)
        latency = round(time.perf_counter() - t0, 2)

        contexts = [
            snippet
            for f in final_state.get("findings", [])
            for snippet in f.evidence_snippets
        ][:8] or ["MCP standardizes tool discovery and execution between LLMs and data sources."]

        final_text = final_state.get("final_report", final_state.get("draft_report", ""))
        initial_text = final_state.get("initial_draft", final_text)

        tc_initial = LLMTestCase(input=query, actual_output=initial_text, retrieval_context=contexts)
        tc_final = LLMTestCase(
            input=query,
            actual_output=final_text,
            expected_output=expected,
            retrieval_context=contexts,
        )

        faithfulness.measure(tc_initial)
        init_faith = round(faithfulness.score, 3)

        faithfulness.measure(tc_final)
        final_faith = round(faithfulness.score, 3)

        relevancy.measure(tc_final)
        rel_score = round(relevancy.score, 3)

        precision.measure(tc_final)
        prec_score = round(precision.score, 3)

        row = {
            "query": query,
            "latency_seconds": latency,
            "revision_count": final_state.get("revision_count", 0),
            "initial_faithfulness": init_faith,
            "final_faithfulness": final_faith,
            "answer_relevancy": rel_score,
            "contextual_precision": prec_score,
        }
        results.append(row)
        print("  Result:", row)

    pathlib.Path("evals/benchmark_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved benchmark results to evals/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
