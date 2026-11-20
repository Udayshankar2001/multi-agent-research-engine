import asyncio
import uuid
from typing import List, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.nodes import (
    critic_node,
    memory_commit_node,
    planner_node,
    worker_node,
    writer_node,
)
from src.agents.state import ResearchGraphState


def dispatch_workers(state: ResearchGraphState) -> List[Send]:
    """Fan out each planned sub-question to an isolated parallel worker node."""
    return [
        Send(
            "worker",
            {
                "session_id": state["session_id"],
                "task": task,
            },
        )
        for task in state["plan"]
    ]


def route_after_critic(state: ResearchGraphState) -> str:
    """Loop back to Writer if Critic rejects the draft (max 1 revision)."""
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)

    if critique and not critique.passed and revision_count <= 1:
        return "writer"
    return "memory_commit"


def build_research_graph():
    builder = StateGraph(ResearchGraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("memory_commit", memory_commit_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_workers, ["worker"])
    builder.add_edge("worker", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "memory_commit": "memory_commit",
        },
    )
    builder.add_edge("memory_commit", END)

    return builder.compile()


research_graph = build_research_graph()


if __name__ == "__main__":
    async def _smoke_test():
        session_id = f"run-{uuid.uuid4().hex[:8]}"
        query = "What is Model Context Protocol (MCP) in AI and why are enterprises adopting it?"

        print(f"Starting Multi-Agent Research Session: {session_id}\n")
        initial_state: ResearchGraphState = {
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

        async for event in research_graph.astream(initial_state, stream_mode="updates"):
            for node_name, output in event.items():
                print(f"[Node Completed] -> {node_name}")
                if node_name == "planner":
                    print("  Sub-questions:", [t.sub_question for t in output["plan"]])
                elif node_name == "critic":
                    c = output["critique"]
                    print(f"  Critic Passed: {c.passed} | Revision Count: {output['revision_count']}")
                    if not c.passed and c.issues:
                        print("  Flagged Issues:", c.issues)

                elif node_name == "memory_commit":
                    print("\n=== FINAL VERIFIED REPORT ===\n")
                    print(output["final_report"])

    asyncio.run(_smoke_test())
