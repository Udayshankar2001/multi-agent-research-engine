import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field


class SubQuestionTask(BaseModel):
    sub_question: str = Field(description="Specific, focused research sub-question to investigate.")
    search_query: str = Field(description="Concise keyword search query optimized for web search.")


class ResearchPlan(BaseModel):
    tasks: List[SubQuestionTask] = Field(
        min_length=2,
        max_length=4,
        description="2 to 4 non-overlapping sub-questions covering all aspects of the user's query.",
    )


class SubTopicFinding(BaseModel):
    sub_question: str
    summary: str
    source_urls: List[str]
    evidence_snippets: List[str]


class CritiqueFeedback(BaseModel):
    passed: bool = Field(
        description="True ONLY if every claim in the draft is supported by retrieved evidence, all sub-questions are answered, and citations are accurate."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific unsupported claims, missing sub-questions, or citation errors to fix. Empty if passed=True.",
    )
    revision_instructions: str = Field(
        default="",
        description="Clear, actionable instructions for the Writer agent to fix the flagged issues.",
    )


class ResearchGraphState(TypedDict):
    session_id: str
    user_query: str
    prior_memories: List[dict]
    plan: List[SubQuestionTask]
    findings: Annotated[List[SubTopicFinding], operator.add]
    draft_report: str
    initial_draft: str
    critique: CritiqueFeedback | None
    revision_count: int
    final_report: str
    all_source_urls: List[str]


class WorkerTaskState(TypedDict):
    session_id: str
    task: SubQuestionTask
