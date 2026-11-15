"""
Central LangGraph state.

This is the single object that flows through every node in the debate
graph. Each node reads the fields it needs and writes the fields it
produces — nodes never share a random/untyped dict, which is what makes
the graph explicit and debuggable.

Using TypedDict (rather than a plain dict) means IDEs/type-checkers catch
typos like state["pro_argument"] vs state["pro_arguments"] before runtime.
"""
from typing import TypedDict

from src.models.schemas import ResearchOutput, DebateSideOutput, RebuttalOutput, JudgeOutput


class DebateState(TypedDict, total=False):
    """State object passed between all nodes in the debate graph.

    total=False means fields are optional at the type level — early in
    the graph (e.g. right after START) most fields haven't been
    populated yet. Nodes should treat missing fields defensively.
    """

    topic: str
    top_k: int  # optional override for number of evidence chunks to retrieve

    research: ResearchOutput

    pro_argument: DebateSideOutput
    con_argument: DebateSideOutput

    pro_rebuttal: RebuttalOutput
    con_rebuttal: RebuttalOutput

    judge_analysis: JudgeOutput

    # Flat list of "Document Name, Page N" strings for the UI's final
    # sources section, deduplicated across all evidence used.
    sources: list[str]

    # Populated only if a node raises a handled error, so the graph can
    # end gracefully instead of crashing, and the UI can show a message.
    error: str
