"""
Research/RAG Agent.

Responsibility: take the debate topic, retrieve evidence from the vector
store, and organize it into structured EvidenceItems that PRO/CON agents
will build arguments from.

This agent does NOT argue for or against anything — it only organizes
what the knowledge base actually contains, distinguishing evidence from
interpretation, and refusing to invent sources.
"""
import logging

from src.llm import get_llm
from src.models.schemas import ResearchOutput
from src.rag.retriever import retrieve_evidence_hybrid, format_evidence_for_prompt

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent for a structured debate system.

You will be given a debate topic and a set of evidence chunks retrieved
from a knowledge base. Your job is to organize this evidence into clear,
attributable evidence items.

STRICT RULES:
- Do not fabricate sources. Only use the sources given to you below.
- Do not invent citations, statistics, or page numbers that are not present in the evidence provided.
- Only claim that something is supported when the retrieved evidence actually supports it.
- Clearly distinguish evidence (what the source states) from interpretation (your summary of it).
- If the evidence is thin or irrelevant to the topic, say so rather than padding with invented claims.
- Every evidence item's "source" and "page" must come directly from the evidence chunks provided \
(the [n] Source: ..., Page: ... labels). Never guess a page number.
"""


def build_research_prompt(topic: str, evidence_text: str) -> str:
    """Build the user-turn prompt for the research agent.

    Kept as a pure function (no LLM/network calls) so it can be unit
    tested independently of any API key.
    """
    return (
        f"Debate topic: {topic}\n\n"
        f"Retrieved evidence chunks:\n{evidence_text}\n\n"
        "Organize this evidence into a list of evidence items. Each item should have "
        "a short claim, the supporting evidence text, and the exact source/page it came from."
    )


def run_research_agent(topic: str, top_k: int | None = None) -> ResearchOutput:
    """Retrieve evidence for the topic and organize it via the LLM into structured output."""
    logger.info("Research agent started for topic: %s", topic)

    kwargs = {"top_k": top_k} if top_k is not None else {}
    try:
        chunks = retrieve_evidence_hybrid(topic, **kwargs)
    except Exception as exc:
        logger.error("All retrieval paths failed for topic '%s': %s", topic, exc)
        chunks = []

    evidence_text = format_evidence_for_prompt(chunks)

    if not chunks:
        logger.warning("No evidence retrieved; returning empty research output.")
        return ResearchOutput(evidence=[])

    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchOutput)
    prompt = build_research_prompt(topic, evidence_text)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("Research agent LLM call failed: %s", exc)
        raise RuntimeError(f"Research agent failed to generate structured output: {exc}") from exc

    logger.info("Research agent produced %d evidence items", len(result.evidence))
    return result
