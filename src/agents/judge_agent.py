"""
Judge/Analysis Agent.

Responsibility: produce a structured, evidence-focused analysis of the
full debate (research + both openings + both rebuttals). Per Section 12,
this agent explicitly does NOT declare a winner or produce a numeric
score — it evaluates evidence usage, source grounding, rebuttal quality,
unsupported claims, disagreements, and open questions.
"""
import logging

from src.llm import get_llm
from src.models.schemas import (
    JudgeOutput,
    ResearchOutput,
    DebateSideOutput,
    RebuttalOutput,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are the Judge/Analysis Agent for a structured, evidence-based debate.

You do NOT declare a winner and you do NOT produce a numeric score. Your job is to
produce an objective, structured analysis covering:

- Evidence usage: did each side actually use the available research evidence?
- Source grounding: are factual claims connected to real sources from the evidence?
- Rebuttal quality: did each side genuinely address the opposing arguments, or just restate their own position?
- Unsupported claims: identify any claims made by either side that are NOT backed by the
  supplied research evidence (list them separately for PRO and CON).
- Key disagreements: where do the two sides genuinely disagree, based on the evidence?
- Unresolved questions: what questions can't be settled with the current evidence?

Be fair and evenhanded to both sides. Base your analysis only on what was actually
argued and the evidence actually supplied — do not introduce outside knowledge or
your own opinion on the proposition.

Return your answer using the required structured format only.
"""


def _format_research(research: ResearchOutput) -> str:
    if not research.evidence:
        return "No evidence was retrieved from the knowledge base for this topic."
    return "\n".join(
        f"- Claim: {item.claim}\n  Evidence: {item.evidence}\n  Source: {item.source}, Page: {item.page}"
        for item in research.evidence
    )


def _format_arguments(side: DebateSideOutput) -> str:
    if not side.arguments:
        return "(no arguments were made)"
    return "\n".join(
        f"- {item.argument}" + (f" [Source: {item.source}, Page: {item.page}]" if item.source else "")
        for item in side.arguments
    )


def _format_rebuttals(rebuttal: RebuttalOutput) -> str:
    if not rebuttal.rebuttals:
        return "(no rebuttals were made)"
    return "\n".join(
        f"- Responding to: \"{item.opponent_claim}\"\n  Response: {item.response}"
        + (f" [Source: {item.source}, Page: {item.page}]" if item.source else "")
        for item in rebuttal.rebuttals
    )


def build_judge_prompt(
    topic: str,
    research: ResearchOutput,
    pro_argument: DebateSideOutput,
    con_argument: DebateSideOutput,
    pro_rebuttal: RebuttalOutput,
    con_rebuttal: RebuttalOutput,
) -> str:
    """Build the user-turn prompt for the judge agent.

    Pure function (no LLM/network calls) so it's unit-testable without an API key.
    """
    return (
        f"Debate topic: {topic}\n\n"
        f"Research evidence available to both sides:\n{_format_research(research)}\n\n"
        f"PRO opening arguments:\n{_format_arguments(pro_argument)}\n\n"
        f"CON opening arguments:\n{_format_arguments(con_argument)}\n\n"
        f"PRO rebuttal (responding to CON):\n{_format_rebuttals(pro_rebuttal)}\n\n"
        f"CON rebuttal (responding to PRO):\n{_format_rebuttals(con_rebuttal)}\n\n"
        "Analyze this debate as instructed. Do not declare a winner or assign a score."
    )


def run_judge_agent(
    topic: str,
    research: ResearchOutput,
    pro_argument: DebateSideOutput,
    con_argument: DebateSideOutput,
    pro_rebuttal: RebuttalOutput,
    con_rebuttal: RebuttalOutput,
) -> JudgeOutput:
    """Produce the final structured analysis of the full debate."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    logger.info("Judge agent started for topic: %s", topic)

    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(JudgeOutput)
    prompt = build_judge_prompt(topic, research, pro_argument, con_argument, pro_rebuttal, con_rebuttal)

    try:
        result: JudgeOutput = structured_llm.invoke(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("Judge agent LLM call failed: %s", exc)
        raise RuntimeError(f"Judge agent failed to generate structured output: {exc}") from exc

    logger.info("Judge agent analysis complete")
    return result
