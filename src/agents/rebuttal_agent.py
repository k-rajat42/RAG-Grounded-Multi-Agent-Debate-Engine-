"""
Rebuttal stage.

Each side now sees the OPPOSING side's opening arguments and must respond
directly to them — not just restate their own position. This is what
turns two independent monologues into an actual debate.

Both rebuttal functions share the same shape (respond to opponent's
arguments, using the same research evidence) but are separate functions
with opposite-facing system prompts, mirroring the pro_agent/con_agent split.
"""
import logging

from src.llm import get_llm
from src.models.schemas import DebateSideOutput, RebuttalOutput, ResearchOutput

logger = logging.getLogger(__name__)

PRO_REBUTTAL_SYSTEM_PROMPT = """You are the PRO side of a structured, evidence-based debate,
now responding to the CON side's arguments.

STRICT RULES:
- You will be given the CON side's arguments. Respond DIRECTLY to each one you address —
  do not simply restate your own opening argument.
- Use the research evidence provided to support your response where possible.
- Do NOT fabricate statistics, studies, quotations, or sources.
- If a CON argument raises a genuine limitation not addressed by the evidence, acknowledge
  it honestly rather than dismissing it without support.
- For each rebuttal, clearly state: the opponent's claim you are responding to, your response,
  and the evidence/source backing your response (if any).

Return your answer using the required structured format only.
"""

CON_REBUTTAL_SYSTEM_PROMPT = """You are the CON side of a structured, evidence-based debate,
now responding to the PRO side's arguments.

STRICT RULES:
- You will be given the PRO side's arguments. Respond DIRECTLY to each one you address —
  do not simply restate your own opening argument.
- Use the research evidence provided to support your response where possible.
- Do NOT fabricate statistics, studies, quotations, or sources.
- If a PRO argument is well-supported by the evidence, acknowledge that rather than
  dismissing it without support.
- For each rebuttal, clearly state: the opponent's claim you are responding to, your response,
  and the evidence/source backing your response (if any).

Return your answer using the required structured format only.
"""


def _format_arguments(side: DebateSideOutput) -> str:
    if not side.arguments:
        return "(no arguments were made)"
    return "\n".join(
        f"- {item.argument}" + (f" [Source: {item.source}, Page: {item.page}]" if item.source else "")
        for item in side.arguments
    )


def _format_research(research: ResearchOutput) -> str:
    if not research.evidence:
        return "No evidence was retrieved from the knowledge base for this topic."
    return "\n".join(
        f"- Claim: {item.claim}\n  Evidence: {item.evidence}\n  Source: {item.source}, Page: {item.page}"
        for item in research.evidence
    )


def build_rebuttal_prompt(
    topic: str,
    own_side_label: str,
    opponent_side_label: str,
    opponent_arguments: DebateSideOutput,
    research: ResearchOutput,
) -> str:
    """Build the user-turn prompt for a rebuttal agent.

    Pure function (no LLM/network calls) so it's unit-testable without an API key.
    """
    return (
        f"Debate topic: {topic}\n\n"
        f"{opponent_side_label}'s arguments (to respond to):\n{_format_arguments(opponent_arguments)}\n\n"
        f"Research evidence available:\n{_format_research(research)}\n\n"
        f"As the {own_side_label} side, write a rebuttal directly addressing each of the "
        f"{opponent_side_label} arguments above."
    )


def run_pro_rebuttal(
    topic: str,
    con_arguments: DebateSideOutput,
    research: ResearchOutput,
) -> RebuttalOutput:
    """PRO responds to CON's opening arguments."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    logger.info("PRO rebuttal started for topic: %s", topic)

    llm = get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(RebuttalOutput)
    prompt = build_rebuttal_prompt(topic, "PRO", "CON", con_arguments, research)

    try:
        result: RebuttalOutput = structured_llm.invoke(
            [
                {"role": "system", "content": PRO_REBUTTAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("PRO rebuttal LLM call failed: %s", exc)
        raise RuntimeError(f"PRO rebuttal failed to generate structured output: {exc}") from exc

    logger.info("PRO rebuttal produced %d rebuttals", len(result.rebuttals))
    return result


def run_con_rebuttal(
    topic: str,
    pro_arguments: DebateSideOutput,
    research: ResearchOutput,
) -> RebuttalOutput:
    """CON responds to PRO's opening arguments."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    logger.info("CON rebuttal started for topic: %s", topic)

    llm = get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(RebuttalOutput)
    prompt = build_rebuttal_prompt(topic, "CON", "PRO", pro_arguments, research)

    try:
        result: RebuttalOutput = structured_llm.invoke(
            [
                {"role": "system", "content": CON_REBUTTAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("CON rebuttal LLM call failed: %s", exc)
        raise RuntimeError(f"CON rebuttal failed to generate structured output: {exc}") from exc

    logger.info("CON rebuttal produced %d rebuttals", len(result.rebuttals))
    return result
