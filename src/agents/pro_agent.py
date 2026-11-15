"""
PRO Debate Agent.

Responsibility: given the debate topic and the Research Agent's organized
evidence, construct ~3 strong arguments IN FAVOR of the proposition.
Does not see the CON agent's output — each side argues independently
first, so neither side's opening position is shaped by the other
(rebuttal.py handles the stage where they respond to each other).
"""
import logging

from src.llm import get_llm
from src.models.schemas import DebateSideOutput, ResearchOutput

logger = logging.getLogger(__name__)

PRO_SYSTEM_PROMPT = """You are the PRO side of a structured, evidence-based debate.

Your job is to argue IN FAVOR of the proposition using the research evidence provided.

STRICT RULES:
- Use the supplied research evidence. Prefer evidence-backed arguments over general reasoning.
- Do NOT fabricate statistics, studies, quotations, or sources.
- For every factual claim, identify the supporting source from the evidence given to you.
- You may reason from the evidence (inference), but clearly distinguish inference from
  facts that are directly stated in a source.
- If the evidence does not support a strong PRO case, make the best evidence-grounded
  argument you honestly can rather than overstating it.
- Construct approximately 3 strong arguments.

Return your answer using the required structured format only.
"""


def build_pro_prompt(topic: str, research: ResearchOutput) -> str:
    """Build the user-turn prompt for the PRO agent.

    Pure function (no LLM/network calls) so it's unit-testable without an API key.
    """
    if not research.evidence:
        evidence_block = "No evidence was retrieved from the knowledge base for this topic."
    else:
        evidence_block = "\n".join(
            f"- Claim: {item.claim}\n  Evidence: {item.evidence}\n  Source: {item.source}, Page: {item.page}"
            for item in research.evidence
        )

    return (
        f"Debate topic: {topic}\n\n"
        f"Research evidence available:\n{evidence_block}\n\n"
        "Construct approximately 3 strong arguments IN FAVOR of this proposition, "
        "grounded in the evidence above wherever possible."
    )


def run_pro_agent(topic: str, research: ResearchOutput) -> DebateSideOutput:
    """Generate the PRO side's opening arguments."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    logger.info("PRO agent started for topic: %s", topic)

    llm = get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(DebateSideOutput)
    prompt = build_pro_prompt(topic, research)

    try:
        result: DebateSideOutput = structured_llm.invoke(
            [
                {"role": "system", "content": PRO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("PRO agent LLM call failed: %s", exc)
        raise RuntimeError(f"PRO agent failed to generate structured output: {exc}") from exc

    logger.info("PRO agent produced %d arguments", len(result.arguments))
    return result
