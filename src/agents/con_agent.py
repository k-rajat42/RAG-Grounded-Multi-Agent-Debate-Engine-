"""
CON Debate Agent.

Mirrors pro_agent.py structurally (same evidence in, same schema out) but
argues the opposite side independently. Keeping PRO and CON symmetric in
structure — while letting their system prompts diverge in content — is
what makes the Judge's comparison later fair: both sides had the same
evidence and the same instructions to be evidence-grounded.
"""
import logging

from src.llm import get_llm
from src.models.schemas import DebateSideOutput, ResearchOutput

logger = logging.getLogger(__name__)

CON_SYSTEM_PROMPT = """You are the CON side of a structured, evidence-based debate.

Your job is to argue AGAINST the proposition using the research evidence provided.

STRICT RULES:
- Use the supplied research evidence. Prefer evidence-backed arguments over general reasoning.
- Do NOT fabricate statistics, studies, quotations, or sources.
- Identify limitations, risks, costs, counterexamples, and uncertainties where they are
  supported by the evidence given to you.
- Do NOT simply contradict what you imagine the PRO side might say without reasoning —
  build your own independent, evidence-grounded case against the proposition.
- You may reason from the evidence (inference), but clearly distinguish inference from
  facts that are directly stated in a source.
- Construct approximately 3 strong arguments.

Return your answer using the required structured format only.
"""


def build_con_prompt(topic: str, research: ResearchOutput) -> str:
    """Build the user-turn prompt for the CON agent.

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
        "Construct approximately 3 strong arguments AGAINST this proposition, "
        "grounded in the evidence above wherever possible."
    )


def run_con_agent(topic: str, research: ResearchOutput) -> DebateSideOutput:
    """Generate the CON side's opening arguments."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    logger.info("CON agent started for topic: %s", topic)

    llm = get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(DebateSideOutput)
    prompt = build_con_prompt(topic, research)

    try:
        result: DebateSideOutput = structured_llm.invoke(
            [
                {"role": "system", "content": CON_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.error("CON agent LLM call failed: %s", exc)
        raise RuntimeError(f"CON agent failed to generate structured output: {exc}") from exc

    logger.info("CON agent produced %d arguments", len(result.arguments))
    return result
