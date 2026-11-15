"""
Multi-query expansion.

A single debate topic often misses evidence that's phrased differently
in the source documents (e.g. "nuclear energy expansion" won't match a
chunk about "reactor construction overruns" by dense similarity alone).
This generates a handful of reformulations so retrieval covers more of
the topic's actual claim-space before fusion narrows it back down.
"""
import logging

from pydantic import BaseModel, Field

from src.llm import get_llm

logger = logging.getLogger(__name__)


class QueryVariants(BaseModel):
    queries: list[str] = Field(
        description="3-4 distinct reformulations of the original query, "
        "each targeting a different facet or phrasing of the topic."
    )


EXPANSION_SYSTEM_PROMPT = """You rewrite a debate proposition into several
search queries for a vector database, so retrieval covers multiple angles
of the topic (costs, risks, benefits, specific terminology) rather than
just the surface phrasing of the original question.

Rules:
- Produce 3 to 4 queries, each under 12 words.
- Each should target a DIFFERENT facet of the topic (not just a reword).
- Do not add opinions or answer the question -- only rewrite it as search queries.
"""


def generate_query_variants(topic: str, n: int = 4) -> list[str]:
    """Return the original topic plus up to n LLM-generated reformulations."""
    llm = get_llm(temperature=0.5)  # a bit more variety than the 0.3 default
    structured_llm = llm.with_structured_output(QueryVariants)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Debate topic: {topic}"},
            ]
        )
        variants = result.queries[:n]
    except Exception as exc:
        logger.warning("Query expansion failed, falling back to original query only: %s", exc)
        variants = []

    # Always include the raw topic itself -- expansion should ADD coverage,
    # never risk losing the direct match if the LLM call fails or drifts.
    all_queries = [topic] + [q for q in variants if q.strip() and q.strip() != topic]
    logger.info("Expanded '%s' into %d queries", topic, len(all_queries))
    return all_queries
