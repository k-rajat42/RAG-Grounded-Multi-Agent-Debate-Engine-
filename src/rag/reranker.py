"""
Cross-encoder reranking.

RRF (and the dense/BM25 retrievers feeding it) score query and chunk
independently, then compare the results -- fast, but approximate. A
cross-encoder scores the (query, chunk) PAIR jointly through one
transformer forward pass, which is far more accurate at judging actual
relevance -- but too slow to run over the whole corpus. So this sits
as the last step: take RRF's top ~20 candidates, rerank just those,
keep the final top_k.
"""
import logging

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_cache: CrossEncoder | None = None

# Small, fast, well-regarded for this exact "narrow a shortlist" use case.
# Runs on CPU fine at this scale (dozens of pairs, not thousands).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _reranker_cache
    if _reranker_cache is None:
        logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        _reranker_cache = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache


def rerank(query: str, candidates: list[Document], top_k: int) -> list[Document]:
    """Rerank candidates by (query, chunk) relevance, return the top_k.

    Falls back to returning candidates[:top_k] unchanged if the model
    fails to load or score -- reranking is a quality improvement, not
    something retrieval should hard-depend on (same resilience pattern
    as the rest of the hybrid pipeline).
    """
    if not candidates:
        return []

    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
    except Exception as exc:
        logger.warning("Reranking failed, falling back to RRF order: %s", exc)
        return candidates[:top_k]

    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    reranked = [doc for doc, _ in scored[:top_k]]
    logger.info("Reranked %d candidates down to top %d", len(candidates), len(reranked))
    return reranked
