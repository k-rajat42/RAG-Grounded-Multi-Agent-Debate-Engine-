"""
Reciprocal Rank Fusion (RRF).

Combines several ranked lists (from different queries and/or different
retrievers -- dense + BM25) into one ranking, without needing comparable
similarity scores across them. Standard formula: score(doc) = sum over
every list it appears in of 1 / (k + rank_in_that_list).
"""
from collections import defaultdict

from langchain_core.documents import Document


def _doc_key(doc: Document) -> tuple:
    """Stable identity for a chunk, since the same chunk can come back
    from multiple queries/retrievers and needs to be recognized as 'the
    same document' rather than counted as separate hits."""
    return (doc.metadata.get("document"), doc.metadata.get("page"), doc.page_content[:100])


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Document]:
    """Fuse multiple ranked lists of Documents into one ranking.

    k=60 is the standard RRF constant from the original paper -- it
    dampens the influence of any single high rank so one retriever
    can't dominate just by ranking something #1.
    """
    scores: dict[tuple, float] = defaultdict(float)
    doc_lookup: dict[tuple, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (k + rank)
            doc_lookup[key] = doc  # first occurrence wins; content is identical

    fused_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    fused_docs = [doc_lookup[key] for key in fused_keys]

    return fused_docs[:top_k] if top_k else fused_docs
