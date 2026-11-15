"""
BM25 sparse retrieval, built from the same chunks ChromaDB stores.

Dense embeddings alone can blur exact terms that matter in debate
evidence (e.g. "120 percent", "Vogtle", "Zaporizhzhia") -- BM25 catches
those lexical matches that similarity search sometimes smooths over.
Rebuilt in-memory from Chroma's stored documents rather than persisted
separately, so there's only one source of truth for what's ingested.

Two settings below fix LangChain's BM25Retriever defaults:
  1. Tokenization: the default is text.split(), which keeps case and
     attached punctuation ("Costs." != "costs"). We lowercase and split
     on word characters instead.
  2. Result count: the default k is 4. We return more candidates so the
     caller's own slice (n_dense_per_query) is what controls the count.
"""
import logging
import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_bm25_cache: BM25Retriever | None = None

# Return more candidates than any caller will slice off, so that
# retriever.py's [:n_dense_per_query] actually controls the count.
BM25_CANDIDATES = 10

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens.

    Applied by BM25Retriever to every chunk at index time AND to every
    query at search time, so both sides are normalized identically.
    \\w+ keeps numbers intact ("120"), which matters because debate
    evidence is full of statistics.
    """
    return _TOKEN_RE.findall(text.lower())


def get_bm25_retriever(force_rebuild: bool = False) -> BM25Retriever:
    """Build (or return a cached) BM25 retriever over all chunks in Chroma.

    Cached at module level since rebuilding the BM25 index from scratch
    on every debate request would be wasteful -- it only needs to change
    when ingestion adds new documents, so call with force_rebuild=True
    after re-ingesting.

    Imports get_vector_store() lazily (inside the function) to avoid a
    circular import, since retriever.py imports from this module too.
    """
    global _bm25_cache
    if _bm25_cache is not None and not force_rebuild:
        return _bm25_cache

    from src.rag.retriever import get_vector_store

    store = get_vector_store()
    raw = store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    if not docs:
        logger.warning("BM25 index built from 0 documents -- has ingestion run yet?")

    _bm25_cache = BM25Retriever.from_documents(
        docs,
        preprocess_func=tokenize,
        k=BM25_CANDIDATES,
    )
    return _bm25_cache