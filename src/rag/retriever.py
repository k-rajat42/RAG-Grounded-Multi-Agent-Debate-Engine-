"""
Retrieval layer.

Separated from ingestion.py because retrieval happens on every debate
request (query-time), while ingestion happens once (index-time). This
module only ever *reads* from the persisted Chroma collection.
"""
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.rag.embeddings import get_embedding_model
from src.rag.query_expansion import generate_query_variants
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.bm25_index import get_bm25_retriever
from src.rag.reranker import rerank

logger = logging.getLogger(__name__)


def get_vector_store(
    persist_directory: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> Chroma:
    """Load the existing persisted Chroma collection (does not re-ingest).

    Uses the same embedding model as ingestion — required so query vectors
    and stored vectors live in the same space.
    """
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )


def retrieve_evidence(
    query: str,
    top_k: int | None = None,
    vector_store: Chroma | None = None,
) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a debate topic/query.

    For the MVP, the debate topic itself IS the query (no query rewriting
    or decomposition — that's a Phase 3 extension). Kept deliberately
    simple per the "do not implement hybrid search" requirement.

    top_k defaults to config.TOP_K, read at CALL time (not bound as a
    function default) so that changing config.TOP_K at runtime -- e.g.
    from a Streamlit UI control -- actually takes effect. A default
    bound at function-definition time would freeze the value the moment
    this module is first imported.
    """
    if not query or not query.strip():
        raise ValueError("Query/topic must not be empty.")

    resolved_top_k = top_k if top_k is not None else config.TOP_K
    store = vector_store or get_vector_store()

    try:
        results = store.similarity_search(query, k=resolved_top_k)
    except Exception as exc:
        logger.error("Retrieval failed for query '%s': %s", query, exc)
        raise RuntimeError(
            f"Vector search failed. Has the knowledge base been ingested? Original error: {exc}"
        ) from exc

    if not results:
        logger.warning("Retrieval returned 0 results for query: %s", query)
    else:
        logger.info("Retrieved %d chunks for query: %s", len(results), query)

    return results


def retrieve_evidence_hybrid(
    query: str,
    top_k: int | None = None,
    vector_store: Chroma | None = None,
    n_dense_per_query: int = 5,
    rerank_candidate_pool: int = 20,
) -> list[Document]:
    """Multi-query + hybrid (dense + BM25) retrieval, fused via RRF and
    reranked by a cross-encoder.

    Phase 3 upgrade over retrieve_evidence(): expands the topic into
    several queries, retrieves both dense and sparse matches for each,
    fuses every ranked list into a top-N shortlist via RRF, then
    reranks that shortlist down to top_k with a cross-encoder that
    scores the (query, chunk) pair jointly -- more accurate than the
    independent dense/sparse scores that fed RRF, but too slow to run
    over the whole corpus, hence only applied to the fused shortlist.

    Every stage degrades independently rather than raising: query
    expansion falls back to the raw topic, BM25 falls back to dense-only,
    and a complete retrieval failure falls back to plain
    retrieve_evidence() (the original MVP path) rather than propagating
    an exception up into research_node, which is a single point of
    failure for the whole (parallel) debate graph.
    """
    if not query or not query.strip():
        raise ValueError("Query/topic must not be empty.")

    resolved_top_k = top_k if top_k is not None else config.TOP_K
    store = vector_store or get_vector_store()

    try:
        bm25 = get_bm25_retriever()
    except Exception as exc:
        logger.warning("BM25 index unavailable, continuing with dense-only: %s", exc)
        bm25 = None

    try:
        queries = generate_query_variants(query)
    except Exception as exc:
        logger.warning("Query expansion failed entirely, using raw topic only: %s", exc)
        queries = [query]

    ranked_lists: list[list[Document]] = []
    for q in queries:
        try:
            ranked_lists.append(store.similarity_search(q, k=n_dense_per_query))
        except Exception as exc:
            logger.warning("Dense retrieval failed for '%s': %s", q, exc)
        if bm25:
            try:
                ranked_lists.append(bm25.invoke(q)[:n_dense_per_query])
            except Exception as exc:
                logger.warning("BM25 retrieval failed for '%s': %s", q, exc)

    if not ranked_lists:
        # Last resort: fall back to the original single-query dense search
        # rather than raising and taking down research_node, and through
        # it, the whole graph run.
        logger.error("Hybrid retrieval fully failed, falling back to plain retrieve_evidence")
        return retrieve_evidence(query, top_k=resolved_top_k, vector_store=store)

    fused = reciprocal_rank_fusion(ranked_lists, top_k=rerank_candidate_pool)
    reranked = rerank(query, fused, top_k=resolved_top_k)

    logger.info(
        "Hybrid retrieval: %d queries -> %d RRF candidates -> %d reranked for topic: %s",
        len(queries), len(fused), len(reranked), query,
    )
    return reranked


def format_evidence_for_prompt(chunks: list[Document]) -> str:
    """Render retrieved chunks as a numbered, source-tagged block for LLM prompts.

    Keeping this format consistent everywhere it's used is what lets
    downstream agents reliably cite "Source: X, Page: Y" instead of
    inventing citations.
    """
    if not chunks:
        return "No evidence was retrieved from the knowledge base."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("document", "unknown")
        page = chunk.metadata.get("page", "unknown")
        lines.append(f"[{i}] Source: {source}, Page: {page}\n{chunk.page_content.strip()}")
    return "\n\n".join(lines)
