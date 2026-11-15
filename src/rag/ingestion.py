"""
PDF ingestion pipeline: PDF -> pages -> chunks -> embeddings -> ChromaDB.

This is the "indexing" half of RAG. It runs once (or whenever documents
change), not on every debate request. The output is a persisted Chroma
collection on disk that the retriever (retriever.py) queries later.
"""
import logging
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def load_pdfs(documents_dir: str = config.DOCUMENTS_DIR) -> list[Document]:
    """Load every PDF in documents_dir into LangChain Document objects (one per page).

    Each Document already carries `source` (file path) and `page` in its
    metadata because PyPDFLoader sets these automatically.
    """
    documents_dir_path = Path(documents_dir)
    if not documents_dir_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    pdf_paths = sorted(documents_dir_path.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(
            f"No PDF files found in {documents_dir}. Add at least one PDF to build a knowledge base."
        )

    all_pages: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            # Normalize metadata: keep a clean "document" name separate from full path.
            for page in pages:
                page.metadata["document"] = pdf_path.name
                page.metadata["source"] = pdf_path.name
                # PyPDFLoader's "page" metadata is 0-indexed; make it human-friendly.
                if "page" in page.metadata:
                    page.metadata["page"] = page.metadata["page"] + 1
            all_pages.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", pdf_path.name, exc)
            raise RuntimeError(f"Failed to load PDF '{pdf_path.name}': {exc}") from exc

    return all_pages


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split page-level documents into smaller overlapping chunks.

    Chunking exists because: (1) LLM context windows and embedding models
    have size limits, and (2) smaller, focused chunks retrieve more
    precisely than whole pages. Overlap prevents cutting a sentence/fact
    in half between two chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks


def build_vector_store(
    chunks: list[Document],
    persist_directory: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> Chroma:
    """Embed chunks and persist them into a Chroma vector store on disk.

    This is the step where text becomes vectors. Chroma stores both the
    vectors and the original chunk text + metadata together, so retrieval
    can return human-readable evidence, not just numbers.
    """
    if not chunks:
        raise ValueError("No chunks to index. Ingestion produced zero chunks.")

    embedding_model = get_embedding_model()
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception as exc:
        logger.error("Vector store build failed: %s", exc)
        raise RuntimeError(
            f"Failed to embed/persist chunks to ChromaDB. "
            f"Check GOOGLE_API_KEY and network access. Original error: {exc}"
        ) from exc
    logger.info(
        "Persisted %d chunks to Chroma collection '%s' at %s",
        len(chunks),
        collection_name,
        persist_directory,
    )
    return vector_store


def ingest_documents(documents_dir: str = config.DOCUMENTS_DIR) -> Chroma:
    """Run the full ingestion pipeline: load -> split -> embed -> persist."""
    pages = load_pdfs(documents_dir)
    chunks = split_documents(pages)
    return build_vector_store(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
