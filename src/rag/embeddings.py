"""
Embeddings provider.

Same branching pattern as src/llm.py: one function, one place to switch
between Gemini embeddings (cloud) and Ollama embeddings (local). Ingestion
and retrieval both call get_embedding_model(), so switching providers
here automatically applies everywhere embeddings are used.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from src import config


def get_embedding_model():
    """Return the embedding model used for both ingestion and query time.

    Important: ingestion and retrieval MUST use the same embedding model
    (and provider), otherwise vectors live in different spaces and
    similarity search breaks silently (no error, just bad/irrelevant
    results). If you switch LLM_PROVIDER, re-run ingestion so the vector
    store is rebuilt with the matching embedding model.
    """
    if config.LLM_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
