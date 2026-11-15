"""
Central configuration for AI Debate Arena.

Loads settings from environment variables (via .env) so that no secrets
or "magic numbers" are hard-coded across the codebase. Every other module
should import from here rather than calling os.getenv() directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "ollama"

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")



# Ollama (local, no API key needed). Used only when LLM_PROVIDER=ollama.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# --- RAG / Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

# --- Paths ---
DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "data/documents")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "debate_evidence")


def validate_config() -> None:
    """Raise a clear error early if required config is missing.

    Called at app startup so the user gets a helpful message instead of
    a cryptic failure deep inside an LLM call.
    """
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama to run fully locally instead."
        )
    if LLM_PROVIDER not in ("gemini", "ollama"):
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'ollama'.")
