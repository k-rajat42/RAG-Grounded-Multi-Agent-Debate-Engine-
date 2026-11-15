"""
LLM provider shared by all agents.

Branches on config.LLM_PROVIDER so every agent calls get_llm() without
knowing or caring whether requests go to Gemini (cloud, needs API key)
or Ollama (local, free, no API key -- good for testing/development).
Adding a third provider means adding one more branch here, not touching
any agent file.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src import config


def get_llm(temperature: float = 0.3):
    """Return the chat LLM used by all debate agents.

    Low-ish temperature (0.3 default) favors consistent, grounded
    responses over creative variation -- appropriate for evidence-based
    argumentation rather than free-form writing.
    """
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=config.OLLAMA_LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )
