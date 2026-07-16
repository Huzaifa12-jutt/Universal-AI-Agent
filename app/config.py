"""
Central configuration for the AI Utility Agent.

Every tunable value (API keys, model names, timeouts, limits) lives here.
Nothing should be hardcoded anywhere else in the codebase - if you need a
new constant, add it to this file and read it from `settings`.
"""

import os
import logging
from dotenv import load_dotenv

# Load variables from a local .env file (ignored in git, see .env.example)
load_dotenv()


class Settings:
    """Strongly-typed access to environment configuration."""

    # ------------------------------------------------------------------
    # App metadata
    # ------------------------------------------------------------------
    APP_NAME: str = os.getenv("APP_NAME", "AI Utility Agent")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ------------------------------------------------------------------
    # API Keys (never hardcode these - always via environment variables)
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ------------------------------------------------------------------
    # LLM Settings (Choose which provider to use)
    # ------------------------------------------------------------------
    # Which model to use: "gemini" or "groq"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    
    # --- Gemini Settings ---
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))
    
    # --- Groq Settings ---
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # ------------------------------------------------------------------
    # PDF RAG (Retrieval-Augmented Generation) settings
    # Groq hosts the chat model (OpenAI-compatible endpoint)
    # FastEmbed handles embeddings locally (lightweight, no extra API key)
    # ------------------------------------------------------------------
    # ✅ FIXED: FastEmbed-compatible model only
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
    RAG_TEMPERATURE: float = float(os.getenv("RAG_TEMPERATURE", "0.2"))
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "20"))
    MAX_PDFS: int = int(os.getenv("MAX_PDFS", "10"))

    # ------------------------------------------------------------------
    # External API URLs
    # ------------------------------------------------------------------
    WEATHER_BASE_URL: str = os.getenv(
        "WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5/weather"
    )

    # ------------------------------------------------------------------
    # Tool behaviour
    # ------------------------------------------------------------------
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    WIKIPEDIA_SENTENCES: int = int(os.getenv("WIKIPEDIA_SENTENCES", "3"))

    # ------------------------------------------------------------------
    # Backend URL used by the Streamlit frontend
    # (localhost for dev, your Render URL in production)
    # ------------------------------------------------------------------
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    # ------------------------------------------------------------------
    # CORS - which frontend origins are allowed to call this API
    # ------------------------------------------------------------------
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]


settings = Settings()


def configure_logging() -> None:
    """Configure application-wide logging. Call once on startup."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------------------------------------------------------
# Helper function to get LLM based on provider
# ------------------------------------------------------------------
def get_llm_config():
    """Returns the appropriate LLM configuration based on provider."""
    if settings.LLM_PROVIDER == "groq":
        return {
            "provider": "groq",
            "api_key": settings.GROQ_API_KEY,
            "model": settings.GROQ_MODEL,
            "base_url": settings.GROQ_BASE_URL,
        }
    else:
        return {
            "provider": "gemini",
            "api_key": settings.GEMINI_API_KEY,
            "model": settings.GEMINI_MODEL,
        }