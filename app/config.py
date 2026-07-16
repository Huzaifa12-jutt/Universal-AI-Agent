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
    # Gemini / LLM settings
    # ------------------------------------------------------------------
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))

    # ------------------------------------------------------------------
    # PDF RAG (Retrieval-Augmented Generation) settings
    # Groq hosts the chat model (OpenAI-compatible endpoint)
    # HuggingFace sentence-transformers handles embeddings locally
    # ------------------------------------------------------------------
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    
    # ✅ CHANGED: Use HuggingFace sentence-transformers (compatible with Python 3.14)
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    
    # ✅ CHANGED: Increased from 4 to 10 for better context retrieval
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