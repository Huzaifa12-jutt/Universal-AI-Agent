"""
Thin wrapper around LLM APIs (Gemini and Groq).
Supports both providers - can switch via config.
"""

import logging
from typing import Generator

from app.config import settings, get_llm_config

logger = logging.getLogger(__name__)

# Lazy-loaded clients
_gemini_client = None
_groq_client = None


def _get_gemini_client():
    """Lazily create a single shared Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
        from google import genai
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _get_groq_client():
    """Lazily create a single shared Groq client (LangChain)."""
    global _groq_client
    if _groq_client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
        from langchain_groq import ChatGroq
        _groq_client = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.7,
        )
    return _groq_client


def _get_llm_client():
    """Get the appropriate LLM client based on provider setting."""
    config = get_llm_config()
    if config["provider"] == "groq":
        return _get_groq_client(), "groq"
    else:
        return _get_gemini_client(), "gemini"


def ask_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the plain-text response.

    Never raises to the caller - on any failure it returns a readable
    error string so the rest of the app can keep working.
    """
    if not prompt or not prompt.strip():
        return "Please ask me something!"

    logger.info("LLM request (provider=%s): %.80s...", settings.LLM_PROVIDER, prompt)

    try:
        client, provider = _get_llm_client()
        
        if provider == "groq":
            # Groq uses LangChain's invoke method
            response = client.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            # Gemini uses google-genai SDK
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            text = (response.text or "").strip()
        
        return text if text else "I couldn't generate a response for that. Try rephrasing?"

    except RuntimeError as e:
        logger.error("LLM config error: %s", e)
        return f"Configuration error: {e}"

    except Exception as e:
        logger.exception("LLM request failed")
        return f"Sorry, I ran into an error talking to the LLM: {e}"


def ask_llm_stream(prompt: str) -> Generator[str, None, None]:
    """Same as ask_llm, but yields text chunks as they arrive for a
    ChatGPT-style typing effect in the UI. Never raises - on failure it
    yields a single readable error chunk instead.
    """
    if not prompt or not prompt.strip():
        yield "Please ask me something!"
        return

    logger.info("LLM streaming request (provider=%s): %.80s...", settings.LLM_PROVIDER, prompt)

    try:
        client, provider = _get_llm_client()
        
        if provider == "groq":
            # Groq streaming via LangChain
            for chunk in client.stream(prompt):
                if chunk.content:
                    yield chunk.content
        else:
            # Gemini streaming via google-genai SDK
            stream = client.models.generate_content_stream(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            got_any = False
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    got_any = True
                    yield text

            if not got_any:
                yield "I couldn't generate a response for that. Try rephrasing?"

    except RuntimeError as e:
        logger.error("LLM config error: %s", e)
        yield f"Configuration error: {e}"

    except Exception as e:
        logger.exception("LLM streaming request failed")
        yield f"Sorry, I ran into an error talking to the LLM: {e}"