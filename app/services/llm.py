"""Thin wrapper around the Gemini API (google-genai SDK)."""

import logging
from google import genai
from app.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily create a single shared Gemini client (avoids import-time crashes
    when GEMINI_API_KEY is missing, e.g. during tests)."""
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def ask_llm(prompt: str) -> str:
    """Send a prompt to Gemini and return the plain-text response.

    Never raises to the caller - on any failure it returns a readable
    error string so the rest of the app can keep working.
    """
    if not prompt or not prompt.strip():
        return "Please ask me something!"

    logger.info("Gemini request: %.80s...", prompt)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text if text else "I couldn't generate a response for that. Try rephrasing?"

    except RuntimeError as e:
        logger.error("Gemini config error: %s", e)
        return f"Configuration error: {e}"

    except Exception as e:  # noqa: BLE001
        logger.exception("Gemini request failed")
        return f"Sorry, I ran into an error talking to Gemini: {e}"


def ask_llm_stream(prompt: str):
    """Same as ask_llm, but yields text chunks as they arrive for a
    ChatGPT-style typing effect in the UI. Never raises - on failure it
    yields a single readable error chunk instead.
    """
    if not prompt or not prompt.strip():
        yield "Please ask me something!"
        return

    logger.info("Gemini streaming request: %.80s...", prompt)

    try:
        client = _get_client()
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
        logger.error("Gemini config error: %s", e)
        yield f"Configuration error: {e}"

    except Exception as e:  # noqa: BLE001
        logger.exception("Gemini streaming request failed")
        yield f"Sorry, I ran into an error talking to Gemini: {e}"
