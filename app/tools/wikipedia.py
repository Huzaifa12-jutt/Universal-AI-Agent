"""Wikipedia tool - returns a concise summary for a topic."""

import logging
import wikipedia
from app.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WikipediaTool(BaseTool):
    """Searches Wikipedia and returns a concise summary of a topic."""

    name = "wikipedia"
    description = "Looks up a topic on Wikipedia and returns a short summary."

    def run(self, query: str) -> dict:
        query = (query or "").strip()

        if not query:
            return {"error": "Please provide a topic to look up."}

        logger.info("Wikipedia tool called with query=%s", query)

        try:
            summary = wikipedia.summary(
                query, sentences=settings.WIKIPEDIA_SENTENCES, auto_suggest=True
            )
            page = wikipedia.page(query, auto_suggest=True)

            return {
                "title": page.title,
                "summary": summary,
                "url": page.url,
            }

        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:5]
            return {
                "error": f"'{query}' is ambiguous. Did you mean one of: {', '.join(options)}?",
                "options": options,
            }

        except wikipedia.exceptions.PageError:
            return {"error": f"No Wikipedia page found for '{query}'."}

        except Exception as e:  # noqa: BLE001
            logger.exception("Wikipedia tool failed")
            return {"error": f"Wikipedia lookup failed: {e}"}
