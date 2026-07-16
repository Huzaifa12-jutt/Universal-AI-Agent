"""DuckDuckGo web search tool with LLM-powered summarisation."""

import logging
from app.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)

# `duckduckgo-search` was renamed to `ddgs` on PyPI - support both so this
# keeps working regardless of which one ends up installed.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore


class SearchTool(BaseTool):
    """Searches the web via DuckDuckGo and returns a short, summarised answer."""

    name = "search"
    description = "Searches the web for real-time information (news, facts, current events)."

    def run(self, query: str) -> dict:
        query = (query or "").strip()

        if not query:
            return {"error": "Please provide something to search for."}

        logger.info("Search tool called with query=%s", query)

        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(query, max_results=settings.SEARCH_MAX_RESULTS)
                )

            if not results:
                return {
                    "query": query,
                    "summary": "No results found for this query.",
                    "sources": [],
                }

            sources = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]

            summary = self._summarise(query, sources)

            return {
                "query": query,
                "summary": summary,
                "sources": sources,
            }

        except Exception as e:  # noqa: BLE001
            logger.exception("Search tool failed")
            return {"error": f"Search failed: {e}"}

    @staticmethod
    def _summarise(query: str, sources: list[dict]) -> str:
        """Use Gemini to turn raw search snippets into a short answer.
        Falls back to a plain snippet join if the LLM call fails.
        """
        from app.services.llm import ask_llm  # local import avoids circular import

        context = "\n".join(
            f"- {s['title']}: {s['snippet']}" for s in sources if s.get("snippet")
        )

        prompt = (
            "You are a helpful research assistant. Using ONLY the search results "
            "below, write a concise 2-4 sentence answer to the user's question. "
            "Do not make anything up beyond what is given.\n\n"
            f"Question: {query}\n\nSearch results:\n{context}\n\nAnswer:"
        )

        try:
            return ask_llm(prompt)
        except Exception:  # noqa: BLE001
            logger.warning("LLM summarisation failed, falling back to raw snippets")
            return " ".join(s["snippet"] for s in sources[:3] if s.get("snippet"))
