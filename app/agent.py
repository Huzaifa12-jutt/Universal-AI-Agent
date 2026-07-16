"""
The Agent is a lightweight rule-based router: it looks at the user's
message and decides which tool should handle it. This keeps the project
fast, cheap (no extra LLM call just to pick a tool) and easy to follow
for beginners. If nothing matches, the message falls through to Gemini.
"""

import re
import logging
from app.services.llm import ask_llm, ask_llm_stream
from app.tools.registry import TOOLS
from app.rag.manager import rag_manager

logger = logging.getLogger(__name__)

# --- Trigger keywords for each tool -----------------------------------
_PDF_KEYWORDS = (
    "pdf", "document", "the file", "uploaded file", "in the paper",
    "according to the paper", "summarize this", "summarize the document",
    "page ", "attached file", "this file", "the report", "in the report",
    "according to my pdf",
)
_WEATHER_KEYWORDS = ("weather", "temperature", "forecast", "climate")
_WIKI_KEYWORDS = ("who is", "what is", "define", "tell me about", "wikipedia")
_SEARCH_KEYWORDS = (
    "latest", "news", "current", "today", "search for", "search",
    "recent", "update on", "happening",
)

# Matches simple arithmetic like "10+20", "30 - 5", "4 * 6", "100/5"
_MATH_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")

_OPERATORS = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}

# Small words to strip off the end of a "weather in X" style query
_TRAILING_PUNCT = re.compile(r"[?.!,]+$")


class Agent:
    """Routes a user query to the correct tool and returns a structured result."""

    def classify(self, text: str) -> str:
        """Decide which tool should handle an already lowercased, stripped query.

        Split out from run() so both the normal (blocking) path and the
        streaming path make the exact same routing decision.
        """
        if _MATH_PATTERN.search(text):
            return "calculator"

        # Only route to the PDF tool if the user has actually uploaded
        # something - otherwise a stray "pdf" mention falls through to Gemini.
        if rag_manager.has_documents() and any(keyword in text for keyword in _PDF_KEYWORDS):
            return "pdf"

        if any(keyword in text for keyword in _WEATHER_KEYWORDS):
            return "weather"

        if any(keyword in text for keyword in _WIKI_KEYWORDS):
            return "wikipedia"

        if any(keyword in text for keyword in _SEARCH_KEYWORDS):
            return "search"

        return "llm"

    def run(self, query: str) -> dict:
        if not query or not query.strip():
            return {"tool": "llm", "response": "Please type a message to get started!"}

        text = query.strip().lower()

        try:
            tool = self.classify(text)

            if tool == "calculator":
                match = _MATH_PATTERN.search(text)
                return self._run_calculator(match)

            if tool == "pdf":
                return self._run_pdf(query)

            if tool == "weather":
                return self._run_weather(text)

            if tool == "wikipedia":
                return self._run_wikipedia(query, text)

            if tool == "search":
                return self._run_search(query)

            logger.info("Routing to Gemini LLM (no tool matched)")
            return {"tool": "llm", "response": ask_llm(query)}

        except Exception as e:  # noqa: BLE001 - the agent must never crash the API
            logger.exception("Agent routing failed")
            return {"tool": "error", "response": f"Something went wrong: {e}"}

    def stream(self, query: str):
        """Generator version of run(), used by the live /agent/stream endpoint.

        Yields small event dicts so the frontend can render a typing effect:
        - {"type": "meta", "tool": ...}            sent first
        - {"type": "chunk", "text": ...}            for llm/pdf, repeated
        - {"type": "sources", "sources": [...]}     once, only for pdf
        - {"type": "result", "tool": ..., "response": ...}  for structured tools
        - {"type": "error", "message": ...}         on failure
        """
        if not query or not query.strip():
            yield {"type": "result", "tool": "llm", "response": "Please type a message to get started!"}
            return

        text = query.strip().lower()

        try:
            tool = self.classify(text)
            yield {"type": "meta", "tool": tool}

            if tool == "llm":
                for chunk in ask_llm_stream(query):
                    yield {"type": "chunk", "text": chunk}
                return

            if tool == "pdf":
                for chunk in rag_manager.ask_stream(query):
                    yield {"type": "chunk", "text": chunk}
                yield {"type": "sources", "sources": rag_manager.last_sources()}
                return

            # Structured tools (calculator/weather/wikipedia/search) are fast
            # and return a JSON payload, not prose - no benefit to streaming.
            result = self.run(query)
            yield {"type": "result", "tool": result["tool"], "response": result["response"]}

        except Exception as e:  # noqa: BLE001
            logger.exception("Agent streaming failed")
            yield {"type": "error", "message": str(e)}

    # -------------------------------------------------------------------
    @staticmethod
    def _run_pdf(question: str) -> dict:
        pdf_tool = TOOLS["pdf"]
        result = pdf_tool.run(question)
        return {"tool": "pdf", "response": result}

    # -------------------------------------------------------------------
    @staticmethod
    def _run_calculator(match: "re.Match") -> dict:
        num1 = float(match.group(1))
        operator = match.group(2)
        num2 = float(match.group(3))

        calculator = TOOLS["calculator"]
        result = calculator.run(num1=num1, num2=num2, operation=_OPERATORS[operator])
        return {"tool": "calculator", "response": result}

    @staticmethod
    def _run_weather(text: str) -> dict:
        city = text
        for keyword in _WEATHER_KEYWORDS:
            city = city.replace(keyword, "")
        for word in ("in", "at", "for", "of", "the"):
            city = re.sub(rf"\b{word}\b", "", city)

        city = _TRAILING_PUNCT.sub("", city).strip()
        # Fall back to the last word if stripping left nothing useful
        if not city:
            words = text.split()
            city = words[-1] if words else ""

        weather = TOOLS["weather"]
        result = weather.run(city.title())
        return {"tool": "weather", "response": result}

    @staticmethod
    def _run_wikipedia(original_query: str, text: str) -> dict:
        topic = text
        for keyword in _WIKI_KEYWORDS:
            topic = topic.replace(keyword, "")
        topic = _TRAILING_PUNCT.sub("", topic).strip() or original_query

        wikipedia_tool = TOOLS["wikipedia"]
        result = wikipedia_tool.run(topic)
        return {"tool": "wikipedia", "response": result}

    @staticmethod
    def _run_search(original_query: str) -> dict:
        search_tool = TOOLS["search"]
        result = search_tool.run(original_query)
        return {"tool": "search", "response": result}


agent = Agent()
