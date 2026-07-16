"""Unit tests for the agent's tool-routing logic."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("WEATHER_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.agent import agent
from app.rag.manager import rag_manager


def test_routes_math_to_calculator():
    result = agent.run("What is 10+20?")
    assert result["tool"] == "calculator"
    assert result["response"]["result"] == 30


def test_routes_weather_query():
    result = agent.run("weather in Lahore")
    assert result["tool"] == "weather"


def test_routes_wikipedia_query():
    result = agent.run("who is Alan Turing")
    assert result["tool"] == "wikipedia"


def test_routes_search_query():
    result = agent.run("latest AI news")
    assert result["tool"] == "search"


def test_empty_message_handled_gracefully():
    result = agent.run("")
    assert result["tool"] == "llm"


def test_pdf_keywords_fall_back_to_llm_when_no_pdf_uploaded():
    """Mentioning 'pdf' shouldn't route to the PDF tool if nothing is indexed."""
    assert rag_manager.has_documents() is False
    assert agent.classify("what does the pdf say") == "llm"


def test_classify_matches_run_for_calculator():
    text = "12 * 4"
    assert agent.classify(text) == "calculator"
    assert agent.run(text)["tool"] == "calculator"
