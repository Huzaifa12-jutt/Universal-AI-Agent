"""Unit tests for individual tools (no network calls where possible)."""

import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("WEATHER_API_KEY", "test-key")

from app.tools.registry import TOOLS


def test_calculator_add():
    result = TOOLS["calculator"].run(num1=20, num2=5, operation="add")
    assert result["result"] == 25


def test_calculator_multiply():
    result = TOOLS["calculator"].run(num1=20, num2=5, operation="multiply")
    assert result["result"] == 100


def test_calculator_divide_by_zero():
    with pytest.raises(ValueError):
        TOOLS["calculator"].run(num1=10, num2=0, operation="divide")


def test_calculator_invalid_operation():
    with pytest.raises(ValueError):
        TOOLS["calculator"].run(num1=1, num2=2, operation="modulo")


def test_weather_empty_city():
    result = TOOLS["weather"].run("")
    assert "error" in result


def test_search_empty_query():
    result = TOOLS["search"].run("")
    assert "error" in result


def test_wikipedia_empty_query():
    result = TOOLS["wikipedia"].run("")
    assert "error" in result
