"""Pydantic request/response schemas used by the FastAPI routes."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message to the agent")


class ChatResponse(BaseModel):
    tool: str = Field(..., description="Which tool handled the request")
    response: Any = Field(..., description="The tool's output")


class CalculatorRequest(BaseModel):
    num1: float
    num2: float
    operation: str = Field(..., description="add | subtract | multiply | divide")


class WeatherRequest(BaseModel):
    city: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class WikipediaRequest(BaseModel):
    query: str = Field(..., min_length=1)


class PDFChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Question about the uploaded PDF(s)")


class PDFDeleteRequest(BaseModel):
    filename: str = Field(..., min_length=1)


class ErrorResponse(BaseModel):
    error: str
