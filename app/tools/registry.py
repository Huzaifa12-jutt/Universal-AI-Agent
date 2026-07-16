"""Central registry mapping tool names to tool instances."""

from app.tools.calculator import CalculatorTool
from app.tools.weather import WeatherTool
from app.tools.search import SearchTool
from app.tools.wikipedia import WikipediaTool
from app.tools.pdf_rag import PDFRAGTool

TOOLS = {
    "calculator": CalculatorTool(),
    "weather": WeatherTool(),
    "search": SearchTool(),
    "wikipedia": WikipediaTool(),
    "pdf": PDFRAGTool(),
}
