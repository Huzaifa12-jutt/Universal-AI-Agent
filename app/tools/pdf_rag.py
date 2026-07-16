"""PDF RAG tool - answers questions using the PDFs the user has uploaded.

This tool doesn't contain any RAG logic itself - it just delegates to
`rag_manager`, which owns the reused loader/chunker/embedder/vector_store/
retriever/chatbot pipeline (see app/rag/manager.py).
"""

import logging

from app.rag.manager import rag_manager
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class PDFRAGTool(BaseTool):
    """Answers questions using the content of PDFs uploaded by the user (RAG)."""

    name = "pdf"
    description = "Answers questions from PDF documents the user has uploaded, with page citations."

    def run(self, question: str) -> dict:
        question = (question or "").strip()

        if not question:
            return {"error": "Please ask a question about your PDF."}

        logger.info("PDF RAG tool called with question=%s", question)

        try:
            return rag_manager.ask(question)
        except Exception as e:  # noqa: BLE001 - never let the agent crash on a bad PDF answer
            logger.exception("PDF RAG tool failed")
            return {"error": f"PDF chat failed: {e}"}
