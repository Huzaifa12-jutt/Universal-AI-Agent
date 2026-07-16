"""
rag/manager.py

Thin orchestration layer on top of the existing RAG building blocks
(loader, chunker, embedder, vector_store, retriever, chatbot).

None of the RAG internals are rewritten here - this module only wires
them together and keeps track of which PDFs are currently indexed, so
the rest of the app (tools/pdf_rag.py, main.py, the Streamlit frontend)
has a single, simple object to talk to: `rag_manager`.

State lives in memory for the lifetime of the backend process. That's
fine for this project (single-instance deployment); if the process
restarts, PDFs just need to be re-uploaded.
"""

import logging
import os
import shutil
import tempfile

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.rag.chatbot import create_chatbot
from app.rag.chunker import split_documents
from app.rag.embedder import get_embedding_model
from app.rag.loader import load_multiple_pdfs
from app.rag.retriever import get_retriever
from app.rag.vector_store import create_vector_store
from app.rag.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RAGManager:
    """Owns the current PDF knowledge base: files, vector index, chatbot."""

    def __init__(self):
        self._embedding_model = None
        self.vector_db = None
        self.retriever = None
        self.chatbot = None
        self.files: dict[str, dict] = {}  # filename -> {"path", "pages", "size_kb"}
        self._tmp_dir = tempfile.mkdtemp(prefix="ai_agent_pdfs_")
        self._all_chunks = []  # ✅ Store all chunks for summary queries

    # ------------------------------------------------------------------
    # Embedding model (loaded once, reused across rebuilds)
    # ------------------------------------------------------------------
    def _get_embedder(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    # ------------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------------
    def has_documents(self) -> bool:
        return bool(self.files) and self.retriever is not None

    def list_pdfs(self) -> list[dict]:
        return [
            {"name": name, "pages": meta["pages"], "size_kb": meta["size_kb"]}
            for name, meta in self.files.items()
        ]

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def add_pdfs(self, uploads: list[tuple[str, bytes]]) -> dict:
        """
        uploads: list of (filename, raw_bytes) tuples.
        Saves each PDF to a temp dir and rebuilds the vector index.
        """
        added, skipped = [], []

        for filename, data in uploads:
            if filename in self.files:
                skipped.append(filename)
                continue
            path = os.path.join(self._tmp_dir, filename)
            with open(path, "wb") as f:
                f.write(data)
            self.files[filename] = {
                "path": path,
                "pages": 0,
                "size_kb": round(len(data) / 1024, 1),
            }
            added.append(filename)

        result = {"added": added, "skipped": skipped, "total_files": len(self.files)}

        if added:
            build_result = self.rebuild()
            result["chunks"] = build_result.get("chunks", 0)

        return result

    def remove_pdf(self, filename: str) -> bool:
        if filename not in self.files:
            return False

        path = self.files[filename]["path"]
        self.files.pop(filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning("Could not delete temp file for %s", filename)

        self.rebuild()
        return True

    def clear_all(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="ai_agent_pdfs_")
        self.files.clear()
        self.vector_db = None
        self.retriever = None
        self.chatbot = None
        self._all_chunks = []  # ✅ Clear stored chunks
        logger.info("RAG knowledge base cleared")

    # ------------------------------------------------------------------
    # Index building - reuses loader -> chunker -> embedder -> vector_store
    # ------------------------------------------------------------------
    def rebuild(self) -> dict:
        if not self.files:
            self.vector_db = None
            self.retriever = None
            self.chatbot = None
            self._all_chunks = []  # ✅ Clear stored chunks
            return {"status": "empty", "chunks": 0, "files": 0}

        pdf_paths_with_names = [(meta["path"], name) for name, meta in self.files.items()]

        documents, stats = load_multiple_pdfs(pdf_paths_with_names)
        for name, page_count in stats.items():
            if name in self.files:
                self.files[name]["pages"] = page_count

        chunks = split_documents(
            documents,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )

        # ✅ Store all chunks for summary queries
        self._all_chunks = chunks

        embedding_model = self._get_embedder()
        self.vector_db = create_vector_store(chunks, embedding_model)
        self.retriever = get_retriever(vector_db=self.vector_db, top_k=settings.RAG_TOP_K)

        if settings.GROQ_API_KEY:
            self.chatbot = create_chatbot(
                retriever=self.retriever,
                api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=settings.RAG_TEMPERATURE,
            )
        else:
            self.chatbot = None

        logger.info(
            "RAG index rebuilt: %d file(s), %d chunk(s)", len(self.files), len(chunks)
        )
        return {"status": "ok", "chunks": len(chunks), "files": len(self.files)}

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    @staticmethod
    def _sources_from_docs(docs) -> list[dict]:
        sources = []
        seen = set()
        for doc in docs or []:
            file_name = doc.metadata.get("source", "unknown")
            page = (doc.metadata.get("page", 0) or 0) + 1  # PyPDFLoader pages are 0-indexed
            key = (file_name, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"file": file_name, "page": page})
        return sources

    def _is_summary_query(self, question: str) -> bool:
        """Check if the query asks for a summary or overview."""
        summary_keywords = [
            "summarize", "summary", "overview", "brief", "gist", 
            "full document", "what is this pdf about", "explain the whole",
            "give me a summary", "summarise", "main points", "key points",
            "abstract", "synopsis", "executive summary", "tl;dr"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in summary_keywords)

    def _ask_with_docs(self, question: str, docs: list) -> str:
        """
        Helper: Ask chatbot with custom docs (bypasses retriever).
        Used for summary queries where we need all chunks.
        """
        if self.chatbot is None:
            return "Chatbot not configured. Please check your GROQ_API_KEY."
        
        # Store docs for sources
        self.chatbot.last_sources = docs
        
        # Build context from all docs
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt with all context
        prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        final_prompt = prompt.invoke({
            "context": context,
            "question": question,
        })
        
        # Get response using the chatbot's LLM
        response = self.chatbot.llm.invoke(final_prompt)
        return response.content

    def ask(self, question: str) -> dict:
        if not self.has_documents():
            return {
                "answer": "No PDFs are indexed yet. Upload one or more PDFs first, then ask again.",
                "sources": [],
            }
        if not settings.GROQ_API_KEY or self.chatbot is None:
            return {
                "answer": "PDF chat isn't configured yet - add a GROQ_API_KEY to your .env file.",
                "sources": [],
            }

        # ✅ NEW: Check if this is a summary/overview query
        if self._is_summary_query(question) and self._all_chunks:
            try:
                # Use ALL chunks for summary
                all_docs = self._all_chunks
                logger.info(f"📝 Summary query detected: using all {len(all_docs)} chunks")
                
                # Limit chunks if too many (to avoid token overflow)
                max_chunks_for_summary = 50
                if len(all_docs) > max_chunks_for_summary:
                    logger.info(f"⚠️ Limiting to {max_chunks_for_summary} chunks to avoid token overflow")
                    all_docs = all_docs[:max_chunks_for_summary]
                
                answer = self._ask_with_docs(question, all_docs)
                return {
                    "answer": answer, 
                    "sources": self._sources_from_docs(all_docs)
                }
            except Exception as e:
                logger.error(f"⚠️ Summary query failed: {e}")
                # Fallback to normal retrieval
                answer = self.chatbot.ask(question)
                return {
                    "answer": answer, 
                    "sources": self._sources_from_docs(self.chatbot.last_sources)
                }
        else:
            # Normal query - use retriever with top-k
            answer = self.chatbot.ask(question)
            return {
                "answer": answer, 
                "sources": self._sources_from_docs(self.chatbot.last_sources)
            }

    def ask_stream(self, question: str):
        """Yields plain text chunks. Caller can fetch sources afterwards via last_sources()."""
        if not self.has_documents():
            yield "No PDFs are indexed yet. Upload one or more PDFs first, then ask again."
            return
        if not settings.GROQ_API_KEY or self.chatbot is None:
            yield "PDF chat isn't configured yet - add a GROQ_API_KEY to your .env file."
            return

        # ✅ NEW: Check if this is a summary/overview query
        if self._is_summary_query(question) and self._all_chunks:
            try:
                all_docs = self._all_chunks
                logger.info(f"📝 Summary query detected (stream): using all {len(all_docs)} chunks")
                
                # Limit chunks if too many
                max_chunks_for_summary = 50
                if len(all_docs) > max_chunks_for_summary:
                    all_docs = all_docs[:max_chunks_for_summary]
                
                # Get summary response
                answer = self._ask_with_docs(question, all_docs)
                yield answer
                return
            except Exception as e:
                logger.error(f"⚠️ Summary query failed (stream): {e}")
                # Fallback to normal streaming
        
        # Normal query or fallback - use streaming
        for chunk in self.chatbot.stream(question):
            yield chunk

    def last_sources(self) -> list[dict]:
        if self.chatbot is None:
            return []
        return self._sources_from_docs(self.chatbot.last_sources)


# Single shared instance used across the whole backend
rag_manager = RAGManager()