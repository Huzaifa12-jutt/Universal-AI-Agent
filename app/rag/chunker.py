# ==========================
# rag/chunker.py
# ==========================

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Split documents into smaller, overlapping chunks so retrieval
    can find precise, relevant passages instead of whole pages.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.RAG_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks
