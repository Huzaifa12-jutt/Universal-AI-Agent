# ==========================
# rag/vector_store.py
# ==========================

import logging
import shutil
import os

from langchain_community.vectorstores import FAISS

from app.config import settings

# FAISS doesn't have telemetry issues, so no logging suppression needed


def reset_vector_store():
    """
    Wipe any existing vector database so a fresh one can be built
    (used when the user uploads a new set of documents).
    """
    if os.path.exists(settings.VECTOR_DB_PATH):
        shutil.rmtree(settings.VECTOR_DB_PATH, ignore_errors=True)
        print(f"✅ Vector store cleared: {settings.VECTOR_DB_PATH}")


def create_vector_store(chunks, embedding_model):
    """
    Build a new FAISS vector database from document chunks.
    Works on Streamlit Cloud (in-memory, no write issues).
    """
    reset_vector_store()
    
    # FAISS stores in-memory by default (no persist_directory needed)
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    
    print(f"✅ FAISS Vector Store Created Successfully")
    return vector_db


def load_vector_store(embedding_model):
    """
    Load an existing FAISS vector database from disk.
    """
    if os.path.exists(settings.VECTOR_DB_PATH):
        vector_db = FAISS.load_local(
            folder_path=settings.VECTOR_DB_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS Vector Store Loaded from: {settings.VECTOR_DB_PATH}")
        return vector_db
    else:
        raise ValueError(f"No vector store found at: {settings.VECTOR_DB_PATH}")