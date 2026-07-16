# ==========================
# rag/retriever.py
# ==========================

import os

from langchain_community.vectorstores import FAISS

from app.config import settings


def get_retriever(vector_db=None, top_k=None, embedding_model=None):
    """
    Turn a vector store into a retriever that returns the top_k
    most relevant chunks for a given question.

    Args:
        vector_db: FAISS vector store instance (optional)
        top_k: Number of chunks to retrieve (default from config)
        embedding_model: Embedding model for loading from disk (optional)

    Returns:
        Retriever object
    """
    # ✅ CHANGE: Use 10 as default if top_k not provided
    # If top_k is None, use 10 instead of settings.RAG_TOP_K
    # This gives more context for normal queries
    k_value = top_k if top_k is not None else 10
    
    # If vector_db is provided, use it directly
    if vector_db is not None:
        retriever = vector_db.as_retriever(
            search_kwargs={"k": k_value}
        )
        print(f"✅ Retriever created with top_k={k_value}")
        return retriever

    # If no vector_db but embedding_model provided, try loading from disk
    if embedding_model is not None and os.path.exists(settings.VECTOR_DB_PATH):
        vector_db = FAISS.load_local(
            folder_path=settings.VECTOR_DB_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        retriever = vector_db.as_retriever(
            search_kwargs={"k": k_value}
        )
        print(f"✅ Retriever loaded from disk with top_k={k_value}")
        return retriever

    raise ValueError("No vector store found. Please process documents first or provide a vector_db.")