# ==========================
# rag/embedder.py
#
# Uses HuggingFace sentence-transformers for embeddings.
# Compatible with Python 3.14 (no fastembed conflicts)
# ==========================

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings


def get_embedding_model():
    """
    Load HuggingFace sentence-transformers embedding model.
    
    Models that work well:
    - sentence-transformers/all-MiniLM-L6-v2 (~80MB, fast, good quality)
    - sentence-transformers/all-mpnet-base-v2 (~420MB, better quality)
    - BAAI/bge-small-en-v1.5 (~33MB, light weight)
    - BAAI/bge-base-en-v1.5 (~438MB, good quality)
    """
    
    model_name = settings.RAG_EMBEDDING_MODEL
    
    # If config still has fastembed model, use default
    if model_name in ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"✅ HuggingFace Embeddings Loaded ({model_name})")
    return embeddings