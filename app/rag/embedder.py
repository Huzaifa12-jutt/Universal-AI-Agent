# ==========================
# rag/embedder.py
#
# Uses FastEmbed (ONNX runtime) for lightweight embeddings.
# FastEmbed is compatible with Python 3.14 and streamlit.
# ==========================

from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import settings


def get_embedding_model():
    """
    Load FastEmbed embeddings model.
    
    FastEmbed supports these models:
    - BAAI/bge-small-en-v1.5 (~33MB, recommended, fast)
    - BAAI/bge-base-en-v1.5 (~438MB, better quality)
    - BAAI/bge-large-en-v1.5 (~1.3GB, best quality)
    
    FastEmbed does NOT support sentence-transformers models.
    """
    
    model_name = settings.RAG_EMBEDDING_MODEL
    
    # Ensure we're using a FastEmbed-compatible model
    fastembed_models = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5", 
        "BAAI/bge-large-en-v1.5"
    ]
    
    if model_name not in fastembed_models:
        print(f"⚠️ Model '{model_name}' not compatible with FastEmbed. Using default.")
        model_name = "BAAI/bge-small-en-v1.5"
    
    try:
        embeddings = FastEmbedEmbeddings(
            model_name=model_name
        )
        print(f"✅ FastEmbed Model Loaded ({model_name})")
        return embeddings
    except Exception as e:
        print(f"❌ FastEmbed failed to load: {e}")
        raise