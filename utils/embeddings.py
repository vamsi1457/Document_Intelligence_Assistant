import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Initializes and returns the HuggingFace Embedding model.
    This runs locally on your machine, completely bypassing API rate limits.
    """
    try:
        # all-MiniLM-L6-v2 is fast, lightweight, and highly effective for RAG
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embedding model. Error: {e}")
        raise