import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)
DB_DIR = "vectorstore"
INDEX_NAME = "documind_faiss"

def create_or_update_vectorstore(chunks: list[Document]):
    """
    Takes text chunks, generates embeddings using HuggingFace, and saves a FAISS vector index locally.
    Now returns a tuple: (vectorstore_object, error_message_string) for brutal UI transparency.
    """
    if not chunks:
        logger.warning("No chunks provided to create vector store.")
        return None, "No text chunks were passed to the database builder."

    try:
        embeddings = get_embedding_model()
        os.makedirs(DB_DIR, exist_ok=True)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(folder_path=DB_DIR, index_name=INDEX_NAME)
        logger.info("Successfully created and saved FAISS vector store locally.")
        return vectorstore, "Success"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to create vector store. Error: {error_msg}")
        return None, error_msg

def load_local_vectorstore():
    """Loads the locally saved FAISS vector index from disk."""
    try:
        embeddings = get_embedding_model()
        if not os.path.exists(os.path.join(DB_DIR, f"{INDEX_NAME}.faiss")):
            return None
            
        return FAISS.load_local(
            folder_path=DB_DIR, 
            embeddings=embeddings, 
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store from disk. Error: {e}")
        return None