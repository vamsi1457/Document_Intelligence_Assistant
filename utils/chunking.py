#Slices massive documents into smaller, readable pieces.
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.
    Maintains a 200-character overlap by default to preserve context between chunks.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully split into {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to split documents. Error: {e}")
        return []