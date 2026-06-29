# The orchestrator: takes user query -> searches DB -> talks to Gemini.
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

logger = logging.getLogger(__name__)

def run_rag_pipeline(query: str, vectorstore, top_k: int = 4, temperature: float = 0.0) -> dict:
    """
    Executes the full RAG pipeline:
    Retrieves top-K chunks -> Formats prompt -> Queries Gemini 2.5 Flash -> Returns answer & sources.
    """
    fallback_response = "I couldn't find relevant information in the uploaded documents."
    
    if not vectorstore:
        return {"answer": fallback_response, "sources": []}

    try:
        # Step 1: Retrieve relevant chunks
        retrieved_docs = vectorstore.similarity_search(query, k=top_k)
        if not retrieved_docs:
            return {"answer": fallback_response, "sources": []}

        # Step 2: Extract text and sources
        context_parts = []
        sources = []
        for doc in retrieved_docs:
            context_parts.append(doc.page_content)
            source_name = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            page = doc.metadata.get("page", None)
            source_str = f"{source_name} (Page {page + 1})" if page is not None else source_name
            if source_str not in sources:
                sources.append(source_str)

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Set up the LLM (Gemini 2.5 Flash)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_retries=2
        )

        # Step 4: Define a strict prompt instruction set
        template = """You are a precise document analysis AI. Answer the user question strictly using the provided context.
If the answer cannot be confidently derived from the context, respond exactly with:
"I couldn't find relevant information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Step 5: Chain execution
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})

        return {
            "answer": answer.strip(),
            "sources": sources,
            "raw_chunks": context_parts
        }

    except Exception as e:
        logger.error(f"Error executing RAG pipeline: {e}")
        return {"answer": f"An error occurred during processing: {str(e)}", "sources": []}