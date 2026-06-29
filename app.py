import os
from dotenv import load_dotenv

# Load environment variables from the .env file BEFORE doing anything else
load_dotenv()
import json
import logging
import shutil
import streamlit as st
from utils.loader import load_document
from utils.chunking import split_documents
from utils.vectorstore import create_or_update_vectorstore, load_local_vectorstore
from utils.rag_pipeline import run_rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Page Configuration (Must be the absolute first Streamlit command)
st.set_page_config(
    page_title="DocuMind AI – Chat with Your Documents",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vectorstore" not in st.session_state:
    # Attempt to pull an existing index from disk on initialization
    st.session_state.vectorstore = load_local_vectorstore()
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# 3. Define Sidebar and Theme Toggle FIRST
with st.sidebar:
    st.markdown("# 🧠 Document Assistant")
    
    # The Theme Toggle
    selected_theme = st.radio("UI Theme", ["Dark Mode", "Light Mode"], horizontal=True)
    st.markdown("---")
    
    st.markdown("### RAG Advanced Configuration")
    
    # Expandable Configuration Parameters
    with st.expander("⚙️ Vector Search Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (Characters)", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=4, step=1)
        
    with st.expander("🤖 LLM Hyperparameters", expanded=False):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.caption("Model Provider: **Google GenAI**")
        st.caption("Model Target: `gemini-2.5-flash`")

    st.markdown("---")
    st.markdown("### Document Pipeline Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload contextual document assets:", 
        type=["pdf", "docx", "pptx", "csv", "txt", "xlsx", "png", "jpg", "jpeg", "md"],
        accept_multiple_files=True
    )

    # Ingestion Core Processing Trigger
    if st.button("⚡ Build Vector Store Index") and uploaded_files:
        all_chunks = []
        os.makedirs("uploads", exist_ok=True)
        
        # UI Feedback initialization
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clear old file references for this indexing process
        st.session_state.processed_files = []
        st.session_state.total_chunks = 0
        
        for idx, file_asset in enumerate(uploaded_files):
            status_text.caption(f"Processing target asset: `{file_asset.name}`")
            
            temp_path = os.path.join("uploads", file_asset.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(file_asset.getvalue())
                
                extracted_docs = load_document(temp_path)
                
                if extracted_docs:
                    document_chunks = split_documents(extracted_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    all_chunks.extend(document_chunks)
                    
                    st.session_state.processed_files.append({
                        "name": file_asset.name, 
                        "chunks": len(document_chunks),
                        "size_kb": round(len(file_asset.getvalue()) / 1024, 2)
                    })
                    st.session_state.total_chunks += len(document_chunks)
                else:
                    st.error(f"Failed parsing structural text from: {file_asset.name}")
                    
            except Exception as e:
                logger.error(f"Pipeline error processing {file_asset.name}: {e}")
                st.error(f"Error parsing asset: {file_asset.name}")
            finally:
                progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        if all_chunks:
            with st.spinner("Generating embeddings locally via HuggingFace (all-MiniLM-L6-v2) & writing to FAISS..."):
                # Capture the explicit error message from our rewritten vectorstore.py
                vector_db, error_msg = create_or_update_vectorstore(all_chunks)
                if vector_db:
                    st.session_state.vectorstore = vector_db
                    st.success(f"Successfully indexed {st.session_state.total_chunks} chunk blocks!")
                else:
                    st.error("Vector database assembly failed.")
                    st.error(f"CRITICAL ERROR DETAIL: {error_msg}")
                    st.warning("If this says 'No module named sentence_transformers', open your terminal and run: pip install langchain-huggingface sentence-transformers")
                    st.warning("If this says 'Could not import faiss', open your terminal and run: pip install faiss-cpu")
        else:
            st.error("Ingestion queue yielded zero valid text segments.")

    # Render Current Active Engine Status
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### Active Knowledge Base")
        st.metric(label="Total Chunks Embedded", value=st.session_state.total_chunks)
        
        for f in st.session_state.processed_files:
            st.markdown(
                f"<div class='sidebar-file-badge'><span>📁 {f['name']} ({f['size_kb']} KB)</span> <code>{f['chunks']} chunks</code></div>", 
                unsafe_allow_html=True
            )
            
        # Context management actions
        st.markdown("##")
        if st.button("🗑️ Clear Active Index & History"):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.session_state.total_chunks = 0
            
            # Wipe local vector directories and temporary uploads
            for target_dir in ["vectorstore", "uploads"]:
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                    
            st.toast("Local context memory flushed successfully.")
            st.rerun()

# 4. Dynamically Inject the chosen CSS based on the toggle state
css_target = "static/style.css" if selected_theme == "Dark Mode" else "static/style_light.css"
if os.path.exists(css_target):
    with open(css_target) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error(f"CSS file not found at {css_target}. Ensure you created it in the static/ folder.")

# 5. Main Chat Interface Workspace
st.markdown("## 💬 Chat with Your Documents Using RAG")
st.caption("Powered by LangChain, FAISS Vector Storage, HuggingFace, and Google Gemini 2.5 Flash.")

# Render Conversation Logs and Citations
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown(
                f"<div class='source-citation'><b>Retrieved Sources:</b><br>{'<br>'.join(message['sources'])}</div>", 
                unsafe_allow_html=True
            )

# Input Box Processing Flow
if prompt_query := st.chat_input("Enter a prompt regarding your context data..."):
    with st.chat_message("user"):
        st.markdown(prompt_query)
    st.session_state.chat_history.append({"role": "user", "content": prompt_query})

    with st.chat_message("assistant"):
        if not st.session_state.vectorstore:
            # Hard fallback if they ignored the red errors during the build phase.
            fallback_msg = "Your database is empty. Either you haven't built it, or the build process crashed. Check the Control Panel for red error boxes."
            st.markdown(fallback_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": fallback_msg, "sources": []})
        else:
            with st.spinner("Searching database indices and constructing context template..."):
                pipeline_result = run_rag_pipeline(
                    query=prompt_query,
                    vectorstore=st.session_state.vectorstore,
                    top_k=top_k,
                    temperature=temperature
                )
                
                st.markdown(pipeline_result["answer"])
                if pipeline_result["sources"]:
                    st.markdown(
                        f"<div class='source-citation'><b>Retrieved Sources:</b><br>{'<br>'.join(pipeline_result['sources'])}</div>", 
                        unsafe_allow_html=True
                    )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": pipeline_result["answer"],
                    "sources": pipeline_result["sources"]
                })

# 6. Utilities for Session Management (Download Chat Logs)
if st.session_state.chat_history:
    st.markdown("---")
    markdown_transcript = "# DocuMind AI Conversation Log\n\n"
    for interaction in st.session_state.chat_history:
        role_title = "### 👤 User" if interaction["role"] == "user" else "### 🤖 Assistant"
        markdown_transcript += f"{role_title}\n{interaction['content']}\n\n"
        if interaction.get("sources"):
            markdown_transcript += f"*Sources Used:* {', '.join(interaction['sources'])}\n\n"
        markdown_transcript += "---\n\n"

    st.download_button(
        label="📥 Export Chat History (.MD)",
        data=markdown_transcript,
        file_name="documind_chat_history.md",
        mime="text/markdown"
    )