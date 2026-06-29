# Document_Intelligence_Assistant

# DocuMind AI: Document Intelligence Assistant

DocuMind AI is a powerful, local-first RAG (Retrieval-Augmented Generation) application built to help you chat with your documents. It uses **LangChain**, **FAISS** for vector storage, **HuggingFace** for local embeddings, and **Google Gemini 2.5 Flash** for intelligent document analysis.



## 🚀 Key Features
* **Local-First Embeddings**: Uses `all-MiniLM-L6-v2` via HuggingFace to keep processing fast and avoid API rate limits.
* **Glassmorphism UI**: A modern, interactive dashboard built with Streamlit.
* **Dual Themes**: Instant toggling between Dark and Light mode directly in the UI.
* **Multi-Format Support**: Ingests PDF, DOCX, PPTX, CSV, TXT, XLSX, and images (via Tesseract OCR).
* **RAG Pipeline**: Retrieves context-specific chunks to ensure precise, source-backed answers.

## 🛠️ Prerequisites
Before running the app, ensure you have:
1. **Python 3.10+**.
2. **Tesseract OCR** installed on your system (required for image/scanned PDF support).
3. **Google Gemini API Key** obtained from [Google AI Studio](https://aistudio.google.com/).

## 📦 Installation
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/vamsi1457/Document_Intelligence_Assistant.git](https://github.com/vamsi1457/Document_Intelligence_Assistant.git)
   cd Document_Intelligence_Assistant

```

2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Configure Environment**:
Create a `.env` file in the root directory and add your key:
```text
GOOGLE_API_KEY=your_actual_api_key_here

```



## 🏃 Running the Application

To start the server, use the following command:

```bash
streamlit run app.py --server.fileWatcherType none

```

## 📖 Usage Guide

1. **Document Upload**: Open the sidebar and upload your files (PDF, DOCX, Images, etc.).
2. **Indexing**: Click the **"⚡ Build Vector Store Index"** button. The app will process, chunk, and embed your documents locally.
3. **Querying**: Use the chat interface to ask questions about your documents. The system will retrieve relevant context and provide a precise, source-backed answer.
4. **Customization**: Use the sidebar to toggle between **Dark/Light Mode**, adjust **Search Settings** (Chunk size, Top-K), or modify **LLM Hyperparameters**.
5. **Export**: Use the **"📥 Export Chat History"** button to save your session as a Markdown file.

## 🏗️ Project Architecture

* `app.py`: Main Streamlit UI and session management.
* `utils/`: Contains modular backend logic:
* `loader.py`: Handles document parsing and image OCR.
* `chunking.py`: Splits text into intelligent, overlapping segments.
* `embeddings.py`: Configures the HuggingFace local embedding model.
* `vectorstore.py`: Manages FAISS vector index creation and loading.
* `rag_pipeline.py`: Orchestrates the retrieval and Gemini response generation.



## 📝 License

This project is for educational and research purposes.

```

### Important Note
If you have not yet created a `requirements.txt` file, be sure to run the following in your terminal before committing your changes:

```bash
pip freeze > requirements.txt

```

This ensures your deployment platform knows exactly which libraries to install to get your assistant running.
