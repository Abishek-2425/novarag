# RAGNOVA — Offline Multimodal RAG MVP

RAGNOVA is an offline, multimodal Retrieval-Augmented Generation (RAG) MVP. It enables you to upload and query documents (PDF, DOCX, PPTX, Excel, images, audio) using a local vector store and an offline LLM (Ollama or GPT4All). All processing is done locally—no cloud required.

---

## Features

- **Streamlit UI** for uploading, managing, and querying files.
- **Multimodal ingestion**: PDF, DOCX, PPTX, Excel, images (OCR), audio (transcription).
- **FAISS vector store** with SentenceTransformers embeddings (`all-MiniLM-L6-v2`).
- **Offline LLM inference** via [Ollama](https://ollama.com/) (default: `phi:2.7b`).
- **Chunked document ingestion** for better retrieval and context.
- **All data and models stored locally** for privacy and offline use.

---

## Quickstart

### 1. Clone and set up environment

```powershell
git clone <your-repo-url>
cd rag
python -m venv .venv
[Activate.ps1](http://_vscodecontentref_/0)
pip install -r [r.txt](http://_vscodecontentref_/1)
```

Note: If you prefer, rename r.txt to requirements.txt.

## 2. Download LLM Model
For Ollama: Install Ollama and pull a model (e.g., ollama pull phi:2.7b).
For GPT4All: Download the .bin model (e.g., ggml-gpt4all-j-v1.3-groovy.bin) and place it in the project root.

## 3. Run the app
streamlit run [app.py](http://_vscodecontentref_/2)

## 4.Usage
Upload files: Use the UI to upload PDFs, DOCX, PPTX, Excel, images, or audio files.
Ingestion: The app extracts and chunks text, creates embeddings, and stores them in FAISS.
Query: Enter a question. The app retrieves relevant chunks and generates an answer using the offline LLM.

## 5.Project Structure
app.py — Streamlit UI and main app logic.
ingestation/ — Ingestion adapters for each file type.
retrieval/vector_store.py — FAISS vector store and embedding logic.
llm/ollama_inference.py — Ollama LLM wrapper.
embeddings/ — Stores FAISS index and metadata.
uploads/ — Uploaded files are saved here for persistent access.
See rag_tree.txt for a full directory tree.

## 6.Supported File Types
PDF (.pdf)
Word (.docx)
PowerPoint (.pptx)
Excel (.xls, .xlsx)
Images (.png, .jpg, .jpeg, .bmp, .tiff)
Audio (.mp3, .wav, .m4a, .flac, .aac, .ogg)

## 7.Dependencies
See r.txt for the full list. Key packages:

streamlit
sentence-transformers
faiss-cpu
pymupdf
python-docx
python-pptx
pandas, openpyxl, xlrd
pytesseract, Pillow
faster-whisper
ollama (or gpt4all if using GPT4All)

## 8.Notes & Tips
Uploaded files are saved in uploads/ and referenced in metadata for correct rendering.
All embeddings and metadata are stored in embeddings/.
To reset the app, use the "Clear All Data" button in the UI.
For large audio files, transcription may be slow (CPU by default).
Ollama must be running locally for LLM inference.

## 9.Troubleshooting
Model not found: Ensure your LLM .bin file is in the project root or Ollama is running with the correct model pulled.
Missing dependencies: Install all packages from r.txt.
File ingestion errors: Check logs for details; ensure files are not corrupted.

## 10.License
This project is for research and prototyping purposes. See LICENSE if present.

## 11.Credits
Built with Streamlit, FAISS, SentenceTransformers, Ollama, and more.
Contributing
PRs and issues welcome! See `.github---
