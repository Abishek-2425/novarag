## Quick orientation — what this project is

This is an offline, multimodal RAG (Retrieval-Augmented Generation) MVP composed of:

- `app.py` — a Streamlit UI that accepts uploads, runs ingestion, queries a VectorStore and calls an offline LLM.
- `ingestation/` — ingestion adapters (PDF, DOCX, Image OCR, Audio transcription). Note: folder name is `ingestation` (non-standard spelling); stay consistent.
- `retrieval/vector_store.py` — FAISS + SentenceTransformers based embeddings and retrieval.
- `llm/ollama_inference.py` — wrapper around the `ollama` package for offline generation.
- `embeddings/` — target folder for FAISS index and metadata pickle (created at runtime).

Read these files first when making changes: `app.py`, `retrieval/vector_store.py`, `llm/gpt4all_inference.py`, and `ingestation/*`.

## Key patterns & integration points

- Ingestion modules currently implement folder-based helpers: `ingest_folder()`, `extract_text_from_pdf(...)`, `transcribe_audio(...)`, etc. They expect local file paths (constructors take `folder_path`).
- `app.py` is a Streamlit front-end that expects per-file ingestion methods (e.g. `Ingestor().ingest(uploaded_file)`) and calls `VectorStore.add_document(text, source)` — those APIs do not exist in the current ingestion/vector store implementations. See "Known mismatches".
- `VectorStore` persists embeddings to `embeddings/faiss_index.index` and metadata to `embeddings/meta.pkl`. The embedding model used is `sentence-transformers` (`all-MiniLM-L6-v2`).
- `GPT4AllInference` expects a local `.bin` model at the project root (default `ggml-gpt4all-j-v1.3-groovy.bin`) and uses the `gpt4all` package.
- Audio uses `faster-whisper` (CPU by default in code). Images use `pytesseract` + `PIL`. PDFs use `PyMuPDF` (`fitz`). DOCX uses `python-docx`.

## Known mismatches and immediate bug risks (actionable)

1. Ingestion API vs UI mismatch

   - `ingestation/*.py` classes are folder-centric and do NOT provide an `ingest(uploaded_file)` method. But `app.py` calls `PDFIngestor().ingest(file)` and similar — this will raise TypeError/AttributeError.
   - Fix: either (A) implement an `ingest(self, uploaded_file)` method on each ingestor that accepts a Streamlit `UploadedFile` or bytes-like object and returns extracted text, or (B) change `app.py` to save uploaded files to a temp/`uploads/` folder and call the existing `extract_...`/`transcribe_...` methods.

2. VectorStore API mismatch

   - `app.py` calls `vs.add_document(text=text, source=file.name)` but `VectorStore` implements `build_index()` and `query()` only. There is no incremental `add_document` method.
   - Fix: implement `VectorStore.add_document(text: str, source: str)` which:
     - encodes `text` using the same `SentenceTransformer` model,
     - adds the vector to `self.index` (create it if None),
     - appends metadata to `self.metadata`, and
     - writes the index + metadata back to disk (atomic / create dir if needed).
   - Note: `IndexFlatL2` supports `.add()` but you must keep metadata indices aligned with index order.

3. Uploaded files vs stored source paths

   - The metadata stores `source` as a filename only. The UI attempts to call `st.image(item['source'])` or `st.audio(item['source'])`, which will fail unless that path is accessible from disk. Uploaded Streamlit files are not automatically saved to disk.
   - Fix: persist uploaded files under `uploads/` and store the full path in metadata, or modify UI to use the uploaded file bytes directly when rendering.

4. Missing or brittle assumptions
   - `gpt4all` usage: confirm the installed `gpt4all` API in your environment; runtime behavior can differ by version. `model.generate(...)` is assumed to return string but validate locally.
   - `faster-whisper` transcription currently runs on CPU by default in `AudioIngestor`; on large audio this can be slow/out-of-memory. Consider optional GPU device selection.

## How to run locally (Windows PowerShell)

1. Create/activate venv and install deps (the repo has `r.txt` listing deps — rename to `requirements.txt` if you prefer):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r r.txt
```

2. Run the Streamlit app:

```powershell
streamlit run app.py
```

3. If the app complains about the LLM model, download the `.bin` model and place it in the project root (name expected in code: `ggml-gpt4all-j-v1.3-groovy.bin`).

## Small, high-value changes an AI contributor can make

- Add `ingest(self, uploaded_file)` adapters in `ingestation/*` that accept Streamlit `UploadedFile` (or bytes) and return text. Keep existing folder-based helpers for batch ingestion.
- Implement `VectorStore.add_document(...)` (incremental add + persist). Add a small helper `ensure_index()` to lazily create the index if missing.
- Add an `uploads/` directory and persist uploaded files there; update metadata to store absolute/relative paths.
- Add a short `README.md` or rename `r.txt` -> `requirements.txt` and provide a reproducible setup snippet in the repo root.

## Files to open first when modifying behavior

- `app.py` — UI glue and where most integration bugs are visible.
- `retrieval/vector_store.py` — implement `add_document()` and review index persistence.
- `llm/gpt4all_inference.py` — confirm API semantics for your installed `gpt4all` version.
- `ingestation/*.py` — add upload-friendly adapters.

If any part of this is unclear or you'd like me to implement any of the fixes (e.g., `add_document()` or `ingest()` adapters), tell me which one to implement first and I'll create the change and run quick checks.
