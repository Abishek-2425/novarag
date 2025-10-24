# app.py
import os
import shutil
import logging
import streamlit as st

from ingestation.ingest_pdf import PDFIngestor
from ingestation.ingest_docx import DOCXIngestor
from ingestation.ingest_image import ImageIngestor
from ingestation.ingest_audio import AudioIngestor
from ingestation.ingest_pptx import PPTXIngestor
from ingestation.ingest_excel import ExcelIngestor

from retrieval.vector_store import VectorStore
from llm.ollama_inference import OllamaInference

# --------------------------------------
# Basic Configurations
# --------------------------------------
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Offline Multimodal RAG", layout="wide")

col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    st.title("RAGNOVA")

# --------------------------------------
# Ensure upload folder exists
# --------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DIR = "embeddings"

# --------------------------------------
# Initialize Session State
# --------------------------------------
if "ingested_files" not in st.session_state:
    st.session_state["ingested_files"] = []

# --------------------------------------
# Instantiate Vector Store
# --------------------------------------
vs = VectorStore()

# --------------------------------------
# Create Tabs
# --------------------------------------
tab1, tab2 = st.tabs(["📂 Document Upload", "💬 Query Interface"])


# ==========================================================
# TAB 1: Document Upload & List
# ==========================================================
with tab1:
    st.header("📤 Upload & Manage Files")

    ## top_k = st.slider("Top-K retrieval", min_value=1, max_value=10, value=3, step=1)

    if st.button("🗑 Clear All Data (index + embeddings)"):
        try:
            vs_tmp = VectorStore()
            vs_tmp.clear()
        except Exception:
            logging.exception("Error clearing vector store (continuing).")

        if os.path.exists(DIR):
            for filename in os.listdir(DIR):
                file_path = os.path.join(DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path, ignore_errors=True)
                except Exception:
                    logging.exception(f"Failed to delete {file_path}, continuing.")

        st.session_state["ingested_files"] = []
        st.success("Cleared vector store and embeddings. Start from scratch!")
        st.rerun()

    st.subheader("📁 Upload New Files")
    uploaded_files = st.file_uploader(
        "Drag & drop or select files (pdf, docx, png, jpg, jpeg, wav, mp3, pptx, xls, xlsx)",
        type=["pdf", "docx", "png", "jpg", "jpeg", "wav", "mp3", "pptx", "xls", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader_only_new"  # unique key to avoid conflicts
    )

    # process new files without showing them in the uploader itself
    if uploaded_files:
        already = set(st.session_state["ingested_files"])
        new_files = [f for f in uploaded_files if f.name not in already]

        if new_files:
            all_docs = {}
            for upload in new_files:
                safe_path = os.path.join(UPLOAD_DIR, upload.name)
                with open(safe_path, "wb") as out:
                    out.write(upload.getbuffer())

                ext = upload.name.split(".")[-1].lower()
                try:
                    if ext == "pdf":
                        text = PDFIngestor(UPLOAD_DIR).extract_text_from_pdf(safe_path)
                        ftype = "pdf"
                    elif ext == "docx":
                        text = DOCXIngestor(UPLOAD_DIR).extract_text_from_docx(safe_path)
                        ftype = "docx"
                    elif ext in ["png", "jpg", "jpeg"]:
                        text = ImageIngestor(UPLOAD_DIR).extract_text_from_image(safe_path)
                        ftype = "image"
                    elif ext in ["wav", "mp3"]:
                        text = AudioIngestor(UPLOAD_DIR).transcribe_audio(safe_path)
                        ftype = "audio"
                    elif ext == "pptx":
                        text = PPTXIngestor(UPLOAD_DIR).extract_text_from_pptx(safe_path)
                        ftype = "pptx"
                    elif ext in ["xls", "xlsx"]:
                        text = ExcelIngestor(UPLOAD_DIR).extract_text_from_excel(safe_path)
                        ftype = "excel"
                    else:
                        st.warning(f"Unsupported file type: {upload.name}")
                        continue

                    if text and text.strip():
                        all_docs[upload.name] = {"text": text, "file_type": ftype}
                        logging.info(f"Ingested {upload.name} -> {len(text)} chars")
                    else:
                        st.warning(f"No text extracted from {upload.name}. Skipping ingestion.")

                except Exception as e:
                    st.error(f"Failed to ingest {upload.name}: {e}")

            if all_docs:
                vs.add_documents(all_docs)
                st.session_state["ingested_files"].extend(list(all_docs.keys()))
                st.success(f"✅ Ingested {len(all_docs)} file(s) successfully!")

    # show only the uploaded documents list separately
    st.subheader("📄 Uploaded Documents")
    if st.session_state["ingested_files"]:
        for file in st.session_state["ingested_files"]:
            st.markdown(f"- {file}")
    else:
        st.info("No documents uploaded yet.")



# ==========================================================
# TAB 2: Query Interface
# ==========================================================
with tab2:
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=10, value=3, step=1)
    st.header("💬 Ask a Question")

    # --- Query input appears immediately after heading ---
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    query = st.text_input(
        "Type your question and press Enter:",
        key="query_input",
        placeholder="Ask anything about your uploaded files..."
    )

    # --- Results container (outputs below query input) ---
    results_container = st.container()

    if query:  # only run query when user presses Enter
        with results_container:
            try:
                retrieved = vs.query(query, top_k=st.session_state.get("top_k", 3))
            except ValueError:
                st.warning("⚠️ No documents found in vector store. Upload and ingest files first.")
                retrieved = []
            except Exception as e:
                st.error(f"Query failed: {e}")
                retrieved = []

            if retrieved:
                with st.expander("📚 Retrieved Context", expanded=True):
                    for i, item in enumerate(retrieved, 1):
                        source_name = item.get("source", "unknown")
                        file_type = item.get("file_type", "text")
                        st.markdown(f"**[{i}] Source:** {source_name} ({file_type})")
                        st.text(item.get("snippet", ""))

                        stored_path = os.path.join(UPLOAD_DIR, source_name)
                        if file_type == "image" and os.path.exists(stored_path):
                            st.image(stored_path, caption=source_name, use_column_width=True)
                        elif file_type == "audio" and os.path.exists(stored_path):
                            try:
                                st.audio(stored_path)
                            except Exception as e:
                                st.warning(f"Could not play audio {source_name}: {e}")

                try:
                    llm = OllamaInference(model_name="phi:2.7b")
                    context_texts = [item["text"] for item in retrieved if item.get("text")]
                    answer = llm.generate_answer(prompt=query, context=context_texts)
                    with st.expander("🧩 Generated Answer", expanded=True):
                        st.markdown(answer, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"LLM inference failed: {e}")
            else:
                st.info("No relevant context found for your query. Consider uploading related files.")
