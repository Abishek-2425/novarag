# vector_store.py
import os
import pickle
import time
import logging
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
class VectorStore:
    """FAISS vector store for multimodal documents (RAG-ready)."""

    def __init__(self,
                 index_path: str = "embeddings/faiss_index.index",
                 meta_path: str = "embeddings/meta.pkl",
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str ="cpu"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.device = device

        self.model = None  # Lazy-load
        self.index = None
        self.metadata = []
        self._load_index()

    def _load_model(self):
        if self.model is None:
            logging.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logging.error(f"Model load failed ({e}), retrying on CPU...")
                self.model = SentenceTransformer(self.model_name, device="cpu")

    # ----------------------------
    # Index Management
    # ----------------------------
    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logging.info("Loading existing FAISS index and metadata...")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logging.info(f"Loaded {len(self.metadata)} documents from vector store.")
        else:
            logging.info("No existing index found. Starting fresh.")
            self.index = None
            self.metadata = []

    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logging.info(f"Vector store saved with {len(self.metadata)} unique documents.")

    # ----------------------------
    # Public Methods
    # ----------------------------
    def clear(self):
        """Completely clears the FAISS index and metadata."""
        self.index = None
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        logging.info("✅ Vector store fully cleared (index + metadata).")

    def add_documents(self, data: Dict[str, Dict[str, str]], force_reingest: bool = False):
        """Add documents to the vector store."""
        self._load_model()

        texts, new_metadata = [], []
        existing_sources = {m["source"] for m in self.metadata}

        for fname, info in data.items():
            text = info.get("text", "").strip()
            file_type = info.get("file_type", "text")
            if not text:
                continue

            if not force_reingest and fname in existing_sources:
                logging.warning(f"Skipping duplicate source: {fname}")
                continue

            texts.append(text)
            new_metadata.append({
                "source": fname,
                "text": text,
                "file_type": file_type,
                "ingested_at": time.time()
            })

        if not texts:
            logging.warning("No new text data to add.")
            return

        logging.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        if self.index is None:
            logging.info("Creating new FAISS index...")
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        else:
            logging.info("Adding to existing FAISS index...")

        self.index.add(embeddings)
        self.metadata.extend(new_metadata)
        self._save_index()

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve top_k relevant documents for a given query."""
        self._load_model()

        if self.index is None or not self.metadata:
            raise ValueError("FAISS index not built yet. Ingest documents first.")

        top_k = int(top_k)
        top_k = max(1, min(top_k, len(self.metadata)))

        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        seen_sources = set()
        for idx in indices[0]:
            if idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                src = doc["source"]
                if src in seen_sources:
                    continue
                seen_sources.add(src)
                doc["snippet"] = doc["text"][:500] + ("..." if len(doc["text"]) > 500 else "")
                results.append(doc)

        logging.info(f"Retrieved {len(results)} docs for query: '{query_text}'")
        return results
