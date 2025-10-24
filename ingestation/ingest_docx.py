# ingest_docx.py
import os
from typing import Dict, List, Optional
from docx import Document
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DOCXIngestor:
    """
    Handles ingestion of DOCX files and extraction of text for embeddings.
    Supports optional text chunking for better RAG integration.
    """

    def __init__(self, folder_path: Optional[str] = None, chunk_size: int = 1000):
        """
        :param folder_path: Folder containing DOCX files
        :param chunk_size: Number of characters per chunk
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        if self.folder_path and not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"DOCX folder not found: {self.folder_path}")

    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extracts all text from a single DOCX file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        logging.info(f"Ingesting DOCX: {file_path}")
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logging.error(f"Failed to read DOCX file {file_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks for embedding.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def ingest_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Ingest a single DOCX file and return list of dicts for each chunk:
        [{ 'text': chunk, 'source': filename }]
        """
        text = self.extract_text_from_docx(file_path)
        chunks = self.chunk_text(text)
        return [{"text": chunk, "source": os.path.basename(file_path)} for chunk in chunks]

    def ingest_folder(self) -> List[Dict[str, str]]:
        """
        Extracts text from all DOCX files in the folder and returns as list of dicts.
        Each dict contains a text chunk and the source filename.
        """
        if not self.folder_path:
            raise ValueError("Folder path not set for folder ingestion.")

        all_texts = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".docx"):
                full_path = os.path.join(self.folder_path, filename)
                try:
                    file_chunks = self.ingest_file(full_path)
                    all_texts.extend(file_chunks)
                except Exception as e:
                    logging.error(f"Failed to ingest {filename}: {e}")
        logging.info(f"Total DOCX chunks ingested: {len(all_texts)}")
        return all_texts


# Example usage
if __name__ == "__main__":
    folder = "sample_data/docs"
    docx_ingestor = DOCXIngestor(folder_path=folder)
    chunks = docx_ingestor.ingest_folder()
    for i, item in enumerate(chunks):
        print(f"----- Chunk {i+1} from {item['source']} -----\n{item['text'][:500]}...\n")
