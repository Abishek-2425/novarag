# ingest_pdf.py
import fitz  # PyMuPDF
import os
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PDFIngestor:
    """
    Handles ingestion of PDFs and extraction of text for embeddings.
    Can ingest a single file or an entire folder.
    Supports optional chunking for RAG workflows.
    """

    def __init__(self, folder_path: Optional[str] = None, chunk_size: int = 1000):
        """
        :param folder_path: Folder containing PDFs to ingest
        :param chunk_size: Number of characters per chunk for splitting text
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        if self.folder_path and not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"PDF folder not found: {self.folder_path}")

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extracts all text from a single PDF file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logging.info(f"Ingesting PDF: {file_path}")
        text = ""
        try:
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf):
                    page_text = page.get_text()
                    text += page_text + "\n"
        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}: {e}")
            return ""
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks for better RAG embedding performance.
        """
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return chunks

    def ingest_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Ingest a single PDF file and return list of dicts for each chunk:
        [{ 'text': chunk, 'source': filename }]
        """
        text = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text)
        return [{"text": chunk, "source": os.path.basename(file_path)} for chunk in chunks]

    def ingest_folder(self) -> List[Dict[str, str]]:
        """
        Extracts text from all PDFs in the folder and returns as list of dicts.
        Each dict contains a text chunk and the source filename.
        """
        if not self.folder_path:
            raise ValueError("Folder path not set for folder ingestion.")

        all_texts = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.folder_path, filename)
                try:
                    file_chunks = self.ingest_file(full_path)
                    all_texts.extend(file_chunks)
                except Exception as e:
                    logging.error(f"Failed to ingest {filename}: {e}")
        logging.info(f"Total PDF chunks ingested: {len(all_texts)}")
        return all_texts


# Example usage
if __name__ == "__main__":
    folder = "sample_data/pdfs"
    pdf_ingestor = PDFIngestor(folder_path=folder)
    chunks = pdf_ingestor.ingest_folder()
    for i, item in enumerate(chunks):
        print(f"----- Chunk {i+1} from {item['source']} -----\n{item['text'][:500]}...\n")
