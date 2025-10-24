# ingest_excel.py
import os
import logging
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ExcelIngestor:
    """
    Handles ingestion of Excel files (.xls, .xlsx) and extraction of text for embeddings.
    Supports single file or folder ingestion.
    Mirrors PDF/DOCX/PPTX ingestors for seamless integration into multimodal RAG.
    """

    def __init__(self, folder_path: Optional[str] = None, chunk_size: int = 1000):
        """
        :param folder_path: Folder containing Excel files
        :param chunk_size: Number of characters per text chunk
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        if self.folder_path and not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Excel folder not found: {self.folder_path}")

    def extract_text_from_excel(self, file_path: str) -> str:
        """
        Extracts all text from an Excel file (all sheets concatenated).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        logging.info(f"Ingesting Excel: {file_path}")
        text = ""
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # Convert all cells to string and join with separators
                sheet_text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist())
                text += f"\n--- Sheet: {sheet_name} ---\n{sheet_text}\n"
        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}: {e}")
            return ""
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks for embedding.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def ingest_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Ingest a single Excel file and return list of dicts for each chunk:
        [{ 'text': chunk, 'source': filename }]
        """
        text = self.extract_text_from_excel(file_path)
        chunks = self.chunk_text(text)
        return [{"text": chunk, "source": os.path.basename(file_path)} for chunk in chunks]

    def ingest_folder(self) -> List[Dict[str, str]]:
        """
        Extracts text from all Excel files in the folder and returns as list of dicts.
        Each dict contains a text chunk and the source filename.
        """
        if not self.folder_path:
            raise ValueError("Folder path not set for folder ingestion.")

        all_texts = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith((".xls", ".xlsx")):
                full_path = os.path.join(self.folder_path, filename)
                try:
                    file_chunks = self.ingest_file(full_path)
                    all_texts.extend(file_chunks)
                except Exception as e:
                    logging.error(f"Failed to ingest {filename}: {e}")
        logging.info(f"Total Excel chunks ingested: {len(all_texts)}")
        return all_texts


# Example usage
if __name__ == "__main__":
    folder = "sample_data/excel"
    excel_ingestor = ExcelIngestor(folder_path=folder)
    chunks = excel_ingestor.ingest_folder()
    for i, item in enumerate(chunks):
        print(f"----- Chunk {i+1} from {item['source']} -----\n{item['text'][:500]}...\n")
