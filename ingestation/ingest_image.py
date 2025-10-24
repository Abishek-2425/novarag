# ingest_image.py
import os
from typing import Dict, List, Optional
from PIL import Image
import pytesseract
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ImageIngestor:
    """
    Handles ingestion of image files and extracts text using OCR.
    Supports optional text chunking for RAG.
    """

    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    def __init__(self, folder_path: Optional[str] = None, chunk_size: int = 1000):
        """
        :param folder_path: Folder containing images
        :param chunk_size: Number of characters per text chunk
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        if self.folder_path and not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Image folder not found: {self.folder_path}")

    def extract_text_from_image(self, file_path: str) -> str:
        """
        Uses OCR to extract text from a single image.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        logging.info(f"Processing image: {file_path}")
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logging.error(f"Failed to process image {file_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks for embeddings.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def ingest_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Ingest a single image and return list of dicts for each text chunk:
        [{ 'text': chunk, 'source': filename }]
        """
        text = self.extract_text_from_image(file_path)
        chunks = self.chunk_text(text)
        return [{"text": chunk, "source": os.path.basename(file_path)} for chunk in chunks]

    def ingest_folder(self) -> List[Dict[str, str]]:
        """
        Extracts text from all supported images in the folder.
        Returns a list of dicts with 'text' and 'source'.
        """
        if not self.folder_path:
            raise ValueError("Folder path not set for folder ingestion.")

        all_texts = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(self.SUPPORTED_EXTENSIONS):
                full_path = os.path.join(self.folder_path, filename)
                try:
                    file_chunks = self.ingest_file(full_path)
                    all_texts.extend(file_chunks)
                except Exception as e:
                    logging.error(f"Failed to ingest {filename}: {e}")
        logging.info(f"Total image text chunks ingested: {len(all_texts)}")
        return all_texts


# Example usage
if __name__ == "__main__":
    folder = "sample_data/images"
    img_ingestor = ImageIngestor(folder_path=folder)
    chunks = img_ingestor.ingest_folder()
    for i, item in enumerate(chunks):
        print(f"----- Chunk {i+1} from {item['source']} -----\n{item['text'][:500]}...\n")
