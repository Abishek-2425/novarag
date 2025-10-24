# ingest_audio.py
import os
from typing import List, Dict, Optional
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AudioIngestor:
    """
    Handles ingestion of audio files and converts them to text using Faster Whisper.
    Supports optional text chunking for embeddings.
    """

    SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")

    def __init__(self, folder_path: Optional[str] = None, model_size: str = "small", chunk_size: int = 1000):
        """
        :param folder_path: folder containing audio files
        :param model_size: faster-whisper model size ("tiny", "small", "medium", "large")
        :param chunk_size: number of characters per chunk for embeddings
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size

        if self.folder_path and not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Audio folder not found: {self.folder_path}")

        dev = "cpu"
        
        logging.info(f"Loading Faster Whisper model '{model_size}' ({dev})...")
        self.model = WhisperModel(model_size, device=dev)

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes a single audio file to text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logging.info(f"Transcribing audio: {file_path}")
        try:
            segments, _ = self.model.transcribe(file_path)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logging.error(f"Failed to transcribe audio {file_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks for embeddings.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def ingest_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Ingest a single audio file and return list of dicts for each text chunk:
        [{ 'text': chunk, 'source': filename }]
        """
        text = self.transcribe_audio(file_path)
        chunks = self.chunk_text(text)
        return [{"text": chunk, "source": os.path.basename(file_path)} for chunk in chunks]

    def ingest_folder(self) -> List[Dict[str, str]]:
        """
        Transcribes all supported audio files in the folder.
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
        logging.info(f"Total audio text chunks ingested: {len(all_texts)}")
        return all_texts


# Example usage
if __name__ == "__main__":
    folder = "sample_data/audio"
    audio_ingestor = AudioIngestor(folder_path=folder)
    chunks = audio_ingestor.ingest_folder()
    for i, item in enumerate(chunks):
        print(f"----- Chunk {i+1} from {item['source']} -----\n{item['text'][:500]}...\n")
