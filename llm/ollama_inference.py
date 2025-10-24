import logging
from typing import List, Optional
from ollama import chat, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class OllamaInference:
    """
    Local LLM inference using Ollama's official Python API.
    Compatible with retrieval-augmented generation (RAG) context.
    """

    def __init__(self, model_name: str = "phi:2.7b"):
        """
        Initialize an Ollama inference instance.
        Args:
            model_name: The local Ollama model name (e.g., 'phi:2.7b', 'llama3', 'gemma3')
        """
        self.model_name = model_name
        logging.info(f"Initialized OllamaInference with model '{self.model_name}'")

    def generate_answer(
        self, 
        prompt: str, 
        context: Optional[List[str]] = None, 
        stream: bool = False
    ) -> str:
        """
        Generate an answer from the local Ollama model.
        Args:
            prompt: The user question or task prompt.
            context: Optional list of retrieved context strings for RAG.
            stream: If True, stream responses progressively.
        Returns:
            Final model response as a string.
        """
        # Construct the final message for the LLM
        if context:
            context_text = "\n\n".join(context)
            full_prompt = f"Context:\n{context_text}\n\nQuestion:\n{prompt}"
        else:
            full_prompt = prompt

        logging.info(f"Sending request to Ollama model '{self.model_name}'...")

        try:
            if stream:
                # Streamed output for live updates (good for Streamlit or terminal apps)
                response_text = ""
                stream_resp = chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    stream=True
                )
                for chunk in stream_resp:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    response_text += content
                print()  # newline after stream
                return response_text.strip()
            else:
                # Standard non-streaming generation
                response: ChatResponse = chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.message.content.strip()

        except Exception as e:
            logging.error(f"Ollama inference failed: {e}")
            raise RuntimeError(f"Ollama inference failed: {e}")
