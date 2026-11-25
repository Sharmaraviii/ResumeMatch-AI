"""Lazy local model initialization and bounded Ollama requests."""
from functools import lru_cache
import os
import ipaddress
from urllib.parse import urlsplit
import requests
import numpy as np

MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')

@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)

class LocalEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(embedding_model().encode(texts, batch_size=16, show_progress_bar=False), dtype=np.float32)


def ollama_generate(prompt: str) -> str:
    # URL comes from trusted deployment configuration, not user input.
    parts = urlsplit(OLLAMA_URL)
    if parts.scheme != 'http' or not parts.hostname or parts.username or parts.password or parts.path not in ('', '/'):
        raise RuntimeError('Invalid Ollama endpoint configuration')
    # Allow local or explicitly configured container service name only.
    host = parts.hostname
    if host not in ('localhost', 'ollama'):
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise RuntimeError('Ollama must be local or on the internal Docker network')
        except ValueError as exc:
            raise RuntimeError('Ollama hostname not permitted') from exc
    try:
        response = requests.post(OLLAMA_URL.rstrip('/') + '/api/generate', json={
            'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 600}
        }, timeout=(5, 150))
        response.raise_for_status()
        answer = response.json().get('response', '').strip()
        if not answer:
            raise RuntimeError('Local model returned an empty response')
        return answer
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError('Ollama unavailable; check local service and installed model') from exc
