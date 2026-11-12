"""
Embedding model wrapper.

Swap out the model by changing EMBED_MODEL in .env.
"""

import os

from langchain_ollama import OllamaEmbeddings


def get_embedder() -> OllamaEmbeddings:
    model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model)