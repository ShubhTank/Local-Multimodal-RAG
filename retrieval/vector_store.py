"""
Qdrant vector store management.

This module is document-agnostic.

Responsibilities:
    1. Connect to the local/remote Qdrant instance.
    2. Create the collection if it does not exist.
    3. Configure the vector dimension from the environment.
    4. Store document embeddings with metadata.
    5. Keep UUIDs so raw tables/images can be retrieved later.
"""

import os
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ============================================================
# QDRANT CLIENT
# ============================================================


def _get_client() -> QdrantClient:
    """
    Create and return a Qdrant client.

    Defaults to local Qdrant running on port 6333.
    """

    qdrant_url = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )

    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY"
    )

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key or None,
    )


# ============================================================
# COLLECTION CONFIGURATION
# ============================================================


def _get_collection_name() -> str:
    """
    Return the configured Qdrant collection name.
    """

    return os.getenv(
        "COLLECTION_NAME",
        "rag_documents",
    )


def _get_embedding_dimension() -> int:
    """
    Return the embedding vector dimension.

    Default:
        768

    This matches:
        nomic-embed-text
    """

    return int(
        os.getenv(
            "EMBEDDING_DIMENSION",
            "768",
        )
    )


# ============================================================
# VECTOR STORE
# ============================================================


def get_vector_store(
    embedder,
) -> QdrantVectorStore:
    """
    Return a LangChain Qdrant vector store.

    Creates the collection automatically if it does not exist.
    """

    client = _get_client()

    collection_name = _get_collection_name()

    embedding_dimension = (
        _get_embedding_dimension()
    )

    # --------------------------------------------------------
    # Check existing collections
    # --------------------------------------------------------

    existing_collections = {
        collection.name
        for collection
        in client.get_collections().collections
    }

    # --------------------------------------------------------
    # Create collection if necessary
    # --------------------------------------------------------

    if collection_name not in existing_collections:

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

    # --------------------------------------------------------
    # Return LangChain vector store
    # --------------------------------------------------------

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
    )


# ============================================================
# DOCUMENT UPSERT
# ============================================================


def upsert_documents(
    docs: list[dict[str, Any]],
    embedder,
) -> None:
    """
    Embed and store documents in Qdrant.

    Each document must contain:

        uuid
        type

    The embedding text is selected in this order:

        embed_content
        content

    This allows image summaries, table representations,
    and normal text to use the same vector store.
    """

    if not docs:
        return

    store = get_vector_store(
        embedder
    )

    langchain_documents: list[
        Document
    ] = []

    for doc in docs:

        # ----------------------------------------------------
        # Select text used for embedding
        # ----------------------------------------------------

        embedding_content = (
            doc.get("embed_content")
            or doc.get("content")
            or ""
        ).strip()

        if not embedding_content:
            continue

        # ----------------------------------------------------
        # Build metadata
        # ----------------------------------------------------

        metadata = {
            "uuid": doc["uuid"],
            "type": doc["type"],
            **doc.get("metadata", {}),
        }

        # ----------------------------------------------------
        # Create LangChain document
        # ----------------------------------------------------

        langchain_documents.append(
            Document(
                page_content=embedding_content,
                metadata=metadata,
            )
        )

    # --------------------------------------------------------
    # Nothing valid to insert
    # --------------------------------------------------------

    if not langchain_documents:
        return

    # --------------------------------------------------------
    # Add documents to Qdrant
    # --------------------------------------------------------

    store.add_documents(
        langchain_documents
    )