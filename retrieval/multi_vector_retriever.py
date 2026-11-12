"""
Generic semantic retriever.

Retrieves relevant document chunks from Qdrant and prepares
clean evidence for the generation layer.

This retriever is document-agnostic and question-type-agnostic.
It does not contain hard-coded keywords for resumes, skills,
companies, departments, projects, etc.
"""

from typing import Any

from retrieval.document_store import load_raw_element


def _clean_content(content: str) -> str:
    """Clean obvious extraction artifacts."""

    if not content:
        return ""

    content = content.strip()

    if content in {
        "|",
        "||",
        "|||",
        "—",
        "-",
    }:
        return ""

    return content


def _build_result(doc) -> dict[str, Any] | None:
    """
    Convert a Qdrant/LangChain document into the common
    context structure used by the generation layer.
    """

    doc_uuid = doc.metadata.get("uuid")

    doc_type = doc.metadata.get(
        "type",
        "Text",
    )

    content = _clean_content(
        doc.page_content or ""
    )

    if not content or len(content) < 3:
        return None

    # ---------------------------------------------------------
    # Tables / Images
    # ---------------------------------------------------------

    if doc_type in (
        "Table",
        "Image",
    ):

        if not doc_uuid:
            return None

        raw = load_raw_element(
            doc_uuid
        )

        if not raw:
            return None

        raw_content = _clean_content(
            raw.get("content", "")
        )

        if not raw_content:
            return None

        return {
            **raw,
            "retrieval_score": getattr(
                doc,
                "metadata",
                {},
            ).get(
                "score"
            ),
        }

    # ---------------------------------------------------------
    # Normal text
    # ---------------------------------------------------------

    return {
        "uuid": doc_uuid,
        "type": doc_type,
        "content": content,
        "metadata": doc.metadata,
    }


def retrieve(
    question: str,
    embedder,
    k: int = 12,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant evidence for any user question.

    The question is passed directly to the embedding model.
    No question-specific keyword rules are applied.

    A larger candidate pool is retrieved first, then cleaned
    and deduplicated before returning the final context.
    """

    from retrieval.vector_store import get_vector_store

    store = get_vector_store(
        embedder
    )

    # Retrieve more candidates than we finally return.
    candidate_count = max(
        k * 3,
        30,
    )

    results = store.similarity_search(
        question,
        k=candidate_count,
    )

    context: list[dict[str, Any]] = []

    seen = set()

    for doc in results:

        result = _build_result(doc)

        if not result:
            continue

        content = _clean_content(
            result.get(
                "content",
                "",
            )
        )

        if not content:
            continue

        # -----------------------------------------------------
        # Deduplicate
        # -----------------------------------------------------

        key = content.lower()

        if key in seen:
            continue

        seen.add(key)

        # -----------------------------------------------------
        # Add result
        # -----------------------------------------------------

        result["content"] = content

        context.append(
            result
        )

        if len(context) >= k:
            break

    return context