"""
Unit tests for the multi-vector retriever.
"""

import pytest
from unittest.mock import MagicMock, patch


def test_retriever_fetches_raw_for_table():
    """
    When a retrieved document is of type Table, the retriever
    should fetch the raw element from the document store.
    """
    from retrieval.multi_vector_retriever import retrieve

    mock_doc = MagicMock()
    mock_doc.page_content = "Revenue table"
    mock_doc.metadata = {"uuid": "test-uuid-123", "type": "Table"}

    raw_element = {
        "uuid": "test-uuid-123",
        "type": "Table",
        "content": "<table><tr><td>Q3</td><td>$1M</td></tr></table>",
        "metadata": {},
    }

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    with patch("retrieval.multi_vector_retriever.get_vector_store", return_value=mock_store), \
         patch("retrieval.multi_vector_retriever.load_raw_element", return_value=raw_element):
        results = retrieve("What is Q3 revenue?", embedder=MagicMock())

    assert results[0]["type"] == "Table"
    assert "<table>" in results[0]["content"]


def test_retriever_uses_text_directly_for_narrative():
    """
    NarrativeText documents should be returned directly from the vector store.
    """
    from retrieval.multi_vector_retriever import retrieve

    mock_doc = MagicMock()
    mock_doc.page_content = "The company was founded in 2010."
    mock_doc.metadata = {"uuid": "test-uuid-456", "type": "NarrativeText"}

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    with patch("retrieval.multi_vector_retriever.get_vector_store", return_value=mock_store):
        results = retrieve("When was the company founded?", embedder=MagicMock())

    assert results[0]["content"] == "The company was founded in 2010."
