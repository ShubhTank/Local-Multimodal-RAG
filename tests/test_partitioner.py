"""
Unit tests for the PDF partitioner.
"""

from unittest.mock import MagicMock, patch


def test_partition_returns_typed_elements():
    """
    The partitioner should return dictionaries containing:
    type, content, uuid, and metadata.
    """
    from ingestion.partitioner import partition_pdf
    from unstructured.documents.elements import NarrativeText, Title

    mock_title = MagicMock(spec=Title)
    mock_title.text = "Annual Report 2024"
    mock_title.metadata.page_number = 1
    mock_title.metadata.coordinates = None
    mock_title.metadata.image_path = None

    mock_text = MagicMock(spec=NarrativeText)
    mock_text.text = "Revenue increased by 15% in Q3."
    mock_text.metadata.page_number = 2
    mock_text.metadata.coordinates = None
    mock_text.metadata.image_path = None

    with patch(
        "ingestion.partitioner._partition",
        return_value=[mock_title, mock_text],
    ):
        results = partition_pdf("dummy.pdf")

    assert len(results) == 2

    assert results[0]["type"] == "Title"
    assert results[1]["type"] == "NarrativeText"

    for result in results:
        assert "uuid" in result
        assert "content" in result
        assert "metadata" in result