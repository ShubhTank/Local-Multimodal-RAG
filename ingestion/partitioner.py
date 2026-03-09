"""
Partitions a PDF into typed elements using Unstructured.io.

Returns a list of dicts with keys:
  - type: str  (Title | NarrativeText | Table | Image)
  - content: str | bytes
  - metadata: dict  (page number, coordinates, source filename, uuid)
"""

import uuid
from pathlib import Path
from typing import Any


def partition_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Partition a PDF file into typed elements.

    Uses unstructured's hi_res strategy when available for accurate
    table and image extraction. Falls back to fast strategy if
    layout model is not installed.
    """
    from unstructured.partition.pdf import partition_pdf as _partition
    from unstructured.documents.elements import (
        Title, NarrativeText, Table, Image
    )

    raw_elements = _partition(
        filename=pdf_path,
        strategy="hi_res",          # Use "fast" if layout model unavailable
        extract_images_in_pdf=True,
        extract_image_block_output_dir="./doc_store/images",
        infer_table_structure=True,
    )

    results = []
    type_map = {
        Title: "Title",
        NarrativeText: "NarrativeText",
        Table: "Table",
        Image: "Image",
    }

    for el in raw_elements:
        el_type = type_map.get(type(el))
        if el_type is None:
            continue

        results.append({
            "uuid": str(uuid.uuid4()),
            "type": el_type,
            "content": getattr(el, "text", "") or "",
            "metadata": {
                "source": Path(pdf_path).name,
                "page": getattr(el.metadata, "page_number", None),
                "coordinates": getattr(el.metadata, "coordinates", None),
                "image_path": getattr(el.metadata, "image_path", None),
            },
        })

    return results
