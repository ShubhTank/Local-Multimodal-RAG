"""
Partitions a PDF into typed elements using Unstructured.io.

Keeps all text-based elements so that resume sections such as
skills, bullet points, projects, and experience are not lost.
"""

import uuid
from pathlib import Path
from typing import Any


def partition_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Partition a PDF into typed elements.

    Text-based Unstructured elements are preserved, including:
    Title, NarrativeText, ListItem, Text, and other elements
    containing textual content.
    """

    from unstructured.partition.pdf import partition_pdf as _partition

    image_output_dir = Path("./doc_store/images")
    image_output_dir.mkdir(parents=True, exist_ok=True)

    raw_elements = _partition(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(image_output_dir),
        infer_table_structure=True,
    )

    results: list[dict[str, Any]] = []

    for el in raw_elements:
        element_name = type(el).__name__
        content = getattr(el, "text", "") or ""

        metadata = getattr(el, "metadata", None)

        # Keep text-based elements.
        if content.strip():
            results.append(
                {
                    "uuid": str(uuid.uuid4()),
                    "type": element_name,
                    "content": content,
                    "metadata": {
                        "source": Path(pdf_path).name,
                        "page": getattr(
                            metadata,
                            "page_number",
                            None,
                        ),
                    },
                }
            )

        # Keep images even when they have no text.
        elif element_name == "Image":
            results.append(
                {
                    "uuid": str(uuid.uuid4()),
                    "type": "Image",
                    "content": "",
                    "metadata": {
                        "source": Path(pdf_path).name,
                        "page": getattr(
                            metadata,
                            "page_number",
                            None,
                        ),
                        "image_path": getattr(
                            metadata,
                            "image_path",
                            None,
                        ),
                    },
                }
            )

    return results