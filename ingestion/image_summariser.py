"""
Summarises chart/figure images using a local Ollama vision model.

The summary is used as the embedding text; the raw image is stored
separately in the document store, linked by UUID.
"""

import base64
import os
from pathlib import Path
from typing import Any

from ollama import chat


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for the Ollama vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def summarise_images(
    image_elements: list[dict[str, Any]],
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call the local Ollama vision model on each image element to generate a
    searchable text summary.

    Returns elements with an added 'embed_content' key containing
    the vision model's description.
    """

    vision_model = model or os.getenv(
        "VISION_MODEL",
        "moondream:1.8b",
    )

    summarised: list[dict[str, Any]] = []

    for el in image_elements:
        image_path = el.get("metadata", {}).get("image_path")

        if not image_path or not Path(image_path).exists():
            # No image file — embed a placeholder.
            el["embed_content"] = "[Image: no extractable content]"
            el["vision_summary"] = "[Image: no extractable content]"
            summarised.append(el)
            continue

        try:
            # Read and encode the image.
            encoded = _encode_image(image_path)

            response = chat(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Describe this image or chart in detail. "
                            "If it is a chart or graph, describe the data it shows, "
                            "including axis labels, trends, key values, and any title. "
                            "If it is a diagram, explain the important components "
                            "and relationships shown. "
                            "Be specific and factual. "
                            "Do not invent information that cannot be observed."
                        ),
                        "images": [encoded],
                    }
                ],
            )

            summary = response["message"]["content"].strip()

        except Exception as e:
            summary = f"[Vision model error: {str(e)}]"

        summarised.append(
            {
                **el,
                "embed_content": f"Figure description: {summary}",
                "vision_summary": summary,
            }
        )

    return summarised