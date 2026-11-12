"""
Multimodal RAG Pipeline — CLI entrypoint.

Usage:
  python main.py ingest --pdf path/to/document.pdf
  python main.py query --question "Who is Bhushan Dangre?"
  python main.py evaluate --dataset data/eval_dataset.json

Local stack:
  - Ollama for embeddings and LLM/vision
  - Qdrant for vector storage
  - Unstructured for PDF partitioning
  - python-dotenv for environment configuration

The pipeline is document-agnostic and can be used with:
  - resumes
  - internship reports
  - research papers
  - books
  - manuals
  - policies
  - technical documentation
  - other PDF documents
"""

import argparse
import os
import sys

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


# ============================================================
# INGEST
# ============================================================


def ingest(pdf_path: str) -> None:
    """
    Partition, process, embed, and index a PDF document.

    The ingestion pipeline supports:
        - normal text
        - tables
        - images
        - image summaries
    """

    from ingestion.partitioner import partition_pdf
    from ingestion.table_converter import convert_tables
    from ingestion.image_summariser import summarise_images

    from retrieval.embedder import get_embedder
    from retrieval.vector_store import upsert_documents
    from retrieval.document_store import save_raw_element

    # --------------------------------------------------------
    # Validate PDF path
    # --------------------------------------------------------

    if not os.path.exists(pdf_path):
        print(
            f"[ingest] ERROR: PDF file not found:"
        )
        print(
            f"         {pdf_path}"
        )
        sys.exit(1)

    print()
    print("=" * 60)
    print("===== MULTIMODAL RAG INGESTION =====")
    print("=" * 60)

    print()
    print(
        f"[ingest] PDF: {pdf_path}"
    )

    # --------------------------------------------------------
    # 1. Partition PDF
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Partitioning PDF..."
    )

    elements = partition_pdf(
        pdf_path
    )

    print(
        f"[ingest] Partitioned "
        f"{len(elements)} raw elements."
    )

    # --------------------------------------------------------
    # 2. Separate text, tables, and images
    # --------------------------------------------------------

    texts = []
    tables = []
    images = []

    for element in elements:

        element_type = element.get(
            "type",
            "",
        )

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        if element_type == "Table":

            tables.append(
                element
            )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        elif element_type == "Image":

            images.append(
                element
            )

        # ----------------------------------------------------
        # Text-based elements
        # ----------------------------------------------------

        else:

            content = (
                element.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if content:

                texts.append(
                    element
                )

    print(
        f"[ingest] Text elements : "
        f"{len(texts)}"
    )

    print(
        f"[ingest] Table elements: "
        f"{len(tables)}"
    )

    print(
        f"[ingest] Image elements: "
        f"{len(images)}"
    )

    # --------------------------------------------------------
    # 3. Convert tables
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Processing tables..."
    )

    table_docs = convert_tables(
        tables
    )

    print(
        f"[ingest] Converted "
        f"{len(table_docs)} tables."
    )

    # --------------------------------------------------------
    # 4. Summarise images
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Processing images..."
    )

    image_docs = summarise_images(
        images
    )

    print(
        f"[ingest] Summarised "
        f"{len(image_docs)} images."
    )

    # --------------------------------------------------------
    # 5. Combine all documents
    # --------------------------------------------------------

    all_docs = (
        texts
        + table_docs
        + image_docs
    )

    print()
    print(
        f"[ingest] Total documents to index: "
        f"{len(all_docs)}"
    )

    # --------------------------------------------------------
    # Validate extracted content
    # --------------------------------------------------------

    if not all_docs:

        print()
        print(
            "[ingest] ERROR:"
        )

        print(
            "[ingest] No usable content was extracted "
            "from the PDF."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # 6. Create embedding model
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Creating embedding model..."
    )

    embedder = get_embedder()

    # --------------------------------------------------------
    # 7. Store embeddings in Qdrant
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Indexing documents in Qdrant..."
    )

    upsert_documents(
        all_docs,
        embedder,
    )

    # --------------------------------------------------------
    # 8. Persist raw tables and images
    # --------------------------------------------------------

    print()
    print(
        "[ingest] Saving raw tables/images..."
    )

    for document in (
        table_docs
        + image_docs
    ):

        save_raw_element(
            document
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "[ingest] INGESTION COMPLETE"
    )
    print("=" * 60)

    print(
        f"[ingest] Indexed documents: "
        f"{len(all_docs)}"
    )

    print(
        f"[ingest] Collection: "
        f"{os.getenv('COLLECTION_NAME', 'rag_documents')}"
    )

    print(
        f"[ingest] Embedding model: "
        f"{os.getenv('EMBED_MODEL', 'nomic-embed-text')}"
    )

    print()


# ============================================================
# QUERY
# ============================================================


def query(
    question: str,
    top_k: int = 12,
) -> None:
    """
    Retrieve relevant evidence and generate a grounded answer.

    The question is passed directly to the generic retriever.

    No document-specific or question-specific retrieval
    logic is implemented here.
    """

    from retrieval.embedder import get_embedder
    from retrieval.multi_vector_retriever import retrieve
    from generation.rag_chain import generate_answer

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = (
        question or ""
    ).strip()

    if not question:

        print(
            "[query] ERROR: Question cannot be empty."
        )

        sys.exit(1)

    print()
    print("=" * 60)
    print("===== MULTIMODAL RAG QUERY =====")
    print("=" * 60)

    print()
    print(
        f"[query] Question: {question}"
    )

    # --------------------------------------------------------
    # 1. Create embedding model
    # --------------------------------------------------------

    print()
    print(
        "[query] Creating embedding model..."
    )

    embedder = get_embedder()

    # --------------------------------------------------------
    # 2. Retrieve evidence
    # --------------------------------------------------------

    print()
    print(
        "[query] Retrieving relevant context..."
    )

    context = retrieve(
        question,
        embedder,
        k=top_k,
    )

    if not context:

        print()
        print(
            "[query] No relevant context found."
        )

        print(
            "[query] The document may not contain "
            "information relevant to this question."
        )

        return

    print(
        f"[query] Retrieved "
        f"{len(context)} context elements."
    )

    # --------------------------------------------------------
    # 3. Debug retrieved context
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("===== RETRIEVED EVIDENCE =====")
    print("=" * 60)

    for index, item in enumerate(
        context,
        start=1,
    ):

        item_type = item.get(
            "type",
            "Unknown",
        )

        content = (
            item.get(
                "content",
                "",
            )
            or ""
        ).strip()

        print()
        print(
            f"--- EVIDENCE {index} ---"
        )

        print(
            f"Type: {item_type}"
        )

        print(
            "Content:"
        )

        print(
            content
        )

    print()
    print("=" * 60)
    print("===== END RETRIEVED EVIDENCE =====")
    print("=" * 60)

    # --------------------------------------------------------
    # 4. Generate answer
    # --------------------------------------------------------

    print()
    print(
        "[query] Generating answer using Ollama..."
    )

    answer = generate_answer(
        question,
        context,
    )

    # --------------------------------------------------------
    # 5. Display answer
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("===== ANSWER =====")
    print("=" * 60)

    if answer:

        print()
        print(
            answer
        )

    else:

        print()
        print(
            "[query] The model did not return an answer."
        )

    print()
    print("=" * 60)


# ============================================================
# EVALUATE
# ============================================================


def evaluate(
    dataset_path: str,
) -> None:
    """
    Run RAGAS evaluation against a labelled dataset.
    """

    from evaluation.ragas_evaluator import run_evaluation

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if not os.path.exists(
        dataset_path
    ):

        print(
            "[evaluate] ERROR: Dataset file not found:"
        )

        print(
            f"             {dataset_path}"
        )

        sys.exit(1)

    print()
    print("=" * 60)
    print("===== RAG EVALUATION =====")
    print("=" * 60)

    print()
    print(
        f"[evaluate] Dataset: "
        f"{dataset_path}"
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    results = run_evaluation(
        dataset_path
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("--- RAGAS Scores ---")

    if not results:

        print(
            "No evaluation results returned."
        )

        return

    for metric, score in results.items():

        try:

            print(
                f"  {metric}: "
                f"{float(score):.3f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                f"  {metric}: "
                f"{score}"
            )

    print()
    print("=" * 60)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    """
    CLI entrypoint.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Local Multimodal RAG Pipeline"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # ========================================================
    # INGEST COMMAND
    # ========================================================

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF document",
    )

    ingest_parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF document",
    )

    # ========================================================
    # QUERY COMMAND
    # ========================================================

    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question about the indexed document",
    )

    query_parser.add_argument(
        "--question",
        required=True,
        help="Question to answer",
    )

    query_parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help=(
            "Number of retrieved evidence elements "
            "(default: 12)"
        ),
    )

    # ========================================================
    # EVALUATE COMMAND
    # ========================================================

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run RAGAS evaluation",
    )

    evaluate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the evaluation dataset JSON",
    )

    # ========================================================
    # PARSE ARGUMENTS
    # ========================================================

    args = parser.parse_args()

    # ========================================================
    # EXECUTE COMMAND
    # ========================================================

    if args.command == "ingest":

        ingest(
            args.pdf
        )

    elif args.command == "query":

        query(
            args.question,
            top_k=args.top_k,
        )

    elif args.command == "evaluate":

        evaluate(
            args.dataset
        )

    else:

        parser.print_help()

        sys.exit(1)


# ============================================================
# ENTRYPOINT
# ============================================================


if __name__ == "__main__":

    main()