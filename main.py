"""
Multimodal RAG Pipeline — CLI entrypoint.

Usage:
  python main.py ingest   --pdf path/to/document.pdf
  python main.py query    --question "What does the Q3 chart show?"
  python main.py evaluate --dataset data/eval_dataset.json
"""

import argparse
import sys


def ingest(pdf_path: str) -> None:
    """Partition, embed, and index a PDF document."""
    from ingestion.partitioner import partition_pdf
    from ingestion.table_converter import convert_tables
    from ingestion.image_summariser import summarise_images
    from retrieval.embedder import get_embedder
    from retrieval.vector_store import upsert_documents
    from retrieval.document_store import save_raw_element

    print(f"[ingest] Partitioning {pdf_path} ...")
    elements = partition_pdf(pdf_path)

    texts, tables, images = [], [], []
    for el in elements:
        if el["type"] in ("Title", "NarrativeText"):
            texts.append(el)
        elif el["type"] == "Table":
            tables.append(el)
        elif el["type"] == "Image":
            images.append(el)

    # Convert tables to HTML/Markdown for embedding
    table_docs = convert_tables(tables)

    # Summarise images with vision model
    image_docs = summarise_images(images)

    all_docs = texts + table_docs + image_docs

    embedder = get_embedder()
    upsert_documents(all_docs, embedder)

    # Persist raw tables and images to document store
    for doc in table_docs + image_docs:
        save_raw_element(doc)

    print(f"[ingest] Done. Indexed {len(all_docs)} elements.")


def query(question: str) -> None:
    """Retrieve context and generate a multimodal answer."""
    from retrieval.embedder import get_embedder
    from retrieval.multi_vector_retriever import retrieve
    from generation.rag_chain import generate_answer

    embedder = get_embedder()
    context = retrieve(question, embedder)
    answer = generate_answer(question, context)
    print("\n--- Answer ---")
    print(answer)


def evaluate(dataset_path: str) -> None:
    """Run RAGAS evaluation against a labelled dataset."""
    from evaluation.ragas_evaluator import run_evaluation
    results = run_evaluation(dataset_path)
    print("\n--- RAGAS Scores ---")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF document")
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF file")

    p_query = sub.add_parser("query", help="Query the pipeline")
    p_query.add_argument("--question", required=True, help="Question to answer")

    p_eval = sub.add_parser("evaluate", help="Run RAGAS evaluation")
    p_eval.add_argument("--dataset", required=True, help="Path to eval dataset JSON")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf)
    elif args.command == "query":
        query(args.question)
    elif args.command == "evaluate":
        evaluate(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
