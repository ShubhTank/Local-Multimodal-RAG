"""
Unit tests for the RAGAS evaluator.
"""

import json
import os
import tempfile
from unittest.mock import patch


def test_ragas_returns_three_metrics():
    """
    run_evaluation should return faithfulness,
    answer_relevancy, and context_precision.
    """
    from evaluation.ragas_evaluator import run_evaluation

    dataset = [
        {
            "question": "What is the revenue?",
            "answer": "Revenue was $10M.",
            "contexts": ["Revenue was $10M in Q3."],
            "ground_truth": "$10M",
        }
    ]

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as file:
        json.dump(dataset, file)
        tmp_path = file.name

    mock_result = {
        "faithfulness": 0.92,
        "answer_relevancy": 0.88,
        "context_precision": 0.85,
    }

    try:
        with patch(
            "evaluation.ragas_evaluator.evaluate",
            return_value=mock_result,
        ):
            scores = run_evaluation(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert set(scores.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    }

    assert all(
        isinstance(value, float)
        for value in scores.values()
    )