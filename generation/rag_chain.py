
"""
Generic document question-answering chain.

Uses a local Ollama model to generate concise,
grounded answers from retrieved document evidence.

Supports different document types:
- Resumes
- Internship reports
- Research papers
- Technical documentation
- Manuals
- Business documents
"""

import os
from typing import Any

from langchain_ollama import ChatOllama


def _prepare_context(
    context: list[dict[str, Any]],
) -> str:
    """
    Prepare retrieved evidence for the LLM.

    Prioritizes descriptive content over isolated headings
    and removes empty or irrelevant extraction artifacts.
    """

    evidence = []

    for item in context:
        content = (
            item.get("content", "") or ""
        ).strip()

        if not content:
            continue

        # Remove extraction artifacts.
        if content in {"|", "||", "|||", "and"}:
            continue

        item_type = item.get("type", "Text")

        priority = {
            "NarrativeText": 0,
            "ListItem": 1,
            "Table": 2,
            "Text": 3,
            "Title": 4,
            "Header": 5,
            "Footer": 6,
        }.get(item_type, 3)

        evidence.append({
            "type": item_type,
            "content": content,
            "priority": priority,
            "length": len(content),
        })

    # Put descriptive content before isolated headings.
    evidence.sort(
        key=lambda x: (
            x["priority"],
            -x["length"],
        )
    )

    parts = []

    for index, item in enumerate(evidence, start=1):
        parts.append(
            f"""
EVIDENCE {index}
TYPE: {item["type"]}
CONTENT: {item["content"]}
""".strip()
        )

    return "\n\n".join(parts)


def generate_answer(
    question: str,
    context: list[dict[str, Any]],
) -> str:
    """
    Generate a concise, document-grounded answer.
    """

    model = os.getenv(
        "LLM_MODEL",
        "qwen2.5:1.5b",
    )

    llm = ChatOllama(
        model=model,
        temperature=0,
        num_ctx=4096,
        num_predict=150,
    )

    context_text = _prepare_context(context)

    if not context_text:
        return (
            "The information is not available "
            "in the document."
        )

    system_prompt = """
You are a concise document question-answering assistant.

Your job is to answer the user's question using the
information provided in the document evidence.

STRICT RULES:

1. Answer the exact question directly.
2. Use only facts supported by the evidence.
3. Write one clear, grammatically correct sentence.
4. Include only information necessary to answer the question.
5. Do not mention evidence numbers or context sections.
6. Do not explain where the information was found.
7. Do not describe your reasoning.
8. Do not repeat the same information.
9. Do not add unnecessary introductions or conclusions.
10. Never invent or assume information.
11. If the answer is unavailable, say:
    "The information is not available in the document."

OUTPUT FORMAT:

Return ONLY the final answer.
Prefer one short sentence.
Do not include labels such as "Answer:".
"""

    user_prompt = f"""
DOCUMENT EVIDENCE:

{context_text}

QUESTION:

{question}

Answer the question directly using the evidence.

Return only one concise, complete sentence.
Do not mention evidence numbers or explain your reasoning.

FINAL ANSWER:
"""

    try:
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        answer = (
            response.content or ""
        ).strip()

        if not answer:
            return (
                "The information is not available "
                "in the document."
            )

        # Remove common unnecessary answer prefixes.
        prefixes = [
            "Answer:",
            "Final answer:",
            "The answer is:",
        ]

        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        return answer

    except Exception as error:
        print(f"[rag_chain] Generation error: {error}")

        return (
            "Unable to generate an answer. "
            "Please check whether Ollama is running."
        )
