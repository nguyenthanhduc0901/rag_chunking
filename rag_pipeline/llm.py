from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from .schema import SearchResult


@dataclass(frozen=True)
class Answer:
    text: str
    used_llm: bool


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []
    used_parent_ids: set[str] = set()

    for result in results:
        chunk = result.chunk
        parent_id = chunk.metadata.get("parent_id")
        parent_text = chunk.metadata.get("parent_text")
        sibling_hits = (
            sum(1 for item in results if item.chunk.metadata.get("parent_id") == parent_id)
            if parent_id
            else 0
        )
        if parent_id and parent_text and sibling_hits >= 2:
            if parent_id in used_parent_ids:
                continue
            used_parent_ids.add(parent_id)
            child_ranks = [
                str(item.rank)
                for item in results
                if item.chunk.metadata.get("parent_id") == parent_id
            ]
            context_blocks.append(
                f"[{result.rank}] {chunk.title or 'Untitled'} by {chunk.author or 'Unknown'} "
                f"(doc_id={chunk.doc_id}, score={result.score:.3f}, "
                f"auto_merged_parent=true, child_ranks={','.join(child_ranks)})\n"
                f"{parent_text}"
            )
            continue

        context_blocks.append(
            f"[{result.rank}] {chunk.title or 'Untitled'} by {chunk.author or 'Unknown'} "
            f"(doc_id={chunk.doc_id}, score={result.score:.3f})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the provided Project Gutenberg context. "
        "If the context is insufficient, say that the retrieved context is not enough.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def answer_with_gemma(
    question: str,
    results: list[SearchResult],
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 120.0,
) -> Answer:
    prompt = build_prompt(question, results)
    url = (base_url or os.environ.get("GEMMA_BASE_URL") or "http://localhost:8000").rstrip("/")
    model_name = model or os.environ.get("GEMMA_MODEL") or "gemma-4-e4b-it"

    try:
        response = httpx.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": "Bearer local-gemma"},
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful RAG assistant for Project Gutenberg books. "
                            "Answer only from the provided context. Cite the bracketed "
                            "context numbers when useful. If the context is insufficient, "
                            "say so clearly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return Answer(text=text, used_llm=True)
    except Exception as exc:
        return Answer(
            text=(
                "Gemma4 is not reachable or returned an error. Retrieval still worked, "
                "so use the chunks below as context.\n\n"
                f"LLM error: {type(exc).__name__}: {exc}"
            ),
            used_llm=False,
        )
