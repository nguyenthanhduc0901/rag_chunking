from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from .io_utils import read_chunks_jsonl
from .text_utils import word_count


def _fallback_keywords(text: str, limit: int = 6) -> list[str]:
    words = [
        word.lower().strip(".,;:!?()[]{}\"'")
        for word in text.split()
        if len(word.strip(".,;:!?()[]{}\"'")) >= 5
    ]
    seen: set[str] = set()
    keywords: list[str] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _ask_gemma_for_question(
    chunk_text: str,
    title: str | None,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> dict:
    prompt = (
        "Create one concise retrieval-evaluation question that can be answered "
        "from the passage. Return strict JSON with keys: question, keywords. "
        "keywords must be 3 to 6 short strings from the passage.\n\n"
        f"Book title: {title or 'Unknown'}\n\n"
        f"Passage:\n{chunk_text[:2200]}"
    )
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": "Bearer local-gemma"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You create JSON-only retrieval benchmark questions.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Gemma did not return JSON: {content}")
    return json.loads(content[start : end + 1])


def generate_questions(
    artifact_dir: Path,
    out_path: Path,
    limit: int,
    min_words: int,
    stride: int,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> None:
    chunks = [
        chunk
        for chunk in read_chunks_jsonl(artifact_dir / "chunks.jsonl")
        if word_count(chunk.text) >= min_words
    ]
    selected = chunks[:: max(1, stride)][:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for index, chunk in enumerate(selected, start=1):
            try:
                generated = _ask_gemma_for_question(
                    chunk_text=chunk.text,
                    title=chunk.title,
                    base_url=base_url,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
                question = str(generated["question"]).strip()
                keywords = [str(item).strip() for item in generated.get("keywords", [])]
            except Exception:
                keywords = _fallback_keywords(chunk.text)
                question = (
                    f"What does {chunk.title or 'the selected book'} say about "
                    f"{', '.join(keywords[:3])}?"
                )

            row = {
                "question": question,
                "expected_doc_id": chunk.doc_id,
                "expected_chunk_id": chunk.chunk_id,
                "expected_keywords": keywords,
                "source_title": chunk.title,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(selected)}] {chunk.chunk_id} -> {question}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic retrieval questions.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval/synthetic_questions.jsonl"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gemma-url", default="http://localhost:8000")
    parser.add_argument("--gemma-model", default="gemma-4-e4b-it")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    generate_questions(
        artifact_dir=args.artifact_dir,
        out_path=args.out,
        limit=args.limit,
        min_words=args.min_words,
        stride=args.stride,
        base_url=args.gemma_url,
        model=args.gemma_model,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()

