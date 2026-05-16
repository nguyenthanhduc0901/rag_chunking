from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from .io_utils import read_chunks_jsonl, write_json
from .text_utils import estimate_tokens, split_sentences_with_offsets


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def evaluate_chunks_file(chunks_path: Path) -> dict[str, Any]:
    chunks = read_chunks_jsonl(chunks_path)
    char_lengths = [len(chunk.text) for chunk in chunks]
    token_lengths = [estimate_tokens(chunk.text) for chunk in chunks]
    sentence_counts = [
        len(split_sentences_with_offsets(chunk.text))
        for chunk in chunks
    ]
    boundary_scores = [
        score
        for chunk in chunks
        for score in chunk.metadata.get("boundary_scores", [])
        if isinstance(score, (int, float))
    ]

    docs = {chunk.doc_id for chunk in chunks}
    return {
        "num_chunks": len(chunks),
        "num_documents": len(docs),
        "chars": _summary([float(x) for x in char_lengths]),
        "estimated_tokens": _summary([float(x) for x in token_lengths]),
        "sentences_per_chunk": _summary([float(x) for x in sentence_counts]),
        "chunks_over_512_estimated_tokens": sum(x > 512 for x in token_lengths),
        "chunks_under_80_chars": sum(x < 80 for x in char_lengths),
        "boundary_similarity": _summary([float(x) for x in boundary_scores]),
        "doc_chunk_counts": _summary(
            [
                float(sum(1 for chunk in chunks if chunk.doc_id == doc_id))
                for doc_id in docs
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk artifacts.")
    parser.add_argument("chunks_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_chunks_file(args.chunks_path)
    out = args.out or args.chunks_path.with_name("chunk_eval.json")
    write_json(out, report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

