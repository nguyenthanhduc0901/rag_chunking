from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import write_json
from .retrieve import Retriever


def _source_span(row: dict) -> tuple[int, int] | None:
    start = row.get("source_start_char", row.get("expected_source_start_char"))
    end = row.get("source_end_char", row.get("expected_source_end_char"))
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if end_i <= start_i:
        return None
    return start_i, end_i


def _overlap_ratio(
    chunk_start: int,
    chunk_end: int,
    source_start: int,
    source_end: int,
) -> float:
    overlap = max(0, min(chunk_end, source_end) - max(chunk_start, source_start))
    source_length = max(1, source_end - source_start)
    return overlap / source_length


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    if not path.exists():
        return questions
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate(retriever: Retriever, questions: list[dict], top_k: int) -> dict:
    if not questions:
        return {
            "num_questions": 0,
            "note": "No questions found. Generate eval/source_questions.jsonl first.",
        }

    doc_hits_at_1 = 0
    doc_hits_at_k = 0
    source_hits_at_1 = 0
    source_hits_at_k = 0
    doc_reciprocal_ranks: list[float] = []
    source_reciprocal_ranks: list[float] = []
    details = []

    for row in questions:
        question = row["question"]
        expected_doc_id = str(row.get("expected_doc_id", ""))
        source_span = _source_span(row)
        expected_keywords = [kw.lower() for kw in row.get("expected_keywords", [])]
        results = retriever.search(question, top_k=top_k)

        doc_ranks = [
            result.rank
            for result in results
            if expected_doc_id and result.chunk.doc_id == expected_doc_id
        ]
        best_doc_rank = min(doc_ranks) if doc_ranks else None
        if best_doc_rank == 1:
            doc_hits_at_1 += 1
        if best_doc_rank is not None and best_doc_rank <= top_k:
            doc_hits_at_k += 1
            doc_reciprocal_ranks.append(1.0 / best_doc_rank)
        else:
            doc_reciprocal_ranks.append(0.0)

        source_matches = []
        if expected_doc_id and source_span:
            source_start, source_end = source_span
            for result in results:
                if result.chunk.doc_id != expected_doc_id:
                    continue
                overlap = _overlap_ratio(
                    result.chunk.start_char,
                    result.chunk.end_char,
                    source_start,
                    source_end,
                )
                if overlap > 0:
                    source_matches.append((result.rank, overlap))

        best_source_rank = min((rank for rank, _overlap in source_matches), default=None)
        best_source_overlap = max((overlap for _rank, overlap in source_matches), default=0.0)
        if best_source_rank == 1:
            source_hits_at_1 += 1
        if best_source_rank is not None and best_source_rank <= top_k:
            source_hits_at_k += 1
            source_reciprocal_ranks.append(1.0 / best_source_rank)
        elif source_span:
            source_reciprocal_ranks.append(0.0)

        top_text = "\n".join(result.chunk.text.lower() for result in results)
        keyword_hits = sum(1 for kw in expected_keywords if kw in top_text)
        details.append(
            {
                "question_id": row.get("question_id"),
                "question": question,
                "expected_doc_id": expected_doc_id,
                "source_start_char": source_span[0] if source_span else None,
                "source_end_char": source_span[1] if source_span else None,
                "best_doc_rank": best_doc_rank,
                "best_source_rank": best_source_rank,
                "best_source_overlap": best_source_overlap,
                "keyword_recall": (
                    keyword_hits / len(expected_keywords)
                    if expected_keywords
                    else None
                ),
                "top_results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "doc_id": result.chunk.doc_id,
                        "title": result.chunk.title,
                    }
                    for result in results
                ],
            }
        )

    n = len(questions)
    report = {
        "num_questions": n,
        "top_k": top_k,
        "doc_hit_at_1": doc_hits_at_1 / n,
        f"doc_hit_at_{top_k}": doc_hits_at_k / n,
        "doc_mrr": sum(doc_reciprocal_ranks) / n,
        "details": details,
    }
    source_question_count = sum(1 for row in questions if _source_span(row))
    if source_question_count:
        report.update(
            {
                "source_questions": source_question_count,
                "source_hit_at_1": source_hits_at_1 / source_question_count,
                f"source_hit_at_{top_k}": source_hits_at_k / source_question_count,
                "source_mrr": sum(source_reciprocal_ranks) / source_question_count,
            }
        )
    else:
        report.update(
            {
                "hit_at_1": report["doc_hit_at_1"],
                f"hit_at_{top_k}": report[f"doc_hit_at_{top_k}"],
                "mrr": report["doc_mrr"],
                "note": "No source spans found; reported legacy document-level metrics.",
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against JSONL questions.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("eval/source_questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    retriever = Retriever(args.artifact_dir, device=args.device)
    report = evaluate(retriever, load_questions(args.questions), top_k=args.top_k)
    out = args.out or args.artifact_dir / "retrieval_eval.json"
    write_json(out, report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
