#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _evaluation_dir(artifacts_dir: Path, chunker_name: str) -> Path:
    return artifacts_dir / "evaluations" / chunker_name


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def collect_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_dir in sorted(artifacts_dir.iterdir() if artifacts_dir.exists() else []):
        if artifact_dir.name in {"evaluations", "answer_quality", "answer_quality_smoke"}:
            continue
        manifest_path = artifact_dir / "manifest.json"
        eval_dir = _evaluation_dir(artifacts_dir, artifact_dir.name)
        chunk_eval_path = _first_existing(eval_dir / "chunk_eval.json", artifact_dir / "chunk_eval.json")
        retrieval_path = _first_existing(eval_dir / "retrieval_eval.json", artifact_dir / "retrieval_eval.json")
        retrieval_top10_path = _first_existing(
            eval_dir / "retrieval_eval_top10.json",
            artifact_dir / "retrieval_eval_top10.json",
        )
        feedback_path = _first_existing(eval_dir / "feedback_report.json", artifact_dir / "feedback_report.json")
        if not manifest_path.exists() or retrieval_path is None:
            continue

        manifest = _load_json(manifest_path)
        chunk_eval = _load_json(chunk_eval_path) if chunk_eval_path else {}
        retrieval = _load_json(retrieval_path)
        retrieval_top10 = (
            _load_json(retrieval_top10_path)
            if retrieval_top10_path
            else {}
        )
        feedback = _load_json(feedback_path) if feedback_path else {}

        top10_key = "source_hit_at_10"
        doc_top10_key = "doc_hit_at_10"
        rows.append(
            {
                "chunker": manifest.get("chunker", artifact_dir.name),
                "chunks": manifest.get("num_chunks"),
                "median_tokens": chunk_eval.get("estimated_tokens", {}).get("median"),
                "over_512": chunk_eval.get("chunks_over_512_estimated_tokens"),
                "doc_hit_at_1": retrieval.get("doc_hit_at_1"),
                "doc_hit_at_5": retrieval.get("doc_hit_at_5"),
                "doc_hit_at_10": retrieval_top10.get(doc_top10_key),
                "doc_mrr": retrieval.get("doc_mrr"),
                "source_hit_at_1": retrieval.get("source_hit_at_1"),
                "source_hit_at_5": retrieval.get("source_hit_at_5"),
                "full_source_hit_at_5": retrieval.get("full_source_hit_at_5"),
                "source_span_recall_at_5": retrieval.get("source_span_recall_at_5"),
                "source_hit_at_10": retrieval_top10.get(top10_key),
                "full_source_hit_at_10": retrieval_top10.get("full_source_hit_at_10"),
                "source_span_recall_at_10": retrieval_top10.get("source_span_recall_at_10"),
                "source_mrr": retrieval.get("source_mrr"),
                "retrieval_issues": feedback.get("summary", {}).get("retrieval_issue_count"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _metric(row, "source_mrr"),
            _metric(row, "source_hit_at_10"),
        ),
        reverse=True,
    )


def write_reports(rows: list[dict[str, Any]], out_dir: Path, questions_file: str) -> None:
    summary = {
        "questions_file": questions_file,
        "ranking": "source_mrr desc, source_hit_at_10 desc",
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    headers = [
        "rank",
        "chunker",
        "chunks",
        "median_tok",
        ">512",
        "doc@1",
        "doc@5",
        "doc@10",
        "doc_mrr",
        "src@1",
        "src@5",
        "src@10",
        "span_recall@5",
        "full_src@5",
        "src_mrr",
        "issues",
    ]
    lines = [
        "# Retrieval Benchmark Summary",
        "",
        f"Questions: `{questions_file}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["chunker"]),
                    str(row.get("chunks", "")),
                    str(row.get("median_tokens", "")),
                    str(row.get("over_512", "")),
                    _pct(row.get("doc_hit_at_1")),
                    _pct(row.get("doc_hit_at_5")),
                    _pct(row.get("doc_hit_at_10")),
                    f"{row['doc_mrr']:.3f}" if row.get("doc_mrr") is not None else "",
                    _pct(row.get("source_hit_at_1")),
                    _pct(row.get("source_hit_at_5")),
                    _pct(row.get("source_hit_at_10")),
                    _pct(row.get("source_span_recall_at_5")),
                    _pct(row.get("full_source_hit_at_5")),
                    f"{row['source_mrr']:.3f}"
                    if row.get("source_mrr") is not None
                    else "",
                    str(row.get("retrieval_issues", "")),
                ]
            )
            + " |"
        )
    (out_dir / "retrieval_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact retrieval comparison.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--questions", default="eval/source_questions.jsonl")
    args = parser.parse_args()

    rows = collect_rows(args.artifacts_dir)
    if not rows:
        print("No retrieval reports found.")
        return
    write_reports(rows, args.artifacts_dir / "evaluations", questions_file=args.questions)


if __name__ == "__main__":
    main()
