from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_chunks_jsonl, read_json, write_json
from .text_utils import estimate_tokens


def default_evaluation_dir(artifact_dir: Path) -> Path:
    return artifact_dir.parent / "evaluations" / artifact_dir.name


def build_feedback_report(
    artifact_dir: Path,
    retrieval_eval_path: Path | None,
    max_token_budget: int,
) -> dict:
    chunks = read_chunks_jsonl(artifact_dir / "chunks.jsonl")
    manifest = read_json(artifact_dir / "manifest.json")
    overlong = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "estimated_tokens": estimate_tokens(chunk.text),
            "suggested_action": "split",
        }
        for chunk in chunks
        if estimate_tokens(chunk.text) > max_token_budget
    ]
    near_boundary = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "suggested_action": chunk.metadata.get("suggested_action"),
        }
        for chunk in chunks
        if chunk.metadata.get("near_boundary")
    ]

    retrieval_findings = []
    if retrieval_eval_path and retrieval_eval_path.exists():
        report = read_json(retrieval_eval_path)
        for detail in report.get("details", []):
            best_rank = detail.get(
                "best_source_rank",
                detail.get("best_doc_rank", detail.get("best_rank")),
            )
            keyword_recall = detail.get("keyword_recall")
            if best_rank is None or best_rank > 3 or (
                keyword_recall is not None and keyword_recall < 0.5
            ):
                retrieval_findings.append(
                    {
                        "question": detail.get("question"),
                        "expected_doc_id": detail.get("expected_doc_id"),
                        "best_rank": best_rank,
                        "best_doc_rank": detail.get("best_doc_rank"),
                        "best_source_rank": detail.get("best_source_rank"),
                        "best_source_overlap": detail.get("best_source_overlap"),
                        "keyword_recall": keyword_recall,
                        "suggested_action": "inspect_top_results_and_split_or_merge",
                        "top_results": detail.get("top_results", []),
                    }
                )

    return {
        "artifact_dir": str(artifact_dir),
        "chunker": manifest.get("chunker"),
        "num_chunks": len(chunks),
        "max_token_budget": max_token_budget,
        "overlong_chunks": overlong,
        "near_boundary_chunks": near_boundary,
        "retrieval_findings": retrieval_findings,
        "summary": {
            "overlong_count": len(overlong),
            "near_boundary_count": len(near_boundary),
            "retrieval_issue_count": len(retrieval_findings),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a feedback repair report.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--retrieval-eval", type=Path, default=None)
    parser.add_argument("--max-token-budget", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    retrieval_eval_path = args.retrieval_eval
    if retrieval_eval_path is None:
        candidate = default_evaluation_dir(args.artifact_dir) / "retrieval_eval.json"
        retrieval_eval_path = candidate if candidate.exists() else None

    report = build_feedback_report(
        artifact_dir=args.artifact_dir,
        retrieval_eval_path=retrieval_eval_path,
        max_token_budget=args.max_token_budget,
    )
    out = args.out or default_evaluation_dir(args.artifact_dir) / "feedback_report.json"
    write_json(out, report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
