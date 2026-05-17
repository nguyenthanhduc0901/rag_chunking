#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_reports(artifacts_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    eval_root = artifacts_dir / "evaluations"
    for artifact_dir in sorted(artifacts_dir.iterdir() if artifacts_dir.exists() else []):
        if artifact_dir.name in {"evaluations", "answer_quality", "answer_quality_smoke"}:
            continue
        path = eval_root / artifact_dir.name / "answer_quality_eval.json"
        if not path.exists():
            path = artifact_dir / "answer_quality_eval.json"
        if path.exists():
            reports.append(_load_json(path))
    return sorted(
        reports,
        key=lambda report: report.get("summary", {}).get("overall_avg", 0),
        reverse=True,
    )


def write_reports(reports: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "ranking": "overall_avg desc",
        "reports": [
            {
                "chunker": report.get("chunker"),
                "artifact_dir": report.get("artifact_dir"),
                "question_count": report.get("question_count"),
                "summary": report.get("summary"),
            }
            for report in reports
        ],
    }
    (out_dir / "answer_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    headers = [
        "rank",
        "chunker",
        "overall",
        "correct",
        "faithful",
        "complete",
        "context",
        "cite",
        "pass",
        "hallucination",
    ]
    lines = [
        "# Answer Quality Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rank, report in enumerate(reports, start=1):
        row = report.get("summary", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(report.get("chunker")),
                    f"{row.get('overall_avg', 0):.3f}",
                    f"{row.get('correctness_avg', 0):.3f}",
                    f"{row.get('faithfulness_avg', 0):.3f}",
                    f"{row.get('completeness_avg', 0):.3f}",
                    f"{row.get('context_usefulness_avg', 0):.3f}",
                    f"{row.get('citation_support_avg', 0):.3f}",
                    f"{row.get('pass_rate', 0) * 100:.1f}%",
                    f"{row.get('hallucination_rate', 0) * 100:.1f}%",
                ]
            )
            + " |"
        )
    (out_dir / "answer_quality_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemini-judged answer quality reports.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluations/answer_quality"))
    args = parser.parse_args()

    reports = collect_reports(args.artifacts_dir)
    if not reports:
        print("No answer_quality_eval.json reports found.")
        return
    write_reports(reports, args.out_dir)


if __name__ == "__main__":
    main()
