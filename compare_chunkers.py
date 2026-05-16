#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact chunker comparison.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.artifacts_dir.iterdir() if args.artifacts_dir.exists() else []):
        manifest_path = path / "manifest.json"
        eval_path = path / "chunk_eval.json"
        if not manifest_path.exists() or not eval_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "chunker": manifest["chunker"],
                "docs": manifest["num_documents"],
                "chunks": manifest["num_chunks"],
                "avg_chars": round(report["chars"]["mean"], 1),
                "median_tokens": round(report["estimated_tokens"]["median"], 1),
                "over_512": report["chunks_over_512_estimated_tokens"],
                "avg_sentences": round(report["sentences_per_chunk"]["mean"], 2),
                "boundary_median": report["boundary_similarity"]["median"],
            }
        )

    if not rows:
        print("No artifact reports found.")
        return

    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


if __name__ == "__main__":
    main()

