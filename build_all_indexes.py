#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline.build_index import build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes for multiple chunkers.")
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=[
            "adaptive_paragraph",
            "hierarchical_auto_merge",
            "feedback_optimized",
            "feedback_optimized_v2",
        ],
        help="Chunker names to build.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_clean"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--target-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--long-paragraph-words", type=int, default=None)
    parser.add_argument("--adaptive-percentile", type=float, default=None)
    parser.add_argument("--std-factor", type=float, default=None)
    parser.add_argument("--min-threshold", type=float, default=None)
    parser.add_argument("--max-threshold", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--parent-target-words", type=int, default=None)
    parser.add_argument("--parent-max-words", type=int, default=None)
    parser.add_argument("--parent-leaf-count", type=int, default=None)
    parser.add_argument("--repair-margin", type=float, default=None)
    parser.add_argument("--sentence-window-size", type=int, default=None)
    parser.add_argument("--sentence-window-overlap", type=int, default=None)
    parser.add_argument("--parent-child-count", type=int, default=None)
    parser.add_argument("--agent-window-words", type=int, default=None)
    parser.add_argument("--agent-max-units", type=int, default=None)
    parser.add_argument("--agent-max-input-chars", type=int, default=None)
    parser.add_argument("--agent-max-output-tokens", type=int, default=None)
    parser.add_argument("--agent-temperature", type=float, default=None)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument(
        "--disable-agent",
        action="store_true",
        help="Build agentic_gemini with deterministic fallback instead of Gemini calls.",
    )
    parser.add_argument("--fallback-threshold", type=float, default=None)
    args = parser.parse_args()

    for chunker in args.chunkers:
        print(f"\n=== Building {chunker} ===", flush=True)
        build_artifacts(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            chunker_name=chunker,
            model_name="BAAI/bge-base-en-v1.5",
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            threshold=args.threshold,
            min_words=args.min_words,
            target_words=args.target_words,
            max_words=args.max_words,
            long_paragraph_words=args.long_paragraph_words,
            adaptive_percentile=args.adaptive_percentile,
            std_factor=args.std_factor,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            window_size=args.window_size,
            parent_target_words=args.parent_target_words,
            parent_max_words=args.parent_max_words,
            parent_leaf_count=args.parent_leaf_count,
            repair_margin=args.repair_margin,
            sentence_window_size=args.sentence_window_size,
            sentence_window_overlap=args.sentence_window_overlap,
            parent_child_count=args.parent_child_count,
            agent_window_words=args.agent_window_words,
            agent_max_units=args.agent_max_units,
            agent_max_input_chars=args.agent_max_input_chars,
            agent_max_output_tokens=args.agent_max_output_tokens,
            agent_temperature=args.agent_temperature,
            agent_model=args.agent_model,
            use_agent=False if args.disable_agent else None,
            fallback_threshold=args.fallback_threshold,
            fixed_size=None,
            fixed_overlap=None,
        )


if __name__ == "__main__":
    main()
