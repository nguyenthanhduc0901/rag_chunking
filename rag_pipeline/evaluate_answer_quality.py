from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from google.genai import types

from .io_utils import write_json
from .llm import build_prompt, make_gemini_client
from .retrieve import Retriever
from .schema import SearchResult
from .text_utils import normalize_inline_text


DEFAULT_CHUNKERS = (
    "parent_child_chunk",
    "agentic_gemini",
    "feedback_optimized_v2",
    "fixed_sentence",
    "adaptive_paragraph",
    "hierarchical_auto_merge",
    "feedback_optimized",
    "paragraph_semantic",
)


def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _quality_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    warnings = row.get("quality_warnings") or []
    warning_penalty = len(warnings) if isinstance(warnings, list) else 1
    status = str(row.get("generator_status", ""))
    status_score = 0 if status == "gemini" else 1
    span_count = int(row.get("source_span_count") or 1)
    source_words = int(row.get("source_word_count") or 0)
    return warning_penalty, status_score, -span_count, f"{source_words:06d}"


def select_questions(
    questions: list[dict[str, Any]],
    limit: int,
    scopes: list[str],
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(questions):
        return questions

    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        scope = str(row.get("question_scope") or "local")
        if not scopes or scope in scopes:
            by_scope[scope].append(row)

    for rows in by_scope.values():
        rows.sort(key=_quality_score)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    scope_order = scopes or sorted(by_scope)
    while len(selected) < limit and scope_order:
        made_progress = False
        for scope in scope_order:
            while by_scope[scope] and by_scope[scope][0].get("question_id") in used_ids:
                by_scope[scope].pop(0)
            if not by_scope[scope]:
                continue
            row = by_scope[scope].pop(0)
            selected.append(row)
            used_ids.add(str(row.get("question_id")))
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return selected


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Expected JSON object, got: {text[:500]}")
    return json.loads(text[start : end + 1])


def _call_gemini(
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float = 0.0,
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip() if isinstance(response.text, str) else str(response).strip()


def _result_context(results: list[SearchResult], max_chars: int = 7000) -> str:
    blocks: list[str] = []
    total = 0
    for result in results:
        chunk = result.chunk
        block = (
            f"[{result.rank}] score={result.score:.3f} doc_id={chunk.doc_id} "
            f"title={chunk.title or 'Untitled'}\n{chunk.text}"
        )
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining <= 200:
                break
            block = block[:remaining] + "\n[retrieved context truncated]"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _source_material(row: dict[str, Any], max_chars: int = 5000) -> str:
    material = str(row.get("source_excerpt") or "")
    material = normalize_inline_text(material)
    if len(material) > max_chars:
        return material[:max_chars] + " [source material truncated]"
    return material


def _answer_prompt(question: str, results: list[SearchResult]) -> str:
    return (
        "You are a careful RAG assistant for Project Gutenberg books. "
        "Answer only from the provided context. If the context is insufficient, "
        "say that the retrieved context is not enough. Keep the answer concise "
        "but complete, and cite bracketed context numbers when useful.\n\n"
        f"{build_prompt(question, results)}"
    )


def _judge_prompt(
    row: dict[str, Any],
    chunker: str,
    answer: str,
    results: list[SearchResult],
) -> str:
    expected_keywords = row.get("expected_keywords") or []
    return f"""
You are an impartial evaluator for a RAG chunking benchmark.

Judge the answer using the gold source material and the retrieved context.
Return strict JSON only. Do not include markdown.

Scoring scale:
- 1 = very poor
- 2 = weak
- 3 = partially correct
- 4 = good
- 5 = excellent

Definitions:
- correctness: answer matches the gold source material and answer hint.
- faithfulness: answer is supported by the retrieved context, without unsupported claims.
- completeness: answer covers the important points needed for the question scope.
- context_usefulness: retrieved context contains enough evidence to answer.
- citation_support: citations or references to retrieved context are useful; score 3 if no citations are needed but the answer is still grounded.

Return schema:
{{
  "correctness": 1,
  "faithfulness": 1,
  "completeness": 1,
  "context_usefulness": 1,
  "citation_support": 1,
  "hallucination": false,
  "verdict": "pass_or_fail",
  "reason": "short reason"
}}

Chunker: {chunker}
Question id: {row.get("question_id")}
Question scope: {row.get("question_scope")}
Question type: {row.get("question_type")}
Question:
{row.get("question")}

Gold answer hint:
{row.get("answer_hint")}

Expected keywords:
{", ".join(str(keyword) for keyword in expected_keywords)}

Gold source material:
{_source_material(row)}

Retrieved context:
{_result_context(results)}

RAG answer:
{answer}
""".strip()


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def _summarize(rows: list[dict[str, Any]], include_by_scope: bool = True) -> dict[str, Any]:
    if not rows:
        return {}
    metrics = (
        "correctness",
        "faithfulness",
        "completeness",
        "context_usefulness",
        "citation_support",
    )
    summary: dict[str, Any] = {
        "samples": len(rows),
        "pass_rate": sum(1 for row in rows if row.get("verdict") == "pass") / len(rows),
        "hallucination_rate": sum(1 for row in rows if row.get("hallucination")) / len(rows),
    }
    for metric in metrics:
        summary[f"{metric}_avg"] = sum(_coerce_score(row.get(metric)) for row in rows) / len(rows)
    summary["overall_avg"] = sum(summary[f"{metric}_avg"] for metric in metrics) / len(metrics)

    if include_by_scope:
        by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scope[str(row.get("question_scope") or "unknown")].append(row)
        summary["by_scope"] = {
            scope: _summarize(scope_rows, include_by_scope=False)
            for scope, scope_rows in sorted(by_scope.items())
        }
    return summary


def evaluate_chunker(
    artifact_dir: Path,
    questions: list[dict[str, Any]],
    client: Any,
    model: str,
    top_k: int,
    device: str,
    batch_size: int,
    answer_max_tokens: int,
    judge_max_tokens: int,
) -> dict[str, Any]:
    retriever = Retriever(artifact_dir, device=device, batch_size=batch_size)
    chunker = str(retriever.manifest.get("chunker", artifact_dir.name))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, row in enumerate(questions, start=1):
        question = str(row.get("question", ""))
        results = retriever.search(question, top_k=top_k)
        answer = _call_gemini(
            client=client,
            model=model,
            prompt=_answer_prompt(question, results),
            max_output_tokens=answer_max_tokens,
            temperature=0.0,
        )
        judge_raw = _call_gemini(
            client=client,
            model=model,
            prompt=_judge_prompt(row, chunker, answer, results),
            max_output_tokens=judge_max_tokens,
            temperature=0.0,
        )
        try:
            judge = _json_object(judge_raw)
        except Exception as exc:
            judge = {
                "correctness": 0,
                "faithfulness": 0,
                "completeness": 0,
                "context_usefulness": 0,
                "citation_support": 0,
                "hallucination": True,
                "verdict": "fail",
                "reason": f"judge_parse_error:{type(exc).__name__}: {judge_raw[:300]}",
            }
        verdict = str(judge.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "fail"}:
            average_score = (
                _coerce_score(judge.get("correctness"))
                + _coerce_score(judge.get("faithfulness"))
                + _coerce_score(judge.get("completeness"))
                + _coerce_score(judge.get("context_usefulness"))
            ) / 4
            verdict = "pass" if average_score >= 3.5 and not bool(judge.get("hallucination")) else "fail"
        judge["verdict"] = verdict
        judge["hallucination"] = bool(judge.get("hallucination"))

        result_row = {
            "question_id": row.get("question_id"),
            "question_scope": row.get("question_scope"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "question": question,
            "answer": answer,
            "retrieved_doc_ids": [result.chunk.doc_id for result in results],
            "retrieved_chunk_ids": [result.chunk.chunk_id for result in results],
            **judge,
        }
        rows.append(result_row)
        print(
            f"[{chunker}] {index}/{len(questions)} {row.get('question_id')} "
            f"scope={row.get('question_scope')} correctness={result_row.get('correctness')} "
            f"faithfulness={result_row.get('faithfulness')} verdict={result_row.get('verdict')}",
            flush=True,
        )

    return {
        "chunker": chunker,
        "artifact_dir": str(artifact_dir),
        "model": model,
        "top_k": top_k,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "question_count": len(questions),
        "question_scope_counts": dict(Counter(str(row.get("question_scope")) for row in questions)),
        "summary": _summarize(rows),
        "rows": rows,
    }


def write_markdown_summary(reports: list[dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(
        reports,
        key=lambda report: report.get("summary", {}).get("overall_avg", 0),
        reverse=True,
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
    for rank, report in enumerate(ordered, start=1):
        summary = report.get("summary", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(report.get("chunker")),
                    f"{summary.get('overall_avg', 0):.3f}",
                    f"{summary.get('correctness_avg', 0):.3f}",
                    f"{summary.get('faithfulness_avg', 0):.3f}",
                    f"{summary.get('completeness_avg', 0):.3f}",
                    f"{summary.get('context_usefulness_avg', 0):.3f}",
                    f"{summary.get('citation_support_avg', 0):.3f}",
                    f"{summary.get('pass_rate', 0) * 100:.1f}%",
                    f"{summary.get('hallucination_rate', 0) * 100:.1f}%",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality with Gemini judge.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-dir", type=Path, action="append", default=None)
    parser.add_argument("--questions", type=Path, default=Path("eval/source_questions.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluations/answer_quality"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--scopes", default="local,multi_paragraph,book_theme")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=384)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    args = parser.parse_args()

    model = args.model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    questions = select_questions(_load_questions(args.questions), limit=args.limit, scopes=scopes)
    artifact_dirs = args.artifact_dir or [
        args.artifacts_dir / name
        for name in DEFAULT_CHUNKERS
        if (args.artifacts_dir / name / "manifest.json").exists()
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.out_dir / "selected_questions.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in questions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports: list[dict[str, Any]] = []
    client = make_gemini_client()
    try:
        for artifact_dir in artifact_dirs:
            report = evaluate_chunker(
                artifact_dir=Path(artifact_dir),
                questions=questions,
                client=client,
                model=model,
                top_k=args.top_k,
                device=args.device,
                batch_size=args.batch_size,
                answer_max_tokens=args.answer_max_tokens,
                judge_max_tokens=args.judge_max_tokens,
            )
            reports.append(report)
            eval_dir = args.artifacts_dir / "evaluations" / Path(artifact_dir).name
            write_json(eval_dir / "answer_quality_eval.json", report)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        if hasattr(client, "close"):
            client.close()

    combined = {
        "model": model,
        "questions": str(args.questions),
        "selected_questions": str(selected_path),
        "limit": args.limit,
        "top_k": args.top_k,
        "reports": [
            {
                "chunker": report.get("chunker"),
                "artifact_dir": report.get("artifact_dir"),
                "summary": report.get("summary"),
            }
            for report in reports
        ],
    }
    write_json(args.out_dir / "answer_quality_summary.json", combined)
    write_markdown_summary(reports, args.out_dir / "answer_quality_summary.md")
    print(f"Wrote {args.out_dir / 'answer_quality_summary.json'}", flush=True)
    print(f"Wrote {args.out_dir / 'answer_quality_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
