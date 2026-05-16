from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .load_data import load_documents
from .text_utils import normalize_inline_text, split_paragraphs_with_offsets, word_count


QUESTION_TYPES = ("fact", "concept", "summary", "why_how")
DIFFICULTIES = ("easy", "medium", "hard")
GENERATOR_VERSION = "source-paragraph-v1"


@dataclass(frozen=True)
class SourcePassage:
    doc_id: str
    source_file: str
    title: str | None
    author: str | None
    start_char: int
    end_char: int
    text: str
    position_ratio: float
    bucket: str


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_good_source_paragraph(text: str, min_words: int) -> bool:
    normalized = normalize_inline_text(text)
    if word_count(normalized) < min_words:
        return False
    lowered = normalized.lower()
    boilerplate_markers = (
        "project gutenberg",
        "gutenberg ebook",
        "start of this project",
        "end of this project",
        "produced by",
        "transcriber's note",
        "table of contents",
    )
    if any(marker in lowered for marker in boilerplate_markers):
        return False

    letters = [ch for ch in normalized if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    if upper_ratio > 0.72 and word_count(normalized) < 80:
        return False
    if len(normalized) < 220:
        return False
    return True


def _trim_to_word_window(
    text: str,
    absolute_start: int,
    max_words: int,
    rng: random.Random,
) -> tuple[str, int, int]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        clean_text = normalize_inline_text(text)
        leading = len(text) - len(text.lstrip())
        trailing = len(text) - len(text.rstrip())
        return clean_text, absolute_start + leading, absolute_start + len(text) - trailing

    window_start = rng.randint(0, len(words) - max_words)
    window_end = window_start + max_words - 1
    start = words[window_start].start()
    end = words[window_end].end()
    return normalize_inline_text(text[start:end]), absolute_start + start, absolute_start + end


def _bucket_for_position(position_ratio: float) -> str:
    if position_ratio < 0.33:
        return "early"
    if position_ratio < 0.66:
        return "middle"
    return "late"


def sample_source_passages(
    data_dir: Path,
    questions_per_doc: int,
    min_words: int,
    max_words: int,
    seed: int,
    limit_docs: int | None,
) -> list[SourcePassage]:
    rng = random.Random(seed)
    passages: list[SourcePassage] = []

    for document in load_documents(data_dir, limit=limit_docs):
        candidates: dict[str, list[tuple[str, int, int, float]]] = {
            "early": [],
            "middle": [],
            "late": [],
        }
        text_length = max(1, len(document.text))
        for paragraph, start, end in split_paragraphs_with_offsets(document.text):
            if not _is_good_source_paragraph(paragraph, min_words=min_words):
                continue
            position_ratio = start / text_length
            bucket = _bucket_for_position(position_ratio)
            candidates[bucket].append((paragraph, start, end, position_ratio))

        selected: list[tuple[str, int, int, float, str]] = []
        bucket_order = ["early", "middle", "late"]
        rng.shuffle(bucket_order)
        for bucket in bucket_order:
            if len(selected) >= questions_per_doc:
                break
            if candidates[bucket]:
                paragraph, start, end, position_ratio = rng.choice(candidates[bucket])
                selected.append((paragraph, start, end, position_ratio, bucket))

        remaining = [
            (paragraph, start, end, position_ratio, bucket)
            for bucket, rows in candidates.items()
            for paragraph, start, end, position_ratio in rows
            if (paragraph, start, end, position_ratio, bucket) not in selected
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, questions_per_doc - len(selected))])

        for paragraph, start, _end, position_ratio, bucket in selected[:questions_per_doc]:
            passage_text, passage_start, passage_end = _trim_to_word_window(
                paragraph,
                absolute_start=start,
                max_words=max_words,
                rng=rng,
            )
            passages.append(
                SourcePassage(
                    doc_id=document.doc_id,
                    source_file=document.source_file,
                    title=document.title,
                    author=document.author,
                    start_char=passage_start,
                    end_char=passage_end,
                    text=passage_text,
                    position_ratio=position_ratio,
                    bucket=bucket,
                )
            )

    rng.shuffle(passages)
    return passages


def _extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a JSON object: {content[:300]}")
    return json.loads(content[start : end + 1])


def _fallback_keywords(text: str, limit: int = 6) -> list[str]:
    stopwords = {
        "about",
        "after",
        "again",
        "against",
        "being",
        "could",
        "every",
        "from",
        "have",
        "into",
        "more",
        "other",
        "their",
        "there",
        "these",
        "this",
        "that",
        "they",
        "were",
        "which",
        "with",
        "would",
    }
    words = [
        word.lower().strip(".,;:!?()[]{}\"'")
        for word in text.split()
        if len(word.strip(".,;:!?()[]{}\"'")) >= 5
    ]
    keywords: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in stopwords or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _longest_common_word_run(a: str, b: str) -> int:
    a_words = re.findall(r"\w+", a.lower())
    b_words = re.findall(r"\w+", b.lower())
    if not a_words or not b_words:
        return 0
    longest = 0
    for i in range(len(a_words)):
        for j in range(len(b_words)):
            run = 0
            while (
                i + run < len(a_words)
                and j + run < len(b_words)
                and a_words[i + run] == b_words[j + run]
            ):
                run += 1
            longest = max(longest, run)
    return longest


def _normalize_generated_question(
    generated: dict[str, Any],
    passage: SourcePassage,
    question_type: str,
    difficulty: str,
    generator_status: str,
) -> dict[str, Any]:
    question = str(generated.get("question", "")).strip()
    if question and not question.endswith("?"):
        question += "?"

    answer_hint = str(generated.get("answer_hint", "")).strip()
    keywords = generated.get("expected_keywords", generated.get("keywords", []))
    if not isinstance(keywords, list):
        keywords = []
    clean_keywords = [
        normalize_inline_text(str(keyword)).lower()
        for keyword in keywords
        if normalize_inline_text(str(keyword))
    ][:6]
    if len(clean_keywords) < 3:
        clean_keywords = _fallback_keywords(passage.text)

    quality_warnings: list[str] = []
    if not question:
        quality_warnings.append("empty_question")
        joined = ", ".join(clean_keywords[:3]) or "the selected source"
        question = f"What does the source discuss about {joined}?"
    if len(question.split()) < 6:
        quality_warnings.append("question_too_short")
    if _longest_common_word_run(question, passage.text) >= 9:
        quality_warnings.append("question_copies_long_phrase")
    if not answer_hint:
        quality_warnings.append("empty_answer_hint")
        answer_hint = normalize_inline_text(passage.text[:280])

    generated_question_type = str(generated.get("question_type", question_type)).strip()
    if generated_question_type not in QUESTION_TYPES:
        generated_question_type = question_type
    generated_difficulty = str(generated.get("difficulty", difficulty)).strip()
    if generated_difficulty not in DIFFICULTIES:
        generated_difficulty = difficulty

    return {
        "question": question,
        "expected_keywords": clean_keywords,
        "answer_hint": answer_hint,
        "question_type": generated_question_type,
        "difficulty": generated_difficulty,
        "generator_status": generator_status,
        "quality_warnings": quality_warnings,
    }


def _ask_gemma(
    passage: SourcePassage,
    question_type: str,
    difficulty: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = f"""
Create one fair retrieval benchmark question from the source excerpt.

Rules:
- The question must be answerable from the excerpt.
- The question must be in English.
- Do not mention "excerpt", "passage", "paragraph", or "source text".
- Do not copy a phrase longer than 8 consecutive words from the excerpt.
- Prefer paraphrasing, especially for medium and hard questions.
- Return strict JSON only, no markdown.

Target question_type: {question_type}
Target difficulty: {difficulty}

Return schema:
{{
  "question": "one question",
  "answer_hint": "short answer grounded in the excerpt",
  "expected_keywords": ["3 to 6 lowercase keywords from the excerpt"],
  "question_type": "{question_type}",
  "difficulty": "{difficulty}"
}}

Book title: {passage.title or "Unknown"}
Author: {passage.author or "Unknown"}

Excerpt:
{passage.text[:2600]}
""".strip()
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": "Bearer local-gemma"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You create JSON-only benchmark questions for retrieval evaluation.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 360,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _extract_json_object(content)


def generate_benchmark_questions(
    data_dir: Path,
    out_path: Path,
    questions_per_doc: int,
    limit_questions: int | None,
    limit_docs: int | None,
    min_words: int,
    max_words: int,
    seed: int,
    gemma_url: str,
    gemma_model: str,
    timeout_seconds: float,
    no_llm: bool,
    fail_on_llm_error: bool,
) -> list[dict[str, Any]]:
    passages = sample_source_passages(
        data_dir=data_dir,
        questions_per_doc=questions_per_doc,
        min_words=min_words,
        max_words=max_words,
        seed=seed,
        limit_docs=limit_docs,
    )
    if limit_questions is not None:
        passages = passages[:limit_questions]

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    for index, passage in enumerate(passages, start=1):
        question_type = QUESTION_TYPES[(index - 1) % len(QUESTION_TYPES)]
        difficulty = DIFFICULTIES[(index - 1 + rng.randint(0, 2)) % len(DIFFICULTIES)]

        generated: dict[str, Any]
        generator_status = "llm"
        if no_llm:
            generator_status = "fallback_no_llm"
            keywords = _fallback_keywords(passage.text)
            generated = {
                "question": (
                    f"What does {passage.title or 'the selected book'} discuss "
                    f"about {', '.join(keywords[:3])}?"
                ),
                "answer_hint": normalize_inline_text(passage.text[:260]),
                "expected_keywords": keywords,
                "question_type": question_type,
                "difficulty": difficulty,
            }
        else:
            try:
                generated = _ask_gemma(
                    passage=passage,
                    question_type=question_type,
                    difficulty=difficulty,
                    base_url=gemma_url,
                    model=gemma_model,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                if fail_on_llm_error:
                    raise
                generator_status = f"fallback_llm_error:{type(exc).__name__}"
                keywords = _fallback_keywords(passage.text)
                generated = {
                    "question": (
                        f"What does {passage.title or 'the selected book'} discuss "
                        f"about {', '.join(keywords[:3])}?"
                    ),
                    "answer_hint": normalize_inline_text(passage.text[:260]),
                    "expected_keywords": keywords,
                    "question_type": question_type,
                    "difficulty": difficulty,
                }

        question_fields = _normalize_generated_question(
            generated=generated,
            passage=passage,
            question_type=question_type,
            difficulty=difficulty,
            generator_status=generator_status,
        )
        row = {
            "question_id": f"q{index:05d}",
            "generator_version": GENERATOR_VERSION,
            **question_fields,
            "expected_doc_id": passage.doc_id,
            "source_file": passage.source_file,
            "source_title": passage.title,
            "source_author": passage.author,
            "source_start_char": passage.start_char,
            "source_end_char": passage.end_char,
            "source_position": passage.bucket,
            "source_position_ratio": round(passage.position_ratio, 4),
            "source_word_count": word_count(passage.text),
            "source_excerpt": passage.text,
        }
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{len(passages)}] doc={passage.doc_id} "
            f"type={row['question_type']} difficulty={row['difficulty']} "
            f"status={row['generator_status']} -> {row['question']}",
            flush=True,
        )

    _jsonl_write(out_path, rows)
    partial_path.unlink(missing_ok=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a source-grounded retrieval benchmark from data_clean."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_clean"))
    parser.add_argument("--out", type=Path, default=Path("eval/source_questions.jsonl"))
    parser.add_argument("--questions-per-doc", type=int, default=2)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=90)
    parser.add_argument("--max-words", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemma-url", default="http://localhost:8000")
    parser.add_argument("--gemma-model", default="gemma-4-e4b-it")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fail-on-llm-error", action="store_true")
    args = parser.parse_args()

    rows = generate_benchmark_questions(
        data_dir=args.data_dir,
        out_path=args.out,
        questions_per_doc=args.questions_per_doc,
        limit_questions=args.limit_questions,
        limit_docs=args.limit_docs,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
        gemma_url=args.gemma_url,
        gemma_model=args.gemma_model,
        timeout_seconds=args.timeout_seconds,
        no_llm=args.no_llm,
        fail_on_llm_error=args.fail_on_llm_error,
    )
    print(f"Wrote {len(rows)} benchmark questions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
