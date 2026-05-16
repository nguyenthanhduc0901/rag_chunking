from __future__ import annotations

import re


def normalize_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def split_paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split Gutenberg-style text on blank lines and keep original offsets."""
    paragraph_pattern = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
    spans: list[tuple[str, int, int]] = []
    for match in paragraph_pattern.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        paragraph = text[start:end]
        if not paragraph:
            continue
        spans.append((paragraph, start, end))
    return spans


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Lightweight sentence splitter with character offsets.

    This is intentionally simple because the project focuses on chunking methods.
    If your improved method needs richer parsing later, replace this one place.
    """
    paragraph_pattern = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
    sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+[)\"']*|$)")

    spans: list[tuple[str, int, int]] = []
    for paragraph in paragraph_pattern.finditer(text):
        paragraph_text = paragraph.group(0)
        for match in sentence_pattern.finditer(paragraph_text):
            raw_sentence = match.group(0)
            sentence = re.sub(r"\s+", " ", raw_sentence).strip()
            if len(sentence) < 2:
                continue
            start = paragraph.start() + match.start()
            end = paragraph.start() + match.end()
            spans.append((sentence, start, end))
    return spans


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))
