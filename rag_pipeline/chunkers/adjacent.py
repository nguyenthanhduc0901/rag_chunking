from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import replace
from typing import Any

import numpy as np

from rag_pipeline.embeddings import encode_texts
from rag_pipeline.schema import Chunk, Document
from rag_pipeline.text_utils import split_sentences_with_offsets, word_count

from .base import BaseChunker


class ParagraphBlockSemanticMixin:
    def _split_long_paragraph(
        self,
        paragraph_text: str,
        start_char: int,
        end_char: int,
    ) -> list[tuple[str, int, int]]:
        if word_count(paragraph_text) <= self.long_paragraph_words:
            return [(paragraph_text, start_char, end_char)]

        local_sentences = split_sentences_with_offsets(paragraph_text)
        if len(local_sentences) <= 1:
            return self._split_long_paragraph_by_words(paragraph_text, start_char, end_char)

        sentence_pieces: list[tuple[str, int, int]] = []
        for sentence, local_start, local_end in local_sentences:
            if word_count(sentence) <= self.max_words:
                sentence_pieces.append((sentence, local_start, local_end))
                continue
            sentence_text = paragraph_text[local_start:local_end]
            for piece_text, piece_start, piece_end in self._split_text_by_words(
                sentence_text,
                local_start,
            ):
                sentence_pieces.append((piece_text, piece_start, piece_end))

        blocks: list[tuple[str, int, int]] = []
        block_start = 0
        block_words = 0
        for i, (sentence, _, _) in enumerate(sentence_pieces):
            sentence_words = word_count(sentence)
            if block_words and block_words + sentence_words > self.target_words:
                start = start_char + sentence_pieces[block_start][1]
                end = start_char + sentence_pieces[i - 1][2]
                text = paragraph_text[sentence_pieces[block_start][1] : sentence_pieces[i - 1][2]]
                blocks.append((" ".join(text.split()), start, end))
                block_start = i
                block_words = 0
            block_words += sentence_words

        if block_start < len(sentence_pieces):
            start = start_char + sentence_pieces[block_start][1]
            end = start_char + sentence_pieces[-1][2]
            text = paragraph_text[sentence_pieces[block_start][1] : sentence_pieces[-1][2]]
            blocks.append((" ".join(text.split()), start, end))
        return blocks

    def _split_long_paragraph_by_words(
        self,
        paragraph_text: str,
        start_char: int,
        end_char: int,
    ) -> list[tuple[str, int, int]]:
        return [
            (text, start_char + start, start_char + end)
            for text, start, end in self._split_text_by_words(paragraph_text, 0)
        ]

    def _split_text_by_words(
        self,
        text: str,
        offset: int,
    ) -> list[tuple[str, int, int]]:
        words = list(re.finditer(r"\S+", text))
        blocks: list[tuple[str, int, int]] = []
        for start in range(0, len(words), self.target_words):
            end = min(start + self.target_words, len(words)) - 1
            block_start = offset + words[start].start()
            block_end = offset + words[end].end()
            block_text = text[words[start].start() : words[end].end()]
            blocks.append((" ".join(block_text.split()), block_start, block_end))
        return blocks

    def _paragraph_units(self, document: Document) -> list[tuple[str, int, int]]:
        units: list[tuple[str, int, int]] = []
        for paragraph_text, start_char, end_char in self.paragraph_spans(document):
            if word_count(paragraph_text) <= 2:
                continue
            units.extend(
                self._split_long_paragraph(paragraph_text, start_char, end_char)
            )
        return units

    def _adjacent_scores(self, embeddings: np.ndarray) -> list[float]:
        return [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

    def _window_scores(
        self,
        embeddings: np.ndarray,
        window_size: int,
    ) -> list[float]:
        if len(embeddings) <= 1:
            return []
        scores: list[float] = []
        for boundary_index in range(len(embeddings) - 1):
            left_start = max(0, boundary_index - window_size + 1)
            left = embeddings[left_start : boundary_index + 1].mean(axis=0)
            right_end = min(len(embeddings), boundary_index + 1 + window_size)
            right = embeddings[boundary_index + 1 : right_end].mean(axis=0)
            left = left / max(float(np.linalg.norm(left)), 1e-12)
            right = right / max(float(np.linalg.norm(right)), 1e-12)
            scores.append(float(np.dot(left, right)))
        return scores

    def _make_single_unit_chunk(
        self,
        document: Document,
        units: list[tuple[str, int, int]],
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        metadata = {
            "unit_type": "paragraph_block",
            "estimated_words": word_count(units[0][0]),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return [
            self.make_chunk(
                document,
                0,
                units,
                0,
                0,
                metadata,
            )
        ]

    def _metadata_prefix(self, document: Document) -> str:
        fields = []
        if document.title:
            fields.append(f"Book: {document.title}")
        if document.author:
            fields.append(f"Author: {document.author}")
        if document.language:
            fields.append(f"Language: {document.language}")
        return "\n".join(f"[{field}]" for field in fields)

    def _build_chunks_from_scores(
        self,
        document: Document,
        units: list[tuple[str, int, int]],
        unit_texts: list[str],
        boundary_scores: list[float],
        threshold: float,
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        current_words = word_count(unit_texts[0])
        chunk_boundary_scores: list[float] = []
        extra_metadata = extra_metadata or {}

        for i, score in enumerate(boundary_scores):
            chunk_boundary_scores.append(score)
            can_break = current_words >= self.min_words
            reached_target = current_words >= self.target_words
            too_long = current_words >= self.max_words
            semantic_break = reached_target and score < threshold
            should_break = can_break and (semantic_break or too_long)
            if should_break:
                metadata = {
                    "unit_type": "paragraph_block",
                    "boundary_scores": chunk_boundary_scores,
                    "estimated_words": current_words,
                    "boundary_threshold": threshold,
                }
                metadata.update(extra_metadata)
                chunks.append(
                    self.make_chunk(document, len(chunks), units, start, i, metadata)
                )
                start = i + 1
                current_words = word_count(unit_texts[start]) if start < len(unit_texts) else 0
                chunk_boundary_scores = []
            else:
                current_words += word_count(unit_texts[i + 1])

        if start < len(units):
            final_words = sum(word_count(text) for text in unit_texts[start:])
            metadata = {
                "unit_type": "paragraph_block",
                "boundary_scores": chunk_boundary_scores,
                "estimated_words": final_words,
                "boundary_threshold": threshold,
            }
            metadata.update(extra_metadata)
            chunks.append(
                self.make_chunk(
                    document,
                    len(chunks),
                    units,
                    start,
                    len(units) - 1,
                    metadata,
                )
            )
        return chunks


class AdjacentSimilarityChunker(ParagraphBlockSemanticMixin, BaseChunker):
    """Basic semantic chunking over paragraph blocks.

    The basic idea compares adjacent units with cosine similarity.
    For Gutenberg books, sentence-level units are too small and expensive, so
    this version uses paragraphs as the primary unit and only splits very long
    paragraphs into sentence-packed blocks.
    """

    name = "paragraph_semantic"

    def __init__(
        self,
        threshold: float = 0.54,
        min_words: int = 120,
        target_words: int = 150,
        max_words: int = 220,
        long_paragraph_words: int = 220,
    ) -> None:
        super().__init__(
            threshold=threshold,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
        )
        self.threshold = threshold
        self.min_words = min_words
        self.target_words = target_words
        self.max_words = max_words
        self.long_paragraph_words = long_paragraph_words

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        if model is None:
            raise ValueError(f"{self.name} requires an embedding model")
        units = self._paragraph_units(document)
        if not units:
            return []
        if len(units) == 1:
            return self._make_single_unit_chunk(
                document,
                units,
                {
                    "threshold_mode": "adaptive",
                    "embedding_prefix": self._metadata_prefix(document),
                    "metadata_prefix_enabled": True,
                },
            )

        unit_texts = [text for text, _, _ in units]
        embeddings = encode_texts(model, unit_texts, batch_size=batch_size)
        adjacent_scores = self._adjacent_scores(embeddings)
        return self._build_chunks_from_scores(
            document=document,
            units=units,
            unit_texts=unit_texts,
            boundary_scores=adjacent_scores,
            threshold=self.threshold,
            extra_metadata={"threshold_mode": "global"},
        )


class AdaptiveParagraphChunker(AdjacentSimilarityChunker):
    """Paragraph semantic chunking with a per-document adaptive boundary threshold."""

    name = "adaptive_paragraph"

    def __init__(
        self,
        threshold: float = 0.54,
        min_words: int = 120,
        target_words: int = 150,
        max_words: int = 220,
        long_paragraph_words: int = 220,
        adaptive_percentile: float = 20.0,
        std_factor: float = 0.75,
        min_threshold: float = 0.35,
        max_threshold: float = 0.72,
        window_size: int = 2,
    ) -> None:
        super().__init__(
            threshold=threshold,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
        )
        self.adaptive_percentile = adaptive_percentile
        self.std_factor = std_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_size = window_size
        self.params.update(
            {
                "adaptive_percentile": adaptive_percentile,
                "std_factor": std_factor,
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
                "window_size": window_size,
            }
        )

    def _adaptive_threshold(self, scores: list[float]) -> float:
        if len(scores) < 4:
            return self.threshold
        percentile_score = float(np.percentile(np.asarray(scores), self.adaptive_percentile))
        median_score = statistics.median(scores)
        std_score = statistics.pstdev(scores)
        std_score_cut = median_score - self.std_factor * std_score
        adaptive = min(self.threshold, percentile_score, std_score_cut)
        return max(self.min_threshold, min(self.max_threshold, adaptive))

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        if model is None:
            raise ValueError(f"{self.name} requires an embedding model")
        units = self._paragraph_units(document)
        if not units:
            return []
        if len(units) == 1:
            return self._make_single_unit_chunk(document, units)

        unit_texts = [text for text, _, _ in units]
        embeddings = encode_texts(model, unit_texts, batch_size=batch_size)
        boundary_scores = self._window_scores(embeddings, self.window_size)
        threshold = self._adaptive_threshold(boundary_scores)
        return self._build_chunks_from_scores(
            document=document,
            units=units,
            unit_texts=unit_texts,
            boundary_scores=boundary_scores,
            threshold=threshold,
            extra_metadata={
                "threshold_mode": "adaptive",
                "adaptive_percentile": self.adaptive_percentile,
                "window_size": self.window_size,
                "embedding_prefix": self._metadata_prefix(document),
                "metadata_prefix_enabled": True,
            },
        )


class HierarchicalAutoMergeChunker(AdaptiveParagraphChunker):
    """Adaptive leaf chunks with parent context metadata for retrieval auto-merge."""

    name = "hierarchical_auto_merge"

    def __init__(
        self,
        threshold: float = 0.54,
        min_words: int = 120,
        target_words: int = 150,
        max_words: int = 220,
        long_paragraph_words: int = 220,
        adaptive_percentile: float = 20.0,
        std_factor: float = 0.75,
        min_threshold: float = 0.35,
        max_threshold: float = 0.72,
        window_size: int = 2,
        parent_target_words: int = 700,
        parent_max_words: int = 950,
        parent_leaf_count: int = 4,
    ) -> None:
        super().__init__(
            threshold=threshold,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
            adaptive_percentile=adaptive_percentile,
            std_factor=std_factor,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            window_size=window_size,
        )
        self.parent_target_words = parent_target_words
        self.parent_max_words = parent_max_words
        self.parent_leaf_count = parent_leaf_count
        self.params.update(
            {
                "parent_target_words": parent_target_words,
                "parent_max_words": parent_max_words,
                "parent_leaf_count": parent_leaf_count,
            }
        )

    def _with_parent_metadata(
        self,
        chunks: list[Chunk],
        document: Document,
    ) -> list[Chunk]:
        enriched: list[Chunk] = []
        group: list[Chunk] = []
        group_words = 0
        parent_index = 0

        def flush_group() -> None:
            nonlocal group, group_words, parent_index
            if not group:
                return
            parent_id = f"{self.name}:{document.doc_id}:parent:{parent_index:06d}"
            parent_text = "\n\n".join(chunk.text for chunk in group)
            parent_start = min(chunk.start_char for chunk in group)
            parent_end = max(chunk.end_char for chunk in group)
            for child_position, chunk in enumerate(group):
                metadata = dict(chunk.metadata)
                metadata.update(
                    {
                        "hierarchy_level": "leaf",
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "parent_child_count": len(group),
                        "parent_child_position": child_position,
                        "parent_start_char": parent_start,
                        "parent_end_char": parent_end,
                        "parent_estimated_words": group_words,
                        "parent_text": parent_text,
                    }
                )
                enriched.append(replace(chunk, metadata=metadata))
            parent_index += 1
            group = []
            group_words = 0

        for chunk in chunks:
            chunk_words = word_count(chunk.text)
            if group and (
                len(group) >= self.parent_leaf_count
                or group_words + chunk_words > self.parent_max_words
            ):
                flush_group()
            group.append(chunk)
            group_words += chunk_words
            if group_words >= self.parent_target_words:
                flush_group()
        flush_group()
        return enriched

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        chunks = super().chunk(document, model=model, batch_size=batch_size)
        return self._with_parent_metadata(chunks, document)


class FeedbackOptimizedChunker(AdaptiveParagraphChunker):
    """Retrieval-feedback-ready chunker with conservative self-repair heuristics.

    The offline retrieval-feedback loop can later split/merge using eval results.
    This chunker adds the first practical repair policy during build time:
    smaller chunks around likely low-precision boundaries and explicit metadata for
    future feedback iterations.
    """

    name = "feedback_optimized"

    def __init__(
        self,
        threshold: float = 0.54,
        min_words: int = 90,
        target_words: int = 130,
        max_words: int = 190,
        long_paragraph_words: int = 190,
        adaptive_percentile: float = 25.0,
        std_factor: float = 0.65,
        min_threshold: float = 0.35,
        max_threshold: float = 0.72,
        window_size: int = 2,
        repair_margin: float = 0.04,
    ) -> None:
        super().__init__(
            threshold=threshold,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
            adaptive_percentile=adaptive_percentile,
            std_factor=std_factor,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            window_size=window_size,
        )
        self.repair_margin = repair_margin
        self.params.update({"repair_margin": repair_margin})

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        chunks = super().chunk(document, model=model, batch_size=batch_size)
        repaired: list[Chunk] = []
        for chunk in chunks:
            scores = [
                float(score)
                for score in chunk.metadata.get("boundary_scores", [])
                if isinstance(score, (int, float))
            ]
            threshold = float(chunk.metadata.get("boundary_threshold", self.threshold))
            near_boundary = any(score < threshold + self.repair_margin for score in scores)
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "feedback_ready": True,
                    "repair_policy": "adaptive_small_leaf",
                    "near_boundary": near_boundary,
                    "suggested_action": (
                        "review_split_or_merge_after_retrieval_eval"
                        if near_boundary
                        else "keep_unless_eval_fails"
                    ),
                }
            )
            repaired.append(replace(chunk, metadata=metadata))
        return repaired


class FeedbackOptimizedV2Chunker(FeedbackOptimizedChunker):
    """Second feedback iteration tuned from source-grounded retrieval misses.

    The benchmark showed that paragraph-level semantic chunks are efficient but
    miss exact source spans more often than fixed sentence windows. V2 therefore
    turns the feedback into a retrieval-first policy: compact overlapping
    sentence windows, enriched with document metadata in the embedding text.
    This keeps answer contexts small for the chatbot while increasing the
    number of retrievable source-local entry points.
    """

    name = "feedback_optimized_v2"

    def __init__(
        self,
        threshold: float = 0.54,
        min_words: int = 85,
        target_words: int = 120,
        max_words: int = 175,
        long_paragraph_words: int = 175,
        adaptive_percentile: float = 25.0,
        std_factor: float = 0.65,
        min_threshold: float = 0.35,
        max_threshold: float = 0.72,
        window_size: int = 2,
        repair_margin: float = 0.05,
        sentence_window_size: int = 8,
        sentence_window_overlap: int = 2,
    ) -> None:
        super().__init__(
            threshold=threshold,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
            adaptive_percentile=adaptive_percentile,
            std_factor=std_factor,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            window_size=window_size,
            repair_margin=repair_margin,
        )
        if sentence_window_size < 1:
            raise ValueError("sentence_window_size must be >= 1")
        if sentence_window_overlap < 0 or sentence_window_overlap >= sentence_window_size:
            raise ValueError(
                "sentence_window_overlap must be >= 0 and < sentence_window_size"
            )
        self.sentence_window_size = sentence_window_size
        self.sentence_window_overlap = sentence_window_overlap
        self.params.update(
            {
                "sentence_window_size": sentence_window_size,
                "sentence_window_overlap": sentence_window_overlap,
            }
        )

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        spans = self.sentence_spans(document)
        if not spans:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(spans):
            end = min(start + self.sentence_window_size, len(spans)) - 1
            window_text = document.text[spans[start][1] : spans[end][2]]
            metadata = {
                "unit_type": "sentence_window",
                "feedback_ready": True,
                "feedback_iteration": 2,
                "repair_policy": "feedback_v2_source_local_window",
                "v2_goal": "maximize_source_span_retrieval_with_compact_context",
                "estimated_words": word_count(window_text),
                "embedding_prefix": self._metadata_prefix(document),
                "metadata_prefix_enabled": True,
                "semantic_feedback_from": "feedback_optimized_and_fixed_sentence_eval",
            }
            chunks.append(
                self.make_chunk(
                    document,
                    len(chunks),
                    spans,
                    start,
                    end,
                    metadata,
                )
            )
            if end == len(spans) - 1:
                break
            start = end + 1 - self.sentence_window_overlap
        return chunks


class ParentChildChunker(ParagraphBlockSemanticMixin, BaseChunker):
    """Small retrievable child chunks with larger parent context for generation."""

    name = "parent_child_chunk"

    def __init__(
        self,
        sentence_window_size: int = 8,
        sentence_window_overlap: int = 2,
        parent_target_words: int = 700,
        parent_max_words: int = 950,
        parent_child_count: int = 5,
    ) -> None:
        super().__init__(
            sentence_window_size=sentence_window_size,
            sentence_window_overlap=sentence_window_overlap,
            parent_target_words=parent_target_words,
            parent_max_words=parent_max_words,
            parent_child_count=parent_child_count,
        )
        if sentence_window_size < 1:
            raise ValueError("sentence_window_size must be >= 1")
        if sentence_window_overlap < 0 or sentence_window_overlap >= sentence_window_size:
            raise ValueError(
                "sentence_window_overlap must be >= 0 and < sentence_window_size"
            )
        if parent_target_words < 1 or parent_max_words < parent_target_words:
            raise ValueError("parent_max_words must be >= parent_target_words >= 1")
        if parent_child_count < 1:
            raise ValueError("parent_child_count must be >= 1")

        self.sentence_window_size = sentence_window_size
        self.sentence_window_overlap = sentence_window_overlap
        self.parent_target_words = parent_target_words
        self.parent_max_words = parent_max_words
        self.parent_child_count = parent_child_count

    def _child_chunks(self, document: Document) -> list[Chunk]:
        spans = self.sentence_spans(document)
        if not spans:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(spans):
            end = min(start + self.sentence_window_size, len(spans)) - 1
            text = document.text[spans[start][1] : spans[end][2]]
            metadata = {
                "unit_type": "sentence_window_child",
                "hierarchy_level": "child",
                "estimated_words": word_count(text),
                "embedding_prefix": self._metadata_prefix(document),
                "metadata_prefix_enabled": True,
                "parent_context_policy": "always",
            }
            chunks.append(
                self.make_chunk(
                    document,
                    len(chunks),
                    spans,
                    start,
                    end,
                    metadata,
                )
            )
            if end == len(spans) - 1:
                break
            start = end + 1 - self.sentence_window_overlap
        return chunks

    def _attach_parent_metadata(
        self,
        chunks: list[Chunk],
        document: Document,
    ) -> list[Chunk]:
        enriched: list[Chunk] = []
        group: list[Chunk] = []
        group_words = 0
        parent_index = 0

        def flush_group() -> None:
            nonlocal group, group_words, parent_index
            if not group:
                return
            parent_id = f"{self.name}:{document.doc_id}:parent:{parent_index:06d}"
            parent_start = min(chunk.start_char for chunk in group)
            parent_end = max(chunk.end_char for chunk in group)
            parent_text = document.text[parent_start:parent_end].strip()
            parent_words = word_count(parent_text)
            for child_position, chunk in enumerate(group):
                metadata = dict(chunk.metadata)
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "parent_child_count": len(group),
                        "parent_child_position": child_position,
                        "parent_start_char": parent_start,
                        "parent_end_char": parent_end,
                        "parent_estimated_words": parent_words,
                        "parent_text": parent_text,
                        "parent_strategy": "sentence_child_contiguous_parent",
                    }
                )
                enriched.append(replace(chunk, metadata=metadata))
            parent_index += 1
            group = []
            group_words = 0

        for chunk in chunks:
            chunk_words = word_count(chunk.text)
            if group and (
                len(group) >= self.parent_child_count
                or group_words + chunk_words > self.parent_max_words
            ):
                flush_group()
            group.append(chunk)
            group_words += chunk_words
            if group_words >= self.parent_target_words:
                flush_group()
        flush_group()
        return enriched

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        return self._attach_parent_metadata(self._child_chunks(document), document)


class AgenticGeminiChunker(ParagraphBlockSemanticMixin, BaseChunker):
    """Gemini-driven semantic boundary selection with deterministic safeguards.

    This is agentic chunking without OpenAI: Gemini proposes paragraph-block
    boundaries, then local code validates min/max size constraints so the build
    remains reproducible enough for benchmarking.
    """

    name = "agentic_gemini"

    def __init__(
        self,
        min_words: int = 120,
        target_words: int = 190,
        max_words: int = 300,
        long_paragraph_words: int = 260,
        agent_window_words: int = 1300,
        agent_max_units: int = 12,
        agent_max_input_chars: int = 7000,
        agent_max_output_tokens: int = 512,
        agent_temperature: float = 0.0,
        agent_model: str | None = None,
        use_agent: bool = True,
        fallback_threshold: float = 0.54,
    ) -> None:
        super().__init__(
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
            long_paragraph_words=long_paragraph_words,
            agent_window_words=agent_window_words,
            agent_max_units=agent_max_units,
            agent_max_input_chars=agent_max_input_chars,
            agent_max_output_tokens=agent_max_output_tokens,
            agent_temperature=agent_temperature,
            agent_model=agent_model,
            use_agent=use_agent,
            fallback_threshold=fallback_threshold,
        )
        if min_words < 1 or target_words < min_words or max_words < target_words:
            raise ValueError("Expected 1 <= min_words <= target_words <= max_words")
        if agent_window_words < target_words:
            raise ValueError("agent_window_words should be >= target_words")
        if agent_max_units < 2:
            raise ValueError("agent_max_units must be >= 2")
        self.min_words = min_words
        self.target_words = target_words
        self.max_words = max_words
        self.long_paragraph_words = long_paragraph_words
        self.agent_window_words = agent_window_words
        self.agent_max_units = agent_max_units
        self.agent_max_input_chars = agent_max_input_chars
        self.agent_max_output_tokens = agent_max_output_tokens
        self.agent_temperature = agent_temperature
        self.agent_model = agent_model
        self.use_agent = use_agent
        self.fallback_threshold = fallback_threshold
        self._client: Any | None = None
        self._client_failed = False

    def _fallback_chunker(self) -> AdaptiveParagraphChunker:
        return AdaptiveParagraphChunker(
            threshold=self.fallback_threshold,
            min_words=self.min_words,
            target_words=self.target_words,
            max_words=self.max_words,
            long_paragraph_words=self.long_paragraph_words,
        )

    def _get_client(self) -> Any | None:
        if not self.use_agent or self._client_failed:
            return None
        if self._client is not None:
            return self._client
        try:
            from google import genai

            project_id = os.environ.get(
                "VERTEX_PROJECT_ID",
                "project-ccf4c6cc-ed33-46e5-acf",
            )
            location = os.environ.get("VERTEX_LOCATION", "global")
            self._client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            return self._client
        except Exception:
            self._client_failed = True
            return None

    def _agent_model_name(self) -> str:
        return self.agent_model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def _agent_windows(
        self,
        units: list[tuple[str, int, int]],
    ) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(units):
            words = 0
            end = start
            while end < len(units):
                next_words = word_count(units[end][0])
                if end > start and (
                    words + next_words > self.agent_window_words
                    or end - start + 1 > self.agent_max_units
                ):
                    break
                words += next_words
                end += 1
            windows.append((start, max(start, end - 1)))
            start = max(start + 1, end)
        return windows

    def _prompt_for_window(
        self,
        document: Document,
        units: list[tuple[str, int, int]],
        start: int,
        end: int,
    ) -> str:
        blocks: list[str] = []
        used_chars = 0
        for local_index, unit_index in enumerate(range(start, end + 1)):
            text = " ".join(units[unit_index][0].split())
            max_unit_chars = 700
            if len(text) > max_unit_chars:
                text = text[:max_unit_chars].rstrip() + "..."
            block = f"[{local_index}] words={word_count(units[unit_index][0])}: {text}"
            if used_chars + len(block) > self.agent_max_input_chars:
                break
            blocks.append(block)
            used_chars += len(block)

        return f"""
You are doing semantic chunking for a RAG pipeline over Project Gutenberg books.
Use meaning, topic shifts, and paragraph continuity to choose chunk boundaries.
Do not use OpenAI tools. Return strict JSON only.

Goal:
- Prefer chunks around {self.target_words} words.
- Avoid chunks below {self.min_words} words unless the source is very short.
- Avoid chunks above {self.max_words} words.
- Preserve coherent ideas, examples, lists, and explanations together.

Book title: {document.title or "Untitled"}
Author: {document.author or "Unknown"}

Paragraph blocks:
{chr(10).join(blocks)}

Return schema:
{{
  "break_after": [0],
  "reason": "short explanation"
}}

`break_after` must contain local paragraph block indexes after which a chunk should end.
Do not include the final paragraph block index.
""".strip()

    def _call_agent(self, client: Any, prompt: str) -> dict[str, Any]:
        from google.genai import types

        response = client.models.generate_content(
            model=self._agent_model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.agent_max_output_tokens,
                temperature=self.agent_temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip() if isinstance(response.text, str) else str(response).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Agent returned non-JSON response: {text[:200]}")
        return json.loads(text[start : end + 1])

    def _agent_boundaries(
        self,
        document: Document,
        units: list[tuple[str, int, int]],
        client: Any,
    ) -> tuple[set[int], int]:
        boundaries: set[int] = set()
        failed_windows = 0
        for start, end in self._agent_windows(units):
            if end <= start:
                continue
            try:
                payload = self._call_agent(
                    client,
                    self._prompt_for_window(document, units, start, end),
                )
                proposed = payload.get("break_after", [])
                if not isinstance(proposed, list):
                    raise ValueError("break_after is not a list")
                for value in proposed:
                    try:
                        local_index = int(value)
                    except (TypeError, ValueError):
                        continue
                    global_index = start + local_index
                    if start <= global_index < end:
                        boundaries.add(global_index)
            except Exception:
                failed_windows += 1
        return boundaries, failed_windows

    def _build_chunks(
        self,
        document: Document,
        units: list[tuple[str, int, int]],
        boundaries: set[int],
        failed_windows: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        current_words = 0
        boundary_count = 0
        unit_texts = [unit[0] for unit in units]

        for index, text in enumerate(unit_texts):
            current_words += word_count(text)
            agent_break = index in boundaries
            can_break = current_words >= self.min_words
            must_break = current_words >= self.max_words
            should_break = index < len(units) - 1 and can_break and (agent_break or must_break)
            if should_break:
                metadata = {
                    "unit_type": "agentic_paragraph_block",
                    "agentic": True,
                    "agent_provider": "gemini_vertex",
                    "agent_model": self._agent_model_name(),
                    "agent_boundary_used": agent_break,
                    "agent_failed_windows": failed_windows,
                    "estimated_words": current_words,
                    "embedding_prefix": self._metadata_prefix(document),
                    "metadata_prefix_enabled": True,
                }
                chunks.append(
                    self.make_chunk(document, len(chunks), units, start, index, metadata)
                )
                boundary_count += int(agent_break)
                start = index + 1
                current_words = 0

        if start < len(units):
            final_words = sum(word_count(text) for text in unit_texts[start:])
            metadata = {
                "unit_type": "agentic_paragraph_block",
                "agentic": True,
                "agent_provider": "gemini_vertex",
                "agent_model": self._agent_model_name(),
                "agent_boundary_used": False,
                "agent_boundary_count": boundary_count,
                "agent_failed_windows": failed_windows,
                "estimated_words": final_words,
                "embedding_prefix": self._metadata_prefix(document),
                "metadata_prefix_enabled": True,
            }
            chunks.append(
                self.make_chunk(
                    document,
                    len(chunks),
                    units,
                    start,
                    len(units) - 1,
                    metadata,
                )
            )
        return chunks

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        units = self._paragraph_units(document)
        if not units:
            return []
        if len(units) == 1:
            return self._make_single_unit_chunk(
                document,
                units,
                {
                    "agentic": True,
                    "agent_provider": "gemini_vertex",
                    "agent_model": self._agent_model_name(),
                    "agent_boundary_count": 0,
                    "embedding_prefix": self._metadata_prefix(document),
                    "metadata_prefix_enabled": True,
                },
            )

        client = self._get_client()
        if client is None:
            if model is None:
                return self._build_chunks(document, units, set(), failed_windows=0)
            chunks = self._fallback_chunker().chunk(document, model=model, batch_size=batch_size)
            return [
                replace(
                    chunk,
                    chunk_id=chunk.chunk_id.replace("adaptive_paragraph:", f"{self.name}:", 1),
                    chunker=self.name,
                    metadata={
                        **chunk.metadata,
                        "chunker_params": self.params,
                        "agentic": True,
                        "agent_provider": "gemini_vertex",
                        "agent_fallback": "adaptive_paragraph",
                        "agent_model": self._agent_model_name(),
                    },
                )
                for chunk in chunks
            ]

        boundaries, failed_windows = self._agent_boundaries(document, units, client)
        if not boundaries and failed_windows:
            if model is None:
                return self._build_chunks(document, units, set(), failed_windows)
            chunks = self._fallback_chunker().chunk(document, model=model, batch_size=batch_size)
            return [
                replace(
                    chunk,
                    chunk_id=chunk.chunk_id.replace("adaptive_paragraph:", f"{self.name}:", 1),
                    chunker=self.name,
                    metadata={
                        **chunk.metadata,
                        "chunker_params": self.params,
                        "agentic": True,
                        "agent_provider": "gemini_vertex",
                        "agent_fallback": "adaptive_paragraph_after_agent_failure",
                        "agent_failed_windows": failed_windows,
                        "agent_model": self._agent_model_name(),
                    },
                )
                for chunk in chunks
            ]
        return self._build_chunks(document, units, boundaries, failed_windows)
