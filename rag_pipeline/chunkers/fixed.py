from __future__ import annotations

from rag_pipeline.schema import Chunk, Document

from .base import BaseChunker


class FixedSentenceChunker(BaseChunker):
    name = "fixed_sentence"

    def __init__(self, size: int = 8, overlap: int = 1) -> None:
        super().__init__(size=size, overlap=overlap)
        if size < 1:
            raise ValueError("size must be >= 1")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        spans = self.sentence_spans(document)
        if not spans:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(spans):
            end = min(start + self.size, len(spans)) - 1
            chunks.append(
                self.make_chunk(document, len(chunks), spans, start, end)
            )
            if end == len(spans) - 1:
                break
            start = end + 1 - self.overlap
        return chunks

