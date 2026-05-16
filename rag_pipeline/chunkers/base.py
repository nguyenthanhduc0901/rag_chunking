from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_pipeline.schema import Chunk, Document
from rag_pipeline.text_utils import (
    split_paragraphs_with_offsets,
    split_sentences_with_offsets,
)


class BaseChunker(ABC):
    name = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def chunk(self, document: Document, model=None, batch_size: int = 64) -> list[Chunk]:
        raise NotImplementedError

    def sentence_spans(self, document: Document) -> list[tuple[str, int, int]]:
        return split_sentences_with_offsets(document.text)

    def paragraph_spans(self, document: Document) -> list[tuple[str, int, int]]:
        return split_paragraphs_with_offsets(document.text)

    def make_chunk(
        self,
        document: Document,
        chunk_index: int,
        sentence_spans: list[tuple[str, int, int]],
        start_sentence: int,
        end_sentence: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        start_char = sentence_spans[start_sentence][1]
        end_char = sentence_spans[end_sentence][2]
        text = document.text[start_char:end_char].strip()
        metadata = dict(document.metadata)
        metadata.update(
            {
                "start_sentence": start_sentence,
                "end_sentence": end_sentence,
                "chunker_params": self.params,
            }
        )
        if extra_metadata:
            metadata.update(extra_metadata)
        return Chunk(
            chunk_id=f"{self.name}:{document.doc_id}:{chunk_index:06d}",
            doc_id=document.doc_id,
            source_file=document.source_file,
            chunker=self.name,
            chunk_index=chunk_index,
            text=text,
            start_char=start_char,
            end_char=end_char,
            title=document.title,
            author=document.author,
            language=document.language,
            gutenberg_url=document.gutenberg_url,
            metadata=metadata,
        )
