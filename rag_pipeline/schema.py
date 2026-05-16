from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_file: str
    text: str
    title: str | None = None
    author: str | None = None
    language: str | None = None
    gutenberg_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunker: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    title: str | None = None
    author: str | None = None
    language: str | None = None
    gutenberg_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Chunk":
        return cls(**row)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

