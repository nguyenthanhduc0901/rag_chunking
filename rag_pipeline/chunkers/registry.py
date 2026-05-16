from __future__ import annotations

import inspect
from typing import Any

from .adjacent import (
    AdaptiveParagraphChunker,
    AdjacentSimilarityChunker,
    FeedbackOptimizedChunker,
    HierarchicalAutoMergeChunker,
)
from .base import BaseChunker
from .fixed import FixedSentenceChunker


CHUNKERS: dict[str, type[BaseChunker]] = {
    FixedSentenceChunker.name: FixedSentenceChunker,
    AdjacentSimilarityChunker.name: AdjacentSimilarityChunker,
    AdaptiveParagraphChunker.name: AdaptiveParagraphChunker,
    HierarchicalAutoMergeChunker.name: HierarchicalAutoMergeChunker,
    FeedbackOptimizedChunker.name: FeedbackOptimizedChunker,
}


def list_chunkers() -> list[str]:
    return sorted(CHUNKERS)


def create_chunker(name: str, **params: Any) -> BaseChunker:
    if name not in CHUNKERS:
        available = ", ".join(list_chunkers())
        raise ValueError(f"Unknown chunker '{name}'. Available: {available}")
    chunker_cls = CHUNKERS[name]
    signature = inspect.signature(chunker_cls.__init__)
    accepted = set(signature.parameters) - {"self"}
    clean_params = {
        key: value
        for key, value in params.items()
        if value is not None and key in accepted
    }
    return chunker_cls(**clean_params)
