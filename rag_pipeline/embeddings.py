from __future__ import annotations

import time
from typing import Iterable

import numpy as np


MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def resolve_device(requested: str | None) -> str:
    requested_device = requested or "cuda"
    if requested_device != "cuda":
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        raise RuntimeError(
            "CUDA is required for model tasks, but PyTorch could not check CUDA."
        ) from exc
    raise RuntimeError(
        "CUDA is required for model tasks, but torch.cuda.is_available() is false. "
        "Run this outside the sandbox/on the VM host where the NVIDIA GPU is visible."
    )


def load_embedding_model(model_name: str = MODEL_NAME, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=resolve_device(device))


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def encode_texts(
    model,
    texts: list[str],
    batch_size: int = 64,
    progress_label: str | None = None,
    progress_every: int = 25,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    started = time.perf_counter()
    for batch_index, batch in enumerate(batched(texts, batch_size), start=1):
        vectors.append(
            model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        if progress_label and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - started
            done = min(batch_index * batch_size, len(texts))
            rate = done / elapsed if elapsed else 0.0
            remaining = (len(texts) - done) / rate if rate else 0.0
            print(
                f"{progress_label}: batch {batch_index}/{total_batches} "
                f"items {done}/{len(texts)} "
                f"rate={rate:.1f}/s elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(remaining)}",
                flush=True,
            )
    return np.asarray(np.vstack(vectors), dtype=np.float32)


def encode_query(model, question: str, batch_size: int = 64) -> np.ndarray:
    return encode_texts(model, [QUERY_INSTRUCTION + question], batch_size=batch_size)
