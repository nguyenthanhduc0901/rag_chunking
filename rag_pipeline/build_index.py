from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np

from .chunkers import create_chunker, list_chunkers
from .embeddings import MODEL_NAME, encode_texts, load_embedding_model, resolve_device
from .evaluate_chunks import evaluate_chunks_file
from .io_utils import write_chunks_jsonl, write_json
from .load_data import load_documents


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _format_rate(count: int, elapsed: float, unit: str) -> str:
    if elapsed <= 0:
        return f"0 {unit}/s"
    return f"{count / elapsed:.2f} {unit}/s"


def _embedding_text(chunk) -> str:
    prefix = chunk.metadata.get("embedding_prefix")
    if prefix:
        return f"{prefix}\n\n{chunk.text}"
    return chunk.text


def build_artifacts(
    data_dir: Path,
    artifacts_dir: Path,
    chunker_name: str,
    model_name: str,
    device: str | None,
    batch_size: int,
    limit: int | None,
    threshold: float | None,
    min_words: int | None,
    target_words: int | None,
    max_words: int | None,
    long_paragraph_words: int | None,
    adaptive_percentile: float | None,
    std_factor: float | None,
    min_threshold: float | None,
    max_threshold: float | None,
    window_size: int | None,
    parent_target_words: int | None,
    parent_max_words: int | None,
    parent_leaf_count: int | None,
    repair_margin: float | None,
    fixed_size: int | None,
    fixed_overlap: int | None,
) -> Path:
    resolved_device = resolve_device(device)
    chunker_params = {
        "threshold": threshold,
        "min_words": min_words,
        "target_words": target_words,
        "max_words": max_words,
        "long_paragraph_words": long_paragraph_words,
        "adaptive_percentile": adaptive_percentile,
        "std_factor": std_factor,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "window_size": window_size,
        "parent_target_words": parent_target_words,
        "parent_max_words": parent_max_words,
        "parent_leaf_count": parent_leaf_count,
        "repair_margin": repair_margin,
        "size": fixed_size,
        "overlap": fixed_overlap,
    }
    chunker = create_chunker(chunker_name, **chunker_params)
    out_dir = artifacts_dir / chunker.name
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    documents = load_documents(data_dir, limit=limit)
    total_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
    total_words = sum(len(document.text.split()) for document in documents)
    print("=" * 88, flush=True)
    print(f"Build started: chunker={chunker.name}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(
        f"Loaded {len(documents)} documents from {data_dir} | "
        f"size={total_bytes / 1024 / 1024:.1f}MB | words~{total_words:,}",
        flush=True,
    )
    print(
        f"Embedding model: {model_name} | device={resolved_device} | "
        f"batch_size={batch_size}",
        flush=True,
    )
    print(f"Chunker params: {chunker.params}", flush=True)
    print("Loading embedding model...", flush=True)
    model = load_embedding_model(model_name, device=resolved_device)
    print(
        f"Model loaded in {_format_duration(time.perf_counter() - started)}",
        flush=True,
    )

    all_chunks = []
    chunking_started = time.perf_counter()
    chunk_total = 0
    for index, document in enumerate(documents, start=1):
        doc_started = time.perf_counter()
        chunks = chunker.chunk(document, model=model, batch_size=batch_size)
        all_chunks.extend(chunks)
        chunk_total += len(chunks)
        elapsed = time.perf_counter() - chunking_started
        doc_elapsed = time.perf_counter() - doc_started
        docs_remaining = len(documents) - index
        avg_per_doc = elapsed / index if index else 0.0
        eta = docs_remaining * avg_per_doc
        doc_words = len(document.text.split())
        print(
            f"[{index}/{len(documents)}] {document.doc_id} "
            f"{document.title or ''} | "
            f"words~{doc_words:,} | chunks={len(chunks)} | "
            f"total_chunks={chunk_total:,} | "
            f"doc_time={_format_duration(doc_elapsed)} | "
            f"elapsed={_format_duration(elapsed)} | "
            f"eta={_format_duration(eta)} | "
            f"rate={_format_rate(index, elapsed, 'docs')}",
            flush=True,
        )

    chunks_path = out_dir / "chunks.jsonl"
    print(
        f"Chunking complete: {len(all_chunks):,} chunks from {len(documents)} docs "
        f"in {_format_duration(time.perf_counter() - chunking_started)}",
        flush=True,
    )
    print(f"Writing chunks JSONL: {chunks_path}", flush=True)
    write_chunks_jsonl(chunks_path, all_chunks)
    print(f"Wrote {len(all_chunks):,} chunks to {chunks_path}", flush=True)

    texts = [_embedding_text(chunk) for chunk in all_chunks]
    print(f"Embedding {len(texts):,} final chunks...", flush=True)
    embedding_started = time.perf_counter()
    embeddings = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        progress_label="Final chunk embedding",
        progress_every=10,
    )
    print(
        f"Final chunk embedding complete in "
        f"{_format_duration(time.perf_counter() - embedding_started)}",
        flush=True,
    )
    embeddings_path = out_dir / "embeddings.npy"
    print(f"Saving embeddings: {embeddings_path}", flush=True)
    np.save(embeddings_path, embeddings)

    dimension = int(embeddings.shape[1])
    print(f"Building FAISS IndexFlatIP dimension={dimension}", flush=True)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    index_path = out_dir / "faiss.index"
    print(f"Writing FAISS index: {index_path}", flush=True)
    faiss.write_index(index, str(index_path))

    print("Evaluating chunks...", flush=True)
    chunk_eval = evaluate_chunks_file(chunks_path)
    write_json(out_dir / "chunk_eval.json", chunk_eval)

    manifest = {
        "chunker": chunker.name,
        "chunker_params": chunker.params,
        "embedding_model": model_name,
        "device_requested": device,
        "device_used": resolved_device,
        "normalize_embeddings": True,
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "embedding_dimension": dimension,
        "source_data_dir": str(data_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "files": {
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "faiss_index": str(index_path),
            "chunk_eval": str(out_dir / "chunk_eval.json"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote index to {index_path}", flush=True)
    print(f"Wrote manifest to {out_dir / 'manifest.json'}", flush=True)
    print(
        f"Build finished in {_format_duration(time.perf_counter() - started)}",
        flush=True,
    )
    print("=" * 88, flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS RAG index.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_clean"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--chunker", choices=list_chunkers(), required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu")
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--fixed-size", type=int, default=None)
    parser.add_argument("--fixed-overlap", type=int, default=None)
    args = parser.parse_args()

    build_artifacts(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        chunker_name=args.chunker,
        model_name=args.model,
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
        fixed_size=args.fixed_size,
        fixed_overlap=args.fixed_overlap,
    )


if __name__ == "__main__":
    main()
