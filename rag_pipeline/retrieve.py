from __future__ import annotations

import argparse
from pathlib import Path

import faiss

from .embeddings import MODEL_NAME, encode_query, load_embedding_model, resolve_device
from .io_utils import read_chunks_jsonl, read_json
from .schema import SearchResult


class Retriever:
    def __init__(
        self,
        artifact_dir: Path,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.manifest = read_json(artifact_dir / "manifest.json")
        self.chunks = read_chunks_jsonl(artifact_dir / "chunks.jsonl")
        self.index = faiss.read_index(str(artifact_dir / "faiss.index"))
        self.model_name = model_name or self.manifest.get("embedding_model") or MODEL_NAME
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model = load_embedding_model(self.model_name, device=self.device)

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = encode_query(self.model, question, batch_size=self.batch_size)
        scores, indices = self.index.search(query_vector, top_k)
        results: list[SearchResult] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                    rank=rank,
                )
            )
        return results


def list_artifact_dirs(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        return []
    return sorted(
        path
        for path in artifacts_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a built FAISS index.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    retriever = Retriever(
        args.artifact_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    results = retriever.search(args.question, top_k=args.top_k)
    for result in results:
        chunk = result.chunk
        print(f"#{result.rank} score={result.score:.4f}")
        print(f"title={chunk.title} | author={chunk.author} | doc_id={chunk.doc_id}")
        print(chunk.text[:1200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()

