from __future__ import annotations

from pathlib import Path

from .io_utils import read_json
from .schema import Document


def load_metadata(data_dir: Path) -> dict[str, dict]:
    metadata_path = data_dir / "metadata.json"
    rows = read_json(metadata_path)
    if isinstance(rows, list):
        return {str(row["id"]): row for row in rows}
    if isinstance(rows, dict):
        return {str(key): value for key, value in rows.items()}
    raise ValueError(f"Unsupported metadata format in {metadata_path}")


def load_documents(data_dir: Path, limit: int | None = None) -> list[Document]:
    metadata_by_id = load_metadata(data_dir)
    paths = sorted(
        data_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if limit is not None:
        paths = paths[:limit]

    documents: list[Document] = []
    for path in paths:
        doc_id = path.stem
        metadata = metadata_by_id.get(doc_id, {})
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        documents.append(
            Document(
                doc_id=doc_id,
                source_file=str(path),
                text=text,
                title=metadata.get("title"),
                author=metadata.get("author"),
                language=metadata.get("language"),
                gutenberg_url=metadata.get("gutenberg_url"),
                metadata=metadata,
            )
        )
    return documents

