from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from .chunkers import create_chunker, list_chunkers
from .embeddings import MODEL_NAME, load_embedding_model
from .llm import answer_with_gemini
from .retrieve import Retriever, list_artifact_dirs
from .schema import Document
from .text_utils import estimate_tokens, word_count


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "widget"
ARTIFACTS_DIR = Path(os.environ.get("RAG_ARTIFACTS_DIR", ROOT / "artifacts"))
DEVICE = os.environ.get("RAG_DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("RAG_BATCH_SIZE", "64"))


def _artifact_dir(chunker: str | None = None) -> Path:
    artifact_dirs = list_artifact_dirs(ARTIFACTS_DIR)
    if not artifact_dirs:
        raise FileNotFoundError(
            f"No built indexes found in {ARTIFACTS_DIR}. Build one first."
        )
    if chunker:
        for path in artifact_dirs:
            if path.name == chunker:
                return path
        available = ", ".join(path.name for path in artifact_dirs)
        raise ValueError(f"Unknown chunker '{chunker}'. Available: {available}")
    return artifact_dirs[0]


@lru_cache(maxsize=8)
def _retriever(artifact_dir: str) -> Retriever:
    return Retriever(Path(artifact_dir), device=DEVICE, batch_size=BATCH_SIZE)


@lru_cache(maxsize=1)
def _preview_embedding_model():
    return load_embedding_model(MODEL_NAME, device=DEVICE)


def _preview_document(text: str, title: str | None = None) -> Document:
    return Document(
        doc_id="preview",
        source_file="user_input",
        text=text,
        title=title or "Preview document",
        author="User provided",
        language="en",
        metadata={"source": "chunk_playground"},
    )


def _needs_embedding_model(chunker_name: str) -> bool:
    return chunker_name in {
        "paragraph_semantic",
        "adaptive_paragraph",
        "hierarchical_auto_merge",
        "feedback_optimized",
        "agentic_gemini",
    }


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _chunk_metrics(chunks) -> dict:
    words = [word_count(chunk.text) for chunk in chunks]
    tokens = [estimate_tokens(chunk.text) for chunk in chunks]
    chars = [len(chunk.text) for chunk in chunks]
    if not chunks:
        return {
            "num_chunks": 0,
            "word_min": 0,
            "word_median": 0,
            "word_avg": 0,
            "word_max": 0,
            "token_avg": 0,
            "over_512_tokens": 0,
            "total_words": 0,
        }
    return {
        "num_chunks": len(chunks),
        "word_min": min(words),
        "word_median": _percentile(words, 0.5),
        "word_p90": _percentile(words, 0.9),
        "word_avg": sum(words) / len(words),
        "word_max": max(words),
        "token_avg": sum(tokens) / len(tokens),
        "char_avg": sum(chars) / len(chars),
        "over_512_tokens": sum(1 for value in tokens if value > 512),
        "total_words": sum(words),
    }


async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "device": DEVICE,
            "artifacts_dir": str(ARTIFACTS_DIR),
            "chunkers": [path.name for path in list_artifact_dirs(ARTIFACTS_DIR)],
        }
    )


async def chunkers(request: Request) -> JSONResponse:
    rows = []
    for path in list_artifact_dirs(ARTIFACTS_DIR):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": path.name,
                "num_documents": manifest.get("num_documents"),
                "num_chunks": manifest.get("num_chunks"),
                "params": manifest.get("chunker_params"),
            }
        )
    return JSONResponse({"chunkers": rows})


async def preview_chunkers(request: Request) -> JSONResponse:
    built = {path.name for path in list_artifact_dirs(ARTIFACTS_DIR)}
    return JSONResponse(
        {
            "chunkers": [
                {
                    "name": name,
                    "requires_embedding": _needs_embedding_model(name),
                    "built_index": name in built,
                    "note": (
                        "Uses Gemini for boundary proposals; slower and may cost API calls."
                        if name == "agentic_gemini"
                        else ""
                    ),
                }
                for name in list_chunkers()
            ]
        }
    )


async def chunk_preview(request: Request) -> JSONResponse:
    payload = await request.json()
    text = str(payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    max_returned_chunks = int(os.environ.get("RAG_CHUNK_PREVIEW_MAX_RETURNED_CHUNKS", "500"))

    requested = payload.get("chunkers") or payload.get("chunker") or []
    if isinstance(requested, str):
        requested = [requested]
    requested = [str(name) for name in requested if str(name).strip()]
    if not requested:
        requested = ["fixed_sentence", "paragraph_semantic", "parent_child_chunk"]

    available = set(list_chunkers())
    unknown = [name for name in requested if name not in available]
    if unknown:
        return JSONResponse(
            {"error": f"Unknown chunker(s): {', '.join(unknown)}"},
            status_code=400,
        )

    document = _preview_document(text, title=str(payload.get("title") or "Preview document"))
    reports = []
    for name in requested:
        chunker = create_chunker(name)
        model = _preview_embedding_model() if _needs_embedding_model(name) else None
        started = time.perf_counter()
        try:
            chunks = chunker.chunk(document, model=model, batch_size=BATCH_SIZE)
            metrics = _chunk_metrics(chunks)
            visible_chunks = chunks[:max_returned_chunks]
            elapsed_seconds = time.perf_counter() - started
            reports.append(
                {
                    "chunker": name,
                    "params": chunker.params,
                    "metrics": metrics,
                    "elapsed_seconds": round(elapsed_seconds, 4),
                    "returned_chunks": len(visible_chunks),
                    "truncated_chunks": len(chunks) > len(visible_chunks),
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "words": word_count(chunk.text),
                            "estimated_tokens": estimate_tokens(chunk.text),
                            "metadata": {
                                key: value
                                for key, value in chunk.metadata.items()
                                if key
                                in {
                                    "unit_type",
                                    "hierarchy_level",
                                    "parent_id",
                                    "parent_estimated_words",
                                    "agentic",
                                    "agent_fallback",
                                    "agent_failed_windows",
                                }
                            },
                            "text": chunk.text,
                        }
                        for chunk in visible_chunks
                    ],
                }
            )
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - started
            reports.append(
                {
                    "chunker": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "params": chunker.params,
                    "metrics": _chunk_metrics([]),
                    "elapsed_seconds": round(elapsed_seconds, 4),
                    "chunks": [],
                }
            )

    return JSONResponse(
        {
            "document": {
                "chars": len(text),
                "words": word_count(text),
                "estimated_tokens": estimate_tokens(text),
            },
            "reports": reports,
        }
    )


async def chat(request: Request) -> JSONResponse:
    payload = await request.json()
    question = str(payload.get("message") or payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "message is required"}, status_code=400)

    chunker = payload.get("chunker")
    top_k = int(payload.get("top_k") or 5)
    use_llm = bool(payload.get("use_llm", True))

    artifact_dir = _artifact_dir(str(chunker) if chunker else None)
    retriever = _retriever(str(artifact_dir))
    results = retriever.search(question, top_k=top_k)
    answer = (
        answer_with_gemini(question, results)
        if use_llm
        else None
    )

    return JSONResponse(
        {
            "answer": answer.text if answer else "Retrieved chunks are shown below.",
            "used_llm": answer.used_llm if answer else False,
            "llm_provider": "gemini" if answer and answer.used_llm else None,
            "chunker": artifact_dir.name,
            "results": [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "chunk_id": result.chunk.chunk_id,
                    "doc_id": result.chunk.doc_id,
                    "title": result.chunk.title,
                    "author": result.chunk.author,
                    "gutenberg_url": result.chunk.gutenberg_url,
                    "text": result.chunk.text,
                }
                for result in results
            ],
        }
    )


async def chatbot_html(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "chatbot.html")


async def chunk_playground_html(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "chunk-playground.html")


async def chatbot_js(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "chatbot.js", media_type="application/javascript")


async def app_css(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "app.css", media_type="text/css")


async def root(request: Request) -> RedirectResponse:
    return RedirectResponse("/chatbot.html")


app = Starlette(
    routes=[
        Route("/", root),
        Route("/health", health),
        Route("/api/chunkers", chunkers),
        Route("/api/preview-chunkers", preview_chunkers),
        Route("/api/chunk-preview", chunk_preview, methods=["POST"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/chatbot.html", chatbot_html),
        Route("/chunk-playground.html", chunk_playground_html),
        Route("/chatbot.js", chatbot_js),
        Route("/app.css", app_css),
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
