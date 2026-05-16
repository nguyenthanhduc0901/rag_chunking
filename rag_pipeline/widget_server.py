from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from .llm import answer_with_gemma
from .retrieve import Retriever, list_artifact_dirs


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
        answer_with_gemma(question, results)
        if use_llm
        else None
    )

    return JSONResponse(
        {
            "answer": answer.text if answer else "Retrieved chunks are shown below.",
            "used_llm": answer.used_llm if answer else False,
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


async def chatbot_js(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "chatbot.js", media_type="application/javascript")


async def root(request: Request) -> RedirectResponse:
    return RedirectResponse("/chatbot.html")


app = Starlette(
    routes=[
        Route("/", root),
        Route("/health", health),
        Route("/api/chunkers", chunkers),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/chatbot.html", chatbot_html),
        Route("/chatbot.js", chatbot_js),
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
