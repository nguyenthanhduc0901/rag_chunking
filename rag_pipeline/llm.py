from __future__ import annotations

import os
from dataclasses import dataclass

from .schema import SearchResult
from .text_utils import estimate_tokens


@dataclass(frozen=True)
class Answer:
    text: str
    used_llm: bool


def _truncate_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= max_tokens:
        return text, False

    max_words = max(1, int(max_tokens / 1.3))
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]).rstrip() + "\n[Context truncated to fit prompt budget.]", True


def build_prompt(
    question: str,
    results: list[SearchResult],
    context_token_budget: int | None = None,
    max_block_tokens: int | None = None,
) -> str:
    context_token_budget = context_token_budget or int(
        os.environ.get("RAG_CONTEXT_TOKEN_BUDGET", "1100")
    )
    max_block_tokens = max_block_tokens or int(
        os.environ.get("RAG_MAX_CONTEXT_BLOCK_TOKENS", "320")
    )
    context_blocks = []
    used_parent_ids: set[str] = set()
    used_context_tokens = 0

    for result in results:
        chunk = result.chunk
        parent_id = chunk.metadata.get("parent_id")
        parent_text = chunk.metadata.get("parent_text")
        parent_context_policy = chunk.metadata.get("parent_context_policy")
        sibling_hits = (
            sum(1 for item in results if item.chunk.metadata.get("parent_id") == parent_id)
            if parent_id
            else 0
        )
        should_use_parent = (
            bool(parent_id and parent_text)
            and (
                parent_context_policy == "always"
                or sibling_hits >= int(os.environ.get("RAG_PARENT_MERGE_MIN_HITS", "2"))
            )
        )
        if should_use_parent:
            if parent_id in used_parent_ids:
                continue
            used_parent_ids.add(parent_id)
            child_ranks = [
                str(item.rank)
                for item in results
                if item.chunk.metadata.get("parent_id") == parent_id
            ]
            block_text, truncated = _truncate_to_token_budget(
                str(parent_text),
                max_block_tokens,
            )
            header = (
                f"[{result.rank}] {chunk.title or 'Untitled'} by {chunk.author or 'Unknown'} "
                f"(doc_id={chunk.doc_id}, score={result.score:.3f}, "
                f"auto_merged_parent=true, child_ranks={','.join(child_ranks)}, "
                f"truncated={str(truncated).lower()})"
            )
        else:
            block_text, truncated = _truncate_to_token_budget(
                chunk.text,
                max_block_tokens,
            )
            header = (
                f"[{result.rank}] {chunk.title or 'Untitled'} by {chunk.author or 'Unknown'} "
                f"(doc_id={chunk.doc_id}, score={result.score:.3f}, "
                f"truncated={str(truncated).lower()})"
            )

        block = f"{header}\n{block_text}"
        block_tokens = estimate_tokens(block)
        remaining_tokens = context_token_budget - used_context_tokens
        if remaining_tokens <= 0:
            break
        header_tokens = estimate_tokens(header)
        if remaining_tokens <= header_tokens + 8:
            break
        if block_tokens > remaining_tokens:
            trimmed_text, _ = _truncate_to_token_budget(
                block_text,
                remaining_tokens - header_tokens,
            )
            block = f"{header}\n{trimmed_text}"
            block_tokens = estimate_tokens(block)

        context_blocks.append(block)
        used_context_tokens += block_tokens

    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the provided Project Gutenberg context. "
        "If the context is insufficient, say that the retrieved context is not enough.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def make_gemini_client():
    from google import genai

    project_id = os.environ.get("VERTEX_PROJECT_ID", "project-ccf4c6cc-ed33-46e5-acf")
    location = os.environ.get("VERTEX_LOCATION", "global")
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


def answer_with_gemini(
    question: str,
    results: list[SearchResult],
    model: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Answer:
    prompt = build_prompt(question, results)
    model_name = model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    output_tokens = max_output_tokens or int(
        os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "512")
    )
    temp = (
        temperature
        if temperature is not None
        else float(os.environ.get("GEMINI_TEMPERATURE", "0"))
    )

    client = None
    try:
        from google.genai import types

        client = make_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=(
                "You are a careful RAG assistant for Project Gutenberg books. "
                "Answer only from the provided context. Cite the bracketed context "
                "numbers when useful. If the context is insufficient, say so clearly.\n\n"
                f"{prompt}"
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=output_tokens,
                temperature=temp,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        text = response.text.strip() if isinstance(response.text, str) else str(response).strip()
        return Answer(text=text, used_llm=True)
    except Exception as exc:
        return Answer(
            text=(
                "Gemini is not reachable or returned an error. Retrieval still worked, "
                "so use the chunks below as context. Check Vertex AI credentials and "
                "the Gemini environment variables.\n\n"
                f"LLM error: {type(exc).__name__}: {exc}"
            ),
            used_llm=False,
        )
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()
