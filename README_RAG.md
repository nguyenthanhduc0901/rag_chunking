# Gutenberg Environmental Issues RAG

This project compares chunking strategies inside a simple local RAG pipeline.

## Build Indexes

Use CUDA when available:

```bash
.venv/bin/python build_all_indexes.py --device cuda --batch-size 256

.venv/bin/python -m rag_pipeline.build_index --chunker paragraph_semantic --device cuda --batch-size 256
.venv/bin/python -m rag_pipeline.build_index --chunker adaptive_paragraph --device cuda --batch-size 256
.venv/bin/python -m rag_pipeline.build_index --chunker hierarchical_auto_merge --device cuda --batch-size 256
.venv/bin/python -m rag_pipeline.build_index --chunker feedback_optimized --device cuda --batch-size 256
.venv/bin/python -m rag_pipeline.build_index --chunker feedback_optimized_v2 --device cuda --batch-size 256
```

`build_all_indexes.py` builds the three improved methods by default:

```text
adaptive_paragraph
hierarchical_auto_merge
feedback_optimized
feedback_optimized_v2
```

For a fast smoke test:

```bash
.venv/bin/python -m rag_pipeline.build_index --chunker adaptive_paragraph --limit 2 --device cuda
```

Each chunking method writes its own artifact folder:

```text
artifacts/<chunker>/
  chunks.jsonl
  embeddings.npy
  faiss.index
  chunk_eval.json
  manifest.json
```

## Search

```bash
.venv/bin/python -m rag_pipeline.retrieve \
  --artifact-dir artifacts/paragraph_semantic \
  --question "What does Thoreau say about walking?"
```

## Embeddable Widget

Run the widget/API server on the VM:

```bash
RAG_DEVICE=cuda RAG_ARTIFACTS_DIR=artifacts \
  .venv/bin/uvicorn rag_pipeline.widget_server:app --host 0.0.0.0 --port 8787
```

Use an iframe directly:

```html
<iframe src="http://localhost:8787/chatbot.html" width="400" height="600"></iframe>
```

Or inject a floating widget:

```html
<script src="http://localhost:8787/chatbot.js"></script>
```

For the real Gutenberg HTTPS page, prefer forwarding the VM port to your local
machine and loading the widget from `http://localhost:8787`, because browsers may
block scripts or iframes from a plain HTTP external VM IP on an HTTPS page.

## Compare Chunkers

```bash
.venv/bin/python compare_chunkers.py
.venv/bin/python compare_retrieval.py
```

## Generate A Fair Retrieval Benchmark

Generate questions from the original `data_clean` source passages, not from any
chunker artifact:

```bash
.venv/bin/python -m rag_pipeline.generate_benchmark_questions \
  --data-dir data_clean \
  --out eval/source_questions.jsonl \
  --questions-per-doc 2 \
  --seed 42
```

Quick smoke test without calling Gemma:

```bash
.venv/bin/python -m rag_pipeline.generate_benchmark_questions \
  --data-dir data_clean \
  --out eval/source_questions_smoke.jsonl \
  --limit-docs 2 \
  --questions-per-doc 1 \
  --no-llm
```

Evaluate any chunker against the same benchmark:

```bash
.venv/bin/python -m rag_pipeline.evaluate_retrieval \
  --artifact-dir artifacts/adaptive_paragraph \
  --questions eval/source_questions.jsonl \
  --top-k 5 \
  --device cuda
```

After running retrieval evaluation for all artifact folders, regenerate the
summary table:

```bash
.venv/bin/python compare_retrieval.py
```

## Feedback Loop

```bash
.venv/bin/python -m rag_pipeline.evaluate_retrieval \
  --artifact-dir artifacts/feedback_optimized \
  --questions eval/source_questions.jsonl

.venv/bin/python -m rag_pipeline.feedback_report \
  --artifact-dir artifacts/feedback_optimized \
  --retrieval-eval artifacts/feedback_optimized/retrieval_eval.json
```
