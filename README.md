# rag_chunking

Local RAG pipeline for comparing chunking strategies on a 100-book Project Gutenberg subset.

## Main Pieces

- `rag_pipeline/`: data loading, chunking, indexing, retrieval, evaluation, and widget API.
- `rag_pipeline/chunkers/`: chunking strategies.
- `data_clean/`: current 100-book text subset plus metadata.
- `widget/`: embeddable chatbot widget files.
- `CHUNKING_METHODS.md`: explanation of each chunking strategy and benchmark plan.
- `README_RAG.md`: commands for building indexes, running retrieval, and serving the widget.
- `compare_chunkers.py` and `compare_retrieval.py`: report-ready comparison tables.

## Notes

Generated FAISS indexes, embeddings, Python caches, virtual environments, and partial benchmark outputs are intentionally ignored by Git. Rebuild indexes locally with the commands in `README_RAG.md`.
