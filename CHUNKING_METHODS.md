# Tổng quan các phương pháp chunking

Dự án hiện có 8 phương pháp chunking chính. Các phương pháp này dùng chung pipeline RAG cơ bản: load 100 sách trong `data_clean`, tạo chunk, embed bằng `BAAI/bge-base-en-v1.5`, lưu FAISS local, rồi đánh giá retrieval bằng bộ câu hỏi source-grounded `eval/source_questions.jsonl`.

Kết luận hiện tại với các artifact đã benchmark: `feedback_optimized_v2` là phương pháp tốt nhất theo benchmark retrieval hiện tại. Nó đứng hạng 1 theo `source_mrr`, đồng thời vượt `fixed_sentence` ở `doc@1`, `doc_mrr`, `src@1`, `src@5`, `src@10`, `src_mrr` và có ít retrieval issues hơn. Hai phương pháp mới `parent_child_chunk` và `agentic_gemini` đã được thêm vào code, cần build/evaluate để có số liệu chính thức.

## 1. fixed_sentence

File code: `rag_pipeline/chunkers/fixed.py`

Ý tưởng: chia văn bản theo cửa sổ câu cố định. Mặc định mỗi chunk gồm 8 câu và overlap 1 câu với chunk kế tiếp.

Đặc điểm kỹ thuật:

- Không dùng embedding trong bước chunking.
- Dựa trên `split_sentences_with_offsets`.
- Mỗi chunk giữ `start_char`, `end_char`, `chunk_index` để đánh giá source overlap.
- Rất nhanh, dễ hiểu, ổn định.

Vai trò trong dự án: baseline mạnh. Ban đầu đây chỉ là phương pháp ngây thơ, nhưng kết quả benchmark cho thấy granularity nhỏ và overlap câu giúp retrieval source span rất tốt.

Nhược điểm:

- Không hiểu ranh giới ngữ nghĩa.
- Có thể cắt ngang một ý lớn nếu đoạn văn dài hoặc câu phân bố không đều.
- Tạo nhiều chunk hơn các phương pháp paragraph-semantic.

## 2. paragraph_semantic

File code: `rag_pipeline/chunkers/adjacent.py`, class `AdjacentSimilarityChunker`

Ý tưởng: dùng paragraph làm đơn vị gốc, sau đó dùng embedding để đo similarity giữa các paragraph block liền kề. Khi chunk đã đạt độ dài mục tiêu và similarity giữa hai block thấp hơn threshold, hệ thống xem đó là điểm ngắt ngữ nghĩa.

Đặc điểm kỹ thuật:

- Dùng paragraph block thay vì sentence để giảm chi phí embedding.
- Paragraph quá dài được tách tiếp thành các sentence-packed block.
- Dùng cosine similarity giữa block hiện tại và block kế tiếp.
- Threshold mặc định: `0.54`.
- Dải độ dài mặc định: `min_words=120`, `target_words=150`, `max_words=220`.

Vai trò trong dự án: semantic chunking baseline. Phương pháp này giữ đúng bản chất semantic chunking, nhưng trên dữ liệu Gutenberg hiện tại nó ít điểm truy cập hơn fixed sentence nên source-level retrieval thấp hơn.

## 3. adaptive_paragraph

File code: `rag_pipeline/chunkers/adjacent.py`, class `AdaptiveParagraphChunker`

Ý tưởng: vẫn dùng paragraph block, nhưng threshold ngắt được tính thích nghi theo từng tài liệu thay vì cố định toàn cục.

Đặc điểm kỹ thuật:

- Tính similarity theo cửa sổ lân cận (`window_size=2`) để giảm nhiễu.
- Tính threshold từ phân bố similarity của từng sách:
  - percentile score,
  - median trừ độ lệch chuẩn nhân hệ số,
  - kẹp trong khoảng `min_threshold=0.35` và `max_threshold=0.72`.
- Có `embedding_prefix` gồm title/author/language để embedding chunk có thêm metadata ngữ cảnh.

Vai trò trong dự án: cải tiến semantic baseline cho dữ liệu không đồng nhất. Sách có phong cách viết khác nhau không bị ép cùng một threshold.

Nhược điểm hiện tại: chunk vẫn khá lớn, median khoảng 327 estimated tokens. Điều này tốt cho ngữ cảnh nhưng kém hơn các cửa sổ câu nhỏ khi benchmark yêu cầu retrieve đúng source span hẹp.

## 4. hierarchical_auto_merge

File code: `rag_pipeline/chunkers/adjacent.py`, class `HierarchicalAutoMergeChunker`

Ý tưởng: tạo leaf chunk giống `adaptive_paragraph`, sau đó gom nhiều leaf gần nhau vào parent metadata. Retrieval vẫn tìm bằng leaf chunk, còn prompt QA có thể tự dùng parent context khi nhiều leaf cùng parent được retrieve.

Đặc điểm kỹ thuật:

- Leaf chunk giống adaptive paragraph.
- Parent context mặc định:
  - `parent_target_words=700`
  - `parent_max_words=950`
  - `parent_leaf_count=4`
- Metadata mỗi leaf có `parent_id`, `parent_text`, `parent_start_char`, `parent_end_char`.
- `rag_pipeline/llm.py` có logic auto-merge parent khi nhiều retrieved leaf cùng parent.

Vai trò trong dự án: phục vụ generation hơn là retrieval thuần. Nó hữu ích cho chatbot khi câu hỏi cần nhiều đoạn liên tiếp, nhưng trong benchmark retrieval hiện tại, điểm số giống `adaptive_paragraph` vì FAISS vẫn index leaf chunk.

## 5. parent_child_chunk

File code: `rag_pipeline/chunkers/adjacent.py`, class `ParentChildChunker`

Ý tưởng: index các child chunk nhỏ để retrieval chính xác, nhưng gắn parent context lớn hơn để chatbot/Gemini có đủ ngữ cảnh khi sinh câu trả lời.

Đặc điểm kỹ thuật:

- Child chunk là cửa sổ câu nhỏ:
  - `sentence_window_size=8`
  - `sentence_window_overlap=2`
- Parent context gom nhiều child liền kề:
  - `parent_target_words=700`
  - `parent_max_words=950`
  - `parent_child_count=5`
- Mỗi child có metadata `parent_id`, `parent_text`, `parent_start_char`, `parent_end_char`.
- Metadata `parent_context_policy=always` cho phép `rag_pipeline/llm.py` dùng parent context ngay cả khi chỉ retrieve được một child trong parent đó.
- Embedding vẫn chạy trên child text, kèm `embedding_prefix` title/author/language.

Vai trò trong dự án: đây là bản parent-child rõ ràng hơn `hierarchical_auto_merge`. Nó giữ lợi thế retrieval của chunk nhỏ gần `feedback_optimized_v2`, đồng thời cải thiện answer quality nhờ đưa parent context vào prompt.

Đánh đổi:

- Parent text có thể làm prompt dài hơn, nên `llm.py` vẫn cắt theo `RAG_CONTEXT_TOKEN_BUDGET` và `RAG_MAX_CONTEXT_BLOCK_TOKENS`.
- Retrieval metric có thể giống sentence-window methods, còn lợi ích chính nên thể hiện rõ hơn ở answer quality benchmark.

## 6. feedback_optimized

File code: `rag_pipeline/chunkers/adjacent.py`, class `FeedbackOptimizedChunker`

Ý tưởng: vòng feedback đầu tiên dựa trên semantic paragraph chunking. Phương pháp này làm chunk nhỏ hơn semantic baseline, giảm chunk quá dài, và gắn metadata để hỗ trợ phân tích sau retrieval eval.

Đặc điểm kỹ thuật:

- Kế thừa `AdaptiveParagraphChunker`.
- Cấu hình hiện tại:
  - `min_words=90`
  - `target_words=130`
  - `max_words=190`
  - `long_paragraph_words=190`
  - `adaptive_percentile=25`
  - `std_factor=0.65`
  - `window_size=2`
  - `repair_margin=0.04`
- Gắn metadata:
  - `feedback_ready=true`
  - `near_boundary`
  - `repair_policy=adaptive_small_leaf`
  - `suggested_action`

Vai trò trong dự án: semantic chunking cải tiến có feedback metadata. Đây là phương pháp semantic thuần tốt nhất hiện tại trong nhóm paragraph-based, đứng thứ 3 toàn bộ benchmark.

Nhược điểm:

- Ít chunk hơn nên tiết kiệm storage, nhưng source-level retrieval vẫn thua fixed sentence và v2.
- Median token 288, còn lớn nếu câu hỏi benchmark nhắm vào một đoạn hẹp.

## 7. feedback_optimized_v2

File code: `rag_pipeline/chunkers/adjacent.py`, class `FeedbackOptimizedV2Chunker`

Ý tưởng hiện tại: đây là vòng feedback thứ hai. Sau khi benchmark cho thấy `fixed_sentence` thắng các phương pháp paragraph-semantic ở source-level retrieval, v2 được chuyển thành chiến lược retrieval-first: compact overlapping sentence windows, kèm metadata prefix trong embedding.

Đặc điểm kỹ thuật:

- Cửa sổ câu mặc định:
  - `sentence_window_size=8`
  - `sentence_window_overlap=2`
- Có `embedding_prefix` gồm title/author/language.
- Metadata có:
  - `unit_type=sentence_window`
  - `feedback_ready=true`
  - `feedback_iteration=2`
  - `repair_policy=feedback_v2_source_local_window`
  - `semantic_feedback_from=feedback_optimized_and_fixed_sentence_eval`
- Không cần embedding model ở bước chunking, nhưng vẫn dùng BGE ở bước build index/retrieval.

Vai trò trong dự án: phương pháp tốt nhất hiện tại theo benchmark. Nó không còn là pure paragraph semantic chunking; đúng hơn, đây là hybrid feedback-optimized chunking: dùng kết quả đánh giá của semantic v1 và fixed baseline để chọn chunk granularity tối ưu cho retrieval.

Đánh đổi:

- Tạo nhiều chunk nhất: 9,088 chunks.
- Tốn storage và embedding time hơn paragraph-semantic.
- Đổi lại, retrieval source span tốt nhất và số retrieval issues thấp nhất.

## 8. agentic_gemini

File code: `rag_pipeline/chunkers/adjacent.py`, class `AgenticGeminiChunker`

Ý tưởng: agentic chunking không dùng OpenAI. Gemini/Vertex đóng vai trò agent chọn ranh giới semantic giữa các paragraph block, sau đó code nội bộ kiểm tra lại bằng rule để giữ chunk trong ngưỡng độ dài an toàn.

Đặc điểm kỹ thuật:

- Không dùng OpenAI API.
- Dùng Gemini qua Vertex AI nếu có credential:
  - env `VERTEX_PROJECT_ID`
  - env `VERTEX_LOCATION`
  - env `GEMINI_MODEL`, mặc định `gemini-2.5-flash`
- Đơn vị đầu vào là paragraph block.
- Gemini chỉ đề xuất `break_after` theo index paragraph block.
- Code nội bộ vẫn enforce:
  - `min_words=120`
  - `target_words=190`
  - `max_words=300`
  - `long_paragraph_words=260`
- Nếu Gemini không gọi được hoặc trả JSON lỗi, chunker fallback sang `adaptive_paragraph` để quá trình build không bị hỏng.
- Metadata ghi rõ `agentic=true`, `agent_provider=gemini_vertex`, `agent_model`, `agent_failed_windows` hoặc `agent_fallback`.

Vai trò trong dự án: đây là phương pháp agentic thật sự đầu tiên. Nó không chỉ dựa trên similarity hay sentence window, mà để LLM đọc một cụm paragraph và đề xuất điểm ngắt theo chủ đề.

Đánh đổi:

- Chậm và tốn chi phí API hơn các phương pháp còn lại.
- Có biến thiên theo model/prompt, nên benchmark cần ghi rõ model Gemini, ngày chạy và cấu hình.
- Vì có fallback, cần kiểm tra manifest/chunks metadata để biết bao nhiêu phần thật sự dùng agent và bao nhiêu phần fallback.

## Artifact hiện tại

Các artifact index chính nằm trong `artifacts/<chunker>/`:

```text
chunks.jsonl
embeddings.npy
faiss.index
manifest.json
```

Các kết quả đánh giá được gom riêng trong `artifacts/evaluations/`:

```text
artifacts/evaluations/<chunker>/chunk_eval.json
artifacts/evaluations/<chunker>/retrieval_eval.json
artifacts/evaluations/<chunker>/retrieval_eval_top10.json
artifacts/evaluations/<chunker>/feedback_report.json
artifacts/evaluations/<chunker>/answer_quality_eval.json
artifacts/evaluations/retrieval_summary.md
artifacts/evaluations/retrieval_summary.json
artifacts/evaluations/answer_quality/answer_quality_summary.md
artifacts/evaluations/answer_quality/answer_quality_summary.json
```

| Chunker | Số sách | Số chunk | GPU build | Build time | Median tokens | Chunk >512 | Retrieval issues |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| feedback_optimized_v2 | 100 | 9,088 | cuda | 123.68s | 146 | 29 | 97 |
| parent_child_chunk | 100 | 9,088 | cuda | 133.37s | 146 | 29 | 97 |
| fixed_sentence | 100 | 7,815 | cuda | 107.57s | 145 | 33 | 111 |
| agentic_gemini | 100 | 3,656 | cuda | 2119.50s | 261 | 118 | 110 |
| feedback_optimized | 100 | 3,516 | cuda | 177.93s | 288 | 8 | 106 |
| paragraph_semantic | 100 | 3,276 | cuda | 177.04s | 315 | 30 | 112 |
| adaptive_paragraph | 100 | 3,092 | cuda | 178.25s | 327 | 33 | 100 |
| hierarchical_auto_merge | 100 | 3,092 | cuda | 179.02s | 327 | 33 | 100 |

## Benchmark hiện tại

Bộ benchmark chính: `eval/source_questions.jsonl`

- 300 câu hỏi.
- Sinh từ source material gốc trong `data_clean`.
- Cân bằng theo scope: `local`, `multi_paragraph`, `book_theme`.
- Không phụ thuộc artifact của bất kỳ chunker nào.
- Mỗi câu hỏi có `expected_doc_id`, một hoặc nhiều source span gốc, expected keywords và answer hint.

Framework đánh giá:

- `rag_pipeline/evaluate_chunks.py`: intrinsic chunk metrics.
- `rag_pipeline/evaluate_retrieval.py`: retrieval metrics theo doc và source span.
- `rag_pipeline/feedback_report.py`: liệt kê câu hỏi/chunk có vấn đề để phục vụ vòng feedback tiếp theo.
- `rag_pipeline/evaluate_answer_quality.py`: sinh câu trả lời RAG và dùng Gemini judge để chấm answer quality.
- `compare_retrieval.py`: gom kết quả thành `artifacts/evaluations/retrieval_summary.md` và `artifacts/evaluations/retrieval_summary.json`.
- `compare_answer_quality.py`: gom kết quả answer quality thành `artifacts/evaluations/answer_quality/answer_quality_summary.md`.

## Kết quả retrieval hiện tại

| Rank | Chunker | doc@1 | doc@5 | doc@10 | doc_mrr | src@1 | src@5 | src@10 | span_recall@5 | full_src@5 | src_mrr |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | feedback_optimized_v2 | 83.0% | 92.0% | 94.7% | 0.869 | 51.7% | 76.3% | 82.0% | 54.6% | 36.0% | 0.610 |
| 2 | parent_child_chunk | 83.0% | 92.0% | 94.7% | 0.869 | 51.7% | 76.3% | 82.0% | 54.6% | 36.0% | 0.610 |
| 3 | fixed_sentence | 81.3% | 91.3% | 94.7% | 0.851 | 48.0% | 73.0% | 78.3% | 52.7% | 36.0% | 0.580 |
| 4 | adaptive_paragraph | 78.3% | 92.0% | 94.0% | 0.838 | 46.0% | 76.7% | 84.0% | 55.1% | 37.7% | 0.579 |
| 5 | hierarchical_auto_merge | 78.3% | 92.0% | 94.0% | 0.838 | 46.0% | 76.7% | 84.0% | 55.1% | 37.7% | 0.579 |
| 6 | feedback_optimized | 78.0% | 90.3% | 93.7% | 0.827 | 46.0% | 75.0% | 86.0% | 53.6% | 35.0% | 0.568 |
| 7 | agentic_gemini | 80.3% | 91.3% | 93.3% | 0.848 | 45.7% | 74.0% | 82.7% | 52.4% | 34.3% | 0.567 |
| 8 | paragraph_semantic | 76.7% | 91.3% | 92.7% | 0.826 | 41.7% | 75.3% | 82.0% | 53.4% | 35.7% | 0.543 |

## Kết quả answer quality mini benchmark

Answer quality benchmark dùng 20 câu được chọn từ `eval/source_questions.jsonl`, cân bằng tương đối theo scope:

- `local`: 7 câu
- `multi_paragraph`: 7 câu
- `book_theme`: 6 câu

Quy trình đánh giá:

1. Với mỗi chunker, retrieve top-5 chunk.
2. Gemini sinh câu trả lời dựa trên retrieved context.
3. Gemini judge chấm câu trả lời theo correctness, faithfulness, completeness, context usefulness và citation support.

Kết quả hiện tại nằm trong `artifacts/evaluations/answer_quality/answer_quality_summary.md`.

| Rank | Chunker | Overall | Correct | Faithful | Pass |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | paragraph_semantic | 4.550 | 4.350 | 4.800 | 95.0% |
| 2 | agentic_gemini | 4.370 | 4.100 | 4.700 | 85.0% |
| 3 | adaptive_paragraph | 4.370 | 4.100 | 4.650 | 85.0% |
| 4 | feedback_optimized | 4.350 | 4.150 | 4.550 | 80.0% |
| 5 | feedback_optimized_v2 | 4.350 | 4.150 | 4.600 | 75.0% |
| 6 | fixed_sentence | 4.330 | 4.150 | 4.500 | 85.0% |
| 7 | hierarchical_auto_merge | 3.660 | 3.350 | 3.950 | 70.0% |
| 8 | parent_child_chunk | 3.590 | 3.500 | 3.600 | 75.0% |

Nhận định quan trọng: `feedback_optimized_v2` tốt nhất ở retrieval, nhưng `paragraph_semantic` tốt nhất trong mini benchmark answer quality 20 câu. Điều này cho thấy retrieval tốt nhất không luôn đồng nghĩa với câu trả lời tốt nhất. Chunk semantic lớn hơn có thể cung cấp ngữ cảnh mạch lạc hơn cho LLM, đặc biệt với câu hỏi nhiều đoạn hoặc câu hỏi mang tính tổng hợp.

## Nhận định báo cáo

Nếu mục tiêu là chứng minh cải tiến chunking qua retrieval benchmark, nên trình bày theo trục sau:

1. `fixed_sentence` là baseline rất mạnh vì chunk nhỏ và có overlap.
2. `paragraph_semantic`, `adaptive_paragraph`, `hierarchical_auto_merge` giữ bản chất semantic nhưng chunk lớn hơn, nên source-level retrieval thường kém hơn chunk nhỏ.
3. `feedback_optimized` là semantic v1: tiết kiệm chunk hơn, ít overlong chunk hơn, và tốt nhất trong nhóm paragraph-semantic nếu xét retrieval/storage trade-off.
4. `feedback_optimized_v2` là bước cải tiến dựa trên feedback thực nghiệm: chuyển sang sentence window có overlap 2 và metadata prefix. Nó đánh đổi số lượng chunk để lấy source-level retrieval tốt nhất.
5. `paragraph_semantic` đang là phương pháp tốt nhất trong mini benchmark answer quality, cho thấy semantic chunking vẫn có lợi khi đánh giá câu trả lời cuối cùng thay vì chỉ đo retrieval.
6. `parent_child_chunk` là ứng viên mới đáng kỳ vọng cho answer quality vì kết hợp child retrieval chính xác với parent context giàu ngữ cảnh.
7. `agentic_gemini` là ứng viên mới giàu tính nghiên cứu nhất, nhưng cần kiểm soát chi phí và kiểm tra tỷ lệ fallback.

Phương pháp tốt nhất theo retrieval hiện tại: `feedback_optimized_v2`.

Phương pháp tốt nhất theo answer quality mini benchmark hiện tại: `paragraph_semantic`.

Phương pháp semantic paragraph tốt nhất hiện tại: `feedback_optimized`.

Phương pháp baseline mạnh nhất: `fixed_sentence`.
