# Tổng quan các phương pháp chunking

Dự án hiện có 6 phương pháp chunking chính. Các phương pháp này dùng chung pipeline RAG cơ bản: load 100 sách trong `data_clean`, tạo chunk, embed bằng `BAAI/bge-base-en-v1.5`, lưu FAISS local, rồi đánh giá retrieval bằng bộ câu hỏi source-grounded `eval/source_questions.jsonl`.

Kết luận hiện tại: `feedback_optimized_v2` là phương pháp tốt nhất theo benchmark retrieval hiện tại. Nó đứng hạng 1 theo `source_mrr`, đồng thời vượt `fixed_sentence` ở `doc@1`, `doc_mrr`, `src@1`, `src@5`, `src@10`, `src_mrr` và có ít retrieval issues hơn.

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

## 5. feedback_optimized

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

## 6. feedback_optimized_v2

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

## Artifact hiện tại

Các artifact chính nằm trong `artifacts/<chunker>/`:

```text
chunks.jsonl
embeddings.npy
faiss.index
chunk_eval.json
retrieval_eval.json
retrieval_eval_top10.json
feedback_report.json
manifest.json
```

| Chunker | Số sách | Số chunk | GPU build | Build time | Median tokens | Chunk >512 | Retrieval issues |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| feedback_optimized_v2 | 100 | 9,088 | cuda | 123.68s | 146 | 29 | 43 |
| fixed_sentence | 100 | 7,815 | cuda | 107.57s | 145 | 33 | 50 |
| feedback_optimized | 100 | 3,516 | cuda | 177.93s | 288 | 8 | 58 |
| paragraph_semantic | 100 | 3,276 | cuda | 177.04s | 315 | 30 | 70 |
| adaptive_paragraph | 100 | 3,092 | cuda | 178.25s | 327 | 33 | 72 |
| hierarchical_auto_merge | 100 | 3,092 | cuda | 179.02s | 327 | 33 | 72 |

## Benchmark hiện tại

Bộ benchmark chính: `eval/source_questions.jsonl`

- 200 câu hỏi.
- Sinh từ source paragraph gốc trong `data_clean`.
- Không phụ thuộc artifact của bất kỳ chunker nào.
- Mỗi câu hỏi có `expected_doc_id`, source span gốc, expected keywords và answer hint.

Framework đánh giá:

- `rag_pipeline/evaluate_chunks.py`: intrinsic chunk metrics.
- `rag_pipeline/evaluate_retrieval.py`: retrieval metrics theo doc và source span.
- `rag_pipeline/feedback_report.py`: liệt kê câu hỏi/chunk có vấn đề để phục vụ vòng feedback tiếp theo.
- `compare_retrieval.py`: gom kết quả thành `artifacts/retrieval_summary.md` và `artifacts/retrieval_summary.json`.

## Kết quả retrieval hiện tại

| Rank | Chunker | doc@1 | doc@5 | doc@10 | doc_mrr | src@1 | src@5 | src@10 | src_mrr |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | feedback_optimized_v2 | 82.5% | 93.0% | 95.0% | 0.865 | 63.5% | 86.0% | 90.5% | 0.721 |
| 2 | fixed_sentence | 82.0% | 93.0% | 96.0% | 0.863 | 59.0% | 82.5% | 88.5% | 0.680 |
| 3 | feedback_optimized | 77.0% | 89.0% | 92.0% | 0.817 | 51.5% | 78.5% | 86.5% | 0.619 |
| 4 | paragraph_semantic | 75.5% | 91.0% | 94.0% | 0.810 | 48.0% | 73.0% | 81.5% | 0.573 |
| 5 | adaptive_paragraph | 74.5% | 89.0% | 92.5% | 0.802 | 43.5% | 77.0% | 85.0% | 0.557 |
| 6 | hierarchical_auto_merge | 74.5% | 89.0% | 92.5% | 0.802 | 43.5% | 77.0% | 85.0% | 0.557 |

## Nhận định báo cáo

Nếu mục tiêu là chứng minh cải tiến chunking qua retrieval benchmark, nên trình bày theo trục sau:

1. `fixed_sentence` là baseline rất mạnh vì chunk nhỏ và có overlap.
2. `paragraph_semantic`, `adaptive_paragraph`, `hierarchical_auto_merge` giữ bản chất semantic nhưng chunk lớn hơn, nên source-level retrieval kém hơn.
3. `feedback_optimized` là semantic v1: tiết kiệm chunk hơn, ít overlong chunk hơn, và tốt nhất trong nhóm paragraph-semantic.
4. `feedback_optimized_v2` là bước cải tiến dựa trên feedback thực nghiệm: chuyển sang sentence window có overlap 2 và metadata prefix. Nó đánh đổi số lượng chunk để lấy source-level retrieval tốt nhất.

Phương pháp tốt nhất hiện tại: `feedback_optimized_v2`.

Phương pháp semantic paragraph tốt nhất hiện tại: `feedback_optimized`.

Phương pháp baseline mạnh nhất: `fixed_sentence`.
