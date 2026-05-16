# Tổng quan các phương pháp chunking

Dự án hiện có 5 phương pháp chunking, trong đó `fixed_sentence` là baseline, `paragraph_semantic` là semantic baseline đã đổi từ tên thử nghiệm cũ, và 3 phương pháp còn lại là các hướng cải tiến chính.

## 1. fixed_sentence

Ý tưởng: chia văn bản theo số câu cố định, mặc định 8 câu mỗi chunk và overlap 1 câu.

Vai trò: baseline đơn giản để so sánh. Phương pháp này không dùng embedding trong bước chunking, chạy nhanh, ổn định, nhưng không hiểu ranh giới ngữ nghĩa. Nó dễ tạo nhiều chunk nhỏ và có thể cắt ngang một ý lớn nếu cấu trúc câu không đều.

## 2. paragraph_semantic

Ý tưởng: chia sách thành các paragraph block trước, sau đó dùng embedding BAAI/bge-base-en-v1.5 để đo độ giống nhau giữa các block liền kề. Khi chunk đã đạt độ dài tối thiểu/độ dài mục tiêu và similarity giữa hai block thấp hơn threshold, hệ thống xem đó là điểm ngắt ngữ nghĩa.

Vai trò: semantic chunking cơ bản. So với chia từng câu, cách này tiết kiệm embedding cost vì số đơn vị đầu vào ít hơn nhiều, phù hợp với dữ liệu Gutenberg đã có phân đoạn tương đối rõ. Các paragraph quá dài được tách tiếp thành block nhỏ hơn để tránh chunk vượt ngưỡng.

## 3. adaptive_paragraph

Ý tưởng: vẫn bắt đầu từ paragraph block, nhưng threshold ngắt không còn cố định toàn cục. Mỗi tài liệu tự tính phân bố similarity và chọn threshold thích nghi theo nội dung sách. Similarity cũng được tính theo cửa sổ lân cận để giảm nhiễu ở các đoạn ngắn.

Điểm cải tiến: phù hợp hơn với bộ sách không đồng nhất. Sách có văn phong liền mạch và sách có nhiều mục ngắn sẽ không bị ép cùng một threshold. Phương pháp này giữ bản chất semantic chunking, nhưng giảm rủi ro over-split hoặc under-split do threshold cố định.

## 4. hierarchical_auto_merge

Ý tưởng: tạo leaf chunk giống `adaptive_paragraph`, đồng thời lưu thêm parent context lớn hơn. Khi retrieval lấy được nhiều leaf chunk cùng parent, prompt QA tự gộp parent context một lần để câu trả lời có ngữ cảnh rộng hơn.

Điểm cải tiến: retrieval vẫn dùng chunk nhỏ/vừa để tìm kiếm chính xác, còn generation được hỗ trợ bởi ngữ cảnh rộng hơn khi cần. Cách này đặc biệt hợp với sách, vì nhiều câu hỏi cần vài đoạn liền nhau thay vì một đoạn rời.

## 5. feedback_optimized

Ý tưởng: dùng semantic paragraph chunking nhưng ưu tiên chunk gọn hơn và ít vượt giới hạn token hơn. Metadata được thiết kế để phục vụ vòng lặp đánh giá: chạy retrieval eval trên bộ câu hỏi source-grounded, rồi tạo feedback report để phát hiện những vùng chunking cần chỉnh.

Điểm cải tiến: đây là hướng tối ưu thực nghiệm. Nó không chỉ tạo chunk, mà còn chuẩn bị dữ liệu để lặp lại quá trình đo - sửa - đo. Với artifact hiện tại, phương pháp này có số chunk vượt 512 estimated tokens thấp nhất trong nhóm semantic.

## Artifact hiện tại

| Chunker | Số sách | Số chunk | GPU build | Thời gian build | Median estimated tokens | Chunk > 512 tokens |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fixed_sentence | 100 | 7,815 | cuda | 107.57s | 145 | 33 |
| paragraph_semantic | 100 | 3,276 | cuda | 177.04s | 315 | 30 |
| adaptive_paragraph | 100 | 3,092 | cuda | 178.25s | 327 | 33 |
| hierarchical_auto_merge | 100 | 3,092 | cuda | 179.02s | 327 | 33 |
| feedback_optimized | 100 | 3,516 | cuda | 177.93s | 288 | 8 |

## Benchmark đang có

- Intrinsic chunk benchmark: mỗi artifact có `chunk_eval.json`, gồm số chunk, số sách, độ dài ký tự/token ước lượng, số câu mỗi chunk, số chunk vượt 512 estimated tokens, số chunk quá ngắn, boundary similarity và số chunk mỗi sách.
- Bảng so sánh nhanh: chạy `python compare_chunkers.py` để gom các chỉ số chunk từ mọi artifact.
- Retrieval benchmark framework: `rag_pipeline/evaluate_retrieval.py` đo `hit@1`, `hit@k`, `mrr`, keyword recall và top results từ file câu hỏi JSONL.
- Source-grounded benchmark generator: `rag_pipeline/generate_benchmark_questions.py` sinh câu hỏi từ paragraph gốc trong `data_clean`, không lấy từ artifact của bất kỳ chunker nào, nên công bằng hơn khi so sánh các phương pháp chunking.
- Feedback report: `rag_pipeline/feedback_report.py` dùng kết quả retrieval eval để phân tích điểm yếu của chunking.

## Kết quả benchmark hiện tại

`eval/source_questions.jsonl` hiện là bộ benchmark chính, gồm 200 câu hỏi được sinh từ paragraph gốc của 100 sách. Kết quả retrieval đã được ghi trong `artifacts/retrieval_summary.md` và `artifacts/retrieval_summary.json`.

Các chỉ số chính để báo cáo là `doc_hit@1`, `doc_hit@5`, `doc_mrr`, `source_hit@1`, `source_hit@5`, `source_hit@10`, `source_mrr` và số retrieval issues trong feedback report.
