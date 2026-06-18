# Báo Cáo Ngắn - CPU LightGBM Benchmark trên AWS

Do tài khoản AWS mới bị giới hạn quota GPU/vCPU, em không thể triển khai GPU instance như kế hoạch ban đầu và chuyển sang phương án CPU với LightGBM trên EC2.

Benchmark được thực hiện trên bộ dữ liệu Credit Card Fraud Detection gồm 284,807 giao dịch và 31 cột dữ liệu. Thời gian load data đạt 2.03 giây, thời gian training đạt 5.49 giây, với best iteration là 268.

Mô hình đạt AUC-ROC 0.9642, cho thấy khả năng phân biệt giao dịch gian lận và giao dịch bình thường khá tốt. Accuracy đạt 0.9995, tuy nhiên do dataset bị mất cân bằng nên các chỉ số F1-score, Precision và Recall có ý nghĩa quan trọng hơn.

Kết quả chi tiết cho bài toán fraud detection: F1-score đạt 0.8454, Precision đạt 0.8542 và Recall đạt 0.8367. Các chỉ số này cho thấy mô hình có khả năng phát hiện giao dịch gian lận tương đối tốt trong khi vẫn giữ tỷ lệ dự đoán sai ở mức hợp lý.

Tốc độ inference trên CPU khá nhanh: latency cho 1 dòng dữ liệu là 1.72 ms. Khi chạy batch 1000 dòng, thời gian inference là 0.0041 giây, tương đương throughput khoảng 241,641 dòng/giây.

So với GPU, phương án CPU không phù hợp cho các mô hình deep learning lớn như LLM, nhưng rất phù hợp với workload ML truyền thống như LightGBM. CPU instance giúp hoàn thành lab mà không cần xin quota GPU, chi phí dễ kiểm soát hơn và vẫn tạo đủ kết quả benchmark để đánh giá pipeline Terraform, cloud instance, training và inference.

## Bảng kết quả

| Metric | Kết quả |
|---|---:|
| Số dòng dataset | 284,807 |
| Thời gian load data | 2.03 giây |
| Thời gian training | 5.49 giây |
| Best iteration | 268 |
| AUC-ROC | 0.9642 |
| Accuracy | 0.9995 |
| F1-Score | 0.8454 |
| Precision | 0.8542 |
| Recall | 0.8367 |
| Inference latency, 1 row | 1.72 ms |
| Inference throughput, 1000 rows | 241,641 rows/sec |
