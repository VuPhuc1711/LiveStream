# Checkpoint

Backend nạp `phobert_binary_v04_20261003_092559_043789/seed42/best`.
Các checkpoint/lượt thử khác đã chuyển nguyên bản sang
`D:\lưu trữ\livestream-violation-detection_20261028\models`.
Không xóa, sửa trọng số hoặc khóa SHA256. Không phải mọi model có trong thư mục này
đều đang chạy; xem pipeline_model.json và /health để biết phiên bản thực tế.
Checkpoint hiện hành và tokenizer/lock/metadata không đổi. Các đường dẫn dữ liệu
train/validation trong metadata là lịch sử thật của lượt train v04; không sửa
metadata khóa chỉ vì tạo bố cục train/test mới. Muốn dùng lại checkpoint cũ cần
khôi phục về đúng đường dẫn gốc theo manifest trước khi rollback.

Backend hiện hành hỗ trợ clone repository sang vị trí khác/Windows/Linux: giữ
nguyên cấu trúc models/<run>/seed42/best và lock ở thư mục cha, xác minh đầy đủ
SHA256 theo định danh model tương đối; không sửa đường dẫn trong lock gốc.
Máy cùng phát triển cần nhận riêng toàn bộ run v04 (gồm5 shard), vì Git không
chứa trọng số. Xem docker/README.md và config/runtime_assets.json để bàn giao
cùng cache Whisper Turbo. Các công cụ lịch sử giữ nguyên và chưa được chuyển
đổi; không tự chạy lại trainer/evaluator từ kho lưu trữ.
