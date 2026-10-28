# Sắp xếp repository — 24/10/2026

Thư mục gốc giảm **82 → 16 file** (gồm `.env` và file cấu hình test mới).
Phân biệt rõ mã ứng dụng, test, công cụ, tài liệu và lịch sử nghiên cứu.

- `backend/`, `frontend/`, `speech/`, `nlp/`: đường chạy ứng dụng hiện tại.
- `tests/`: kiểm thử hiện hành; `tools/diagnostics/`: replay/kiểm tra audio.
- `config/`: các bản rollback, cấu hình ASR thử và cấu hình train lịch sử.
- `archive/`:73 file gốc được bảo toàn, có đường dẫn và SHA256 trong manifest.
- README mới mô tả Turbo/binary_v04/PostgreSQL, cách chạy và cách đọc code.
- PROJECT_STATUS mới nêu trạng thái/giới hạn hiện tại; tài liệu đầy đủ cũ được giữ.

Tách hàm nạp model từ CLI cũ sang `backend/model_runtime.py`; cấu hình dùng
`backend/model_config.py`. Hàm giữ nguyên hành vi. `main.py`/`model_config.py` gốc
vẫn giữ tương thích. Mã Rules/Hybrid cũ được lưu trong archive, không nằm trong
đường import ứng dụng. Không train/đổi model/nhãn/dataset trong bước sắp xếp.

Chỉ xóa cache Python sinh lại được; VS Code ẩn môi trường/cache/node_modules/dist.
Không xóa dataset, checkpoint, report, debug, lịch sử hay feedback. Không commit/push.
Script lưu trữ không được tự chạy lại; tái lập nghiên cứu cần đúng đường dẫn/hash cũ.

**Kiểm tra:**94 test ứng dụng,32 test PostgreSQL database tạm,61 test frontend và
build đạt.73 hash archive khớp.83 phiên/438 đoạn/16 feedback không đổi cả nội dung
và quan hệ. FE5173 HTTP200; BE8000 ready với Turbo CUDA/FP16 + binary_v04 CPU +
PostgreSQL. Server đã bật lại. Sắp xếp mã không chứng minh độ chính xác model;
kiểm chứng toàn phiên tab của bản sửa ASR trước đó vẫn chưa chốt.

Đọc [README](../README.md) trước, sau đó
[Hướng dẫn dự án](HUONG_DAN_DU_AN.md). Bằng chứng kỹ thuật local ở
`runtime/reorganization_20261024/`.
