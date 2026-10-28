# Công cụ bảo trì và chẩn đoán

Chạy từ gốc repository bằng `.\.venv-nlp\Scripts\python.exe -m ...`.

- `tools.check_container_assets`: kiểm tra offline14 artifact v04, bản khóa gốc
  và SHA cache Turbo; không tải/nạp model hoặc nối database. Trong Docker dùng
  `docker compose run --rm --no-deps backend python -m tools.check_container_assets`.

- `tools.verify_archive`: kiểm tra73 nguồn lịch sử và hai file CLI trong manifest
  bổ sung; `--all` kiểm tra thêm754 file của đợt chuyển trước. Vị trí và danh sách
  manifest bổ sung được đọc từ `config/storage.json` local; không bắt buộc có kho
  lịch sử trên máy chỉ chạy ứng dụng.
- `tools.diagnostics.smoke_live`: replay audio qua backend đang chạy. Công cụ này
  tạo phiên lịch sử, chỉ chạy khi được phép và chỉ định đúng môi trường.
- `tools.diagnostics.audit_live_audio`: kiểm tra WAV/trace của một phiên debug.

Không có cơ chế train tự động. Các script nghiên cứu cũ ở `archive/research/`
dưới `D:\lưu trữ\livestream-violation-detection_20261028`, không thuộc code ứng dụng.
