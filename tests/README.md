# Kiểm thử ứng dụng hiện tại

- `backend/`: API, realtime, database/lịch sử/feedback, upload và chất lượng ASR.
- `speech/`: cấu hình và bộ nạp Whisper với model/fixture giả lập được ghi rõ.
- `test_project_structure.py`: đường dẫn cấu hình, archive và ranh giới code hiện hành.
- `test_model_portability.py`: đổi vị trí repository/Windows-Linux vẫn giữ đúng
  định danh và đầy đủ hash; từ chối sai/thừa/thiếu artifact.
- `test_container_entrypoint.py`: URL database Docker, mật khẩu có ký tự đặc
  biệt và lệnh bảo trì không tự migrate/nạp model.
- Test giao diện ở `frontend/src/*.test.tsx` / `*.test.ts`.

```powershell
.\.venv-nlp\Scripts\python.exe -m unittest discover -s tests -t .
# Trong Docker, gồm PostgreSQL integration trên các database tạm:
docker compose run --rm --no-deps -e RUN_POSTGRES_TESTS=1 backend python -m unittest discover -s tests -t .
```

PostgreSQL integration mặc định bỏ qua; bật `RUN_POSTGRES_TESTS=1` chỉ khi có
Docker sẵn sàng. Các test tạo database tên riêng và chỉ xóa database tạm của test.
Không sửa/xóa dữ liệu sử dụng thật. Test nghiên cứu cũ ở `archive/research/`,
không trộn với bộ kiểm tra ứng dụng binary_v04 hiện tại.

Các nguồn lịch sử trên hiện ở thư mục ngoài dự án do `config/storage.json` chỉ
định; test kiểm tra archive được bỏ qua rõ ràng nếu ổ lưu trữ không có trên máy
khác. Bộ test dữ liệu mới xác nhận584 train/182 test, hash/nhãn và family tách biệt;
182 câu test là validation cũ đã xem, không được quảng bá là test độc lập.
