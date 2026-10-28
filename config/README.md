# Cấu hình hiện hành và vị trí lưu trữ

Hai cấu hình **đang chạy** vẫn ở gốc repository: `pipeline_model.json` và
`asr_config.json`. Backend đọc chúng khi khởi động; không đổi checkpoint/nhãn.

- `data/current_dataset.json`: hai split train/test mới, hash/nguồn và vai trò hồi quy.
- `runtime_assets.json`: định danh model v04, hash bản khóa gốc và cache Turbo để
  bàn giao riêng cho máy khác; kiểm tra bằng `tools.check_container_assets`.
- `storage.example.json`: bản mẫu được đưa lên Git cho máy khác.
- `storage.json`: vị trí kho lịch sử riêng của máy hiện tại; Git bỏ qua file này.
  Không cần file này để chạy backend, frontend hoặc quản lý dữ liệu hiện hành.

Chỉ khi đã có bản sao kho lịch sử và muốn kiểm tra hash, sao chép bản mẫu:

```powershell
if (Test-Path -LiteralPath .\config\storage.json) { throw 'storage.json đã tồn tại; giữ nguyên cấu hình local.' }
Copy-Item -LiteralPath .\config\storage.example.json -Destination .\config\storage.json
```

Sửa `external_root` và `transfer_manifest` thành vị trí thật trên máy của bạn.
`E:/project-history` chỉ là ví dụ, không phải đường dẫn bắt buộc. Không ghi đè
file `storage.json` đã cấu hình của máy khác. Bản mẫu không chứa dữ liệu lịch sử.
Nếu nhận cả hai file CLI lưu trữ ngày28/10, thêm
`"additional_manifests": ["manifest_legacy_cli_20261028.json"]` vào cấu hình local
để kiểm tra hai file đó cùng lịch sử cũ. Mỗi lần chuyển bổ sung có manifest riêng;
không sửa các manifest cũ để cập nhật tổng số file.
Không cần sao chép bản mẫu nếu chỉ phát triển/chạy ứng dụng. Test kiểm tra kho
lưu trữ sẽ được báo bỏ qua khi thiếu cấu hình hoặc không có kho trên máy mới.

`rollback/`, `asr_trials/`, `training/` cũ đã ở
`D:\lưu trữ\livestream-violation-detection_20261028\config`.

Việc có cấu hình train không có nghĩa được phép train. Hiện tạm dừng ở binary_v04.
Các đường dẫn cũ trong cấu hình hoặc báo cáo lịch sử giữ nguyên để bảo toàn bằng
chứng. Manifest chuyển ngoài dự án là `manifest_20261028.json` tại root lưu trữ;
manifest73 file trước đó ở `archive/manifest_20261024.json` bên trong cùng root đó.
