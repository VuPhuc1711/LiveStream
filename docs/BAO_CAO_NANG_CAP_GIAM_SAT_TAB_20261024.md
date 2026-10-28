# Báo cáo ngắn: bổ sung giám sát tab livestream

Ngày thực hiện: 24/10/2026. Mục đích: hỗ trợ demo giám sát lời nói trong một phiên
livestream đang phát trên trình duyệt, bên cạnh chức năng microphone hiện có.

## Các nâng cấp

1. **Hai chức năng riêng:** Giám sát microphone và Giám sát tab livestream, có liên
   kết truy cập nhanh. Upload và thu âm thành file được giữ nguyên.
2. **Thu âm thanh tab:** người dùng chủ động chọn tab Chrome/Edge và chia sẻ audio.
   Có thông báo khi hủy, chọn sai nguồn hoặc thiếu audio. Chỉ audio được gửi lên
   backend; không gửi hình ảnh, URL hoặc tiêu đề tab.
3. **Tái sử dụng pipeline:** audio được chuyển thành PCM16 mono/16kHz rồi xử lý qua
   WebSocket → Silero VAD → Whisper Turbo → PhoBERT binary_v04. Giữ hai nhãn
   KHONG_CANH_BAO/CANH_BAO; không thay model hoặc huấn luyện.
4. **Kết thúc an toàn:** dừng chia sẻ vẫn gửi phần audio cuối và chờ xử lý hết.
   Khóa các luồng để tránh chạy chồng; chặn kết quả trả muộn của phiên cũ. Hàng chờ,
   tồn đọng và trạng thái hoàn tất tiếp tục được hiển thị.
5. **Lịch sử và bàn giao:** bổ sung nguồn phiên tab trong PostgreSQL và báo cáo
   JSON, giữ phản hồi riêng với dự đoán gốc. Có migration và hướng dẫn chạy/demo
   trong README. Dữ liệu cũ được sao lưu và đối chiếu hash trước/sau migration.

## Kết quả kiểm tra

| Kiểm tra | Kết quả |
| --- | --- |
| Backend/API, dữ liệu tạm | 71 test đạt |
| PostgreSQL riêng cho test | 32 test đạt |
| Frontend và build | 58 test đạt; build thành công |
| Replay giọng thật 34,72 giây, cả hai nguồn | Giữ đúng hai câu đã xác nhận, gồm Zalo/TikTok/rẻ nhất và phủ định; tồn đọng cuối 0 |
| Replay bản im lặng thật 94,75 giây | Không transcript, không câu phân loại, không gọi ASR |
| Stress 3.600 chunk, tương đương 30 phút audio | Không mất/trùng kết quả, tồn đọng cuối 0; dùng mock VAD/ASR và phát tăng tốc |
| Bảo toàn PostgreSQL sau migration | 81 phiên, 396 đoạn, 146 trace, 16 feedback, 14 mẫu bổ sung, 26 revision, 3 CSV export giữ nguyên hash |

## Giới hạn và cách demo

Frontend đang chạy ở cổng 5173, backend ở 8000; `/health` xác nhận Turbo CUDA/FP16,
PhoBERT binary_v04 CPU và PostgreSQL sẵn sàng. Giao diện đã đọc lại 81 phiên cũ.

Chưa kiểm chứng bộ chọn chia sẻ tab thật của Chrome/Edge hoặc microphone vật lý
trong lượt này. Replay không thay thế việc thử một livestream liên tục 30 phút.
Chất lượng nhận dạng và các lỗi phân loại binary_v04 đã biết chưa được sửa bởi
việc bổ sung nguồn thu. Đây là hỗ trợ kiểm duyệt theo phạm vi dự án, chưa phải
kết luận vi phạm chính thức của TikTok.

Windows đã ghi cảnh báo thiếu bộ nhớ ảo trong các lần nạp model ban đầu khi Docker
cùng hoạt động. Lần khởi động sau đã thành công với cấu hình giữ nguyên; vấn đề
dung lượng bộ nhớ lúc khởi động chưa được giải quyết bằng thay đổi model.

Mở `http://127.0.0.1:5173/` bằng Chrome/Edge → **Tab livestream** → **Chọn tab
livestream** → chọn tab đang phát, bật **Chia sẻ âm thanh của tab**. Sau khi thử,
bấm **Kết thúc tab**, chờ hoàn tất rồi mở lịch sử và tải JSON. Kiểm tra lần lượt
microphone và tab; không chạy đồng thời hai nguồn.

Bằng chứng kỹ thuật: `runtime/tab-live-check/real_pipeline_results.json` và
`runtime/tab-live-check/production_migration.json`. Backup trước migration:
`backups/before_tab_live_20261024_151700.dump`. Không commit/push.
