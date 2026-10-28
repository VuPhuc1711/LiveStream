# Hướng dẫn đọc và trình bày dự án

Tài liệu mô tả ứng dụng hiện tại, cập nhật ngày **24/10/2026**. Bắt đầu từ
[README](../README.md) để chạy dự án; đọc tài liệu này để hiểu từng phần trước
khi xem mã nguồn hoặc trình bày trước hội đồng.

Cập nhật bố cục ngày28/10/2026: dữ liệu hiện chỉ có `data/current/train.csv`
(584 câu) và `test.csv` (182 câu validation đã xem trước đây, chỉ là hồi quy).
Chưa train lại; checkpoint binary_v04 không đổi. Dataset/checkpoint và tài liệu
cũ ở `D:\lưu trữ\livestream-violation-detection_20261028` ngoài repository;
`config/storage.json` chỉ vị trí, `config/data/current_dataset.json` ghi hash/nguồn.

## 1. Dự án giải quyết việc gì?

Ứng dụng hỗ trợ người kiểm duyệt theo dõi **lời nói trong livestream bán hàng**.
Người dùng chọn nguồn âm thanh, xem transcript và những câu được mô hình đánh
dấu cần chú ý, sau đó có thể xác nhận hoặc sửa kết quả. Ứng dụng lưu lịch sử
để xem lại, xuất báo cáo và tích lũy dữ liệu đã được người dùng duyệt.

Đây là bản thử nghiệm theo **quy ước của dự án**. Ứng dụng chưa đánh giá đầy đủ
mọi chính sách TikTok, hình ảnh, sản phẩm, bình luận hoặc ngữ cảnh của toàn bộ
phiên live. Kết quả không xác nhận TikTok sẽ xử phạt, hạn chế đề xuất hay làm
giảm doanh thu của kênh.

## 2. Các chức năng trên giao diện

| Chức năng | Nguồn và cách xử lý |
| --- | --- |
| **Giám sát microphone** | Nhận âm thanh từ microphone; gửi dần và trả kết quả khi phiên vẫn đang thu. |
| **Giám sát tab livestream** | Người dùng chọn tab Chrome/Edge đang phát tiếng; nhận âm thanh của tab, không lấy microphone. |
| **Chọn bản ghi / Thu âm trực tiếp** | Tải file có sẵn, hoặc thu xong và nghe lại rồi gửi cả bản thu để phân tích. |
| **Lịch sử phân tích** | Xem phiên đã lưu, transcript, dự đoán gốc, phản hồi và tải báo cáo JSON. |
| **Dữ liệu bổ sung** | Quản lý mẫu được sửa/duyệt, lịch sử phiên bản, thống kê chưa dùng train và xuất CSV. |

Chỉ một công việc phân tích hoạt động tại một thời điểm. Giám sát là xử lý gần
thời gian thực theo các cửa sổ audio, chưa phải nhận dạng từng từ ngay khi nói.

## 3. Luồng xử lý hiện tại

```text
Microphone hoặc âm thanh tab
        ↓
Trình duyệt: mono PCM16, 16 kHz → WebSocket
        ↓
Silero VAD: xác định vùng có tiếng nói, giữ đệm đầu/cuối
        ↓
Whisper Turbo: audio → văn bản tiếng Việt
        ↓
Kiểm tra kết quả ASR → ghép ngữ cảnh/tách đơn vị câu
        ↓
PhoBERT binary_v04: văn bản → một trong hai nhãn
        ↓
PostgreSQL: lưu phiên, đoạn, dự đoán gốc và trace
        ↓
Frontend: transcript, cảnh báo, lịch sử và phản hồi
```

Luồng upload cũng qua VAD và cùng model ASR, nhưng dùng cửa sổ audio dài hơn.
Frontend và backend chạy trực tiếp trên Windows; Docker Compose chạy PostgreSQL.

| Thành phần | Trách nhiệm chính | Mã nguồn cần đọc |
| --- | --- | --- |
| Frontend React | Giao diện, quyền thu/chia sẻ, gửi audio, hiển thị kết quả | `frontend/src/App.tsx`, `LiveMonitoring.tsx`, `liveCapture.ts` |
| Backend FastAPI | API, vòng đời phiên, điều phối suy luận và lưu kết quả | `backend/app.py`, `live_audio.py`, `pipeline_service.py` |
| VAD và chia audio | Xác nhận tiếng nói, đệm và giới hạn cửa sổ | `backend/neural_vad.py`, `speech_buffer.py`, `upload_audio.py` |
| Nhận dạng tiếng nói | Nạp Whisper một lần, cấu hình và giải mã | `speech/transcribe.py`, `asr_config.py` |
| Phân loại văn bản | Tiền xử lý và suy luận PhoBERT | `nlp/model_only_detector.py`, `phobert_detector.py`, `label_mapping.py` |
| Lưu trữ | Phiên, feedback, mẫu bổ sung, phiên bản và migration | `backend/database.py`, `history.py`, `review_data.py`, `migrations/` |

Whisper hiện dùng **Turbo CUDA/FP16**, LayerNorm FP32, tiếng Việt và tác vụ
`transcribe`, với gợi ý từ vựng ngắn. PhoBERT chạy **binary_v04 trên CPU**.
Luồng phân loại hiện tại chỉ dùng PhoBERT; không dùng Rules hoặc ngưỡng Hybrid.
Trang `/health` của backend phản ánh cấu hình thực sự đã nạp sau khởi động.

## 4. Hiểu hai nhãn và phản hồi

| Nhãn | Ý nghĩa trong ứng dụng |
| --- | --- |
| `KHONG_CANH_BAO` | Mô hình chưa đánh dấu câu cần cảnh báo; không có nghĩa nội dung đã được xác thực hoặc chắc chắn phù hợp mọi chính sách. |
| `CANH_BAO` | Mô hình đề nghị người kiểm duyệt xem lại câu theo phạm vi dự án. |

Ví dụ quy ước: khẳng định “rẻ nhất thị trường” hoặc giới thiệu/mời khách sang
kênh mua hàng ngoài TikTok thuộc nhóm cảnh báo; so sánh giới hạn rõ trong danh
mục của shop khác với so sánh toàn thị trường. Phải xét toàn câu: phủ định,
câu hỏi thật, bác bỏ và tán thành lời trích dẫn có ý nghĩa khác nhau. Xem
[quy ước gán nhãn](../BINARY_LABELING_GUIDE.md) để đọc đầy đủ.

**Quy ước mong muốn và dự đoán của model là hai việc riêng.** Binary_v04 còn
bỏ sót một số cách điều hướng Zalo và có thể sai khi gặp cách nói mới. Số phần
trăm bên cạnh nhãn là **xác suất mô hình**, không phải độ chính xác hệ thống.

Transcript và dự đoán gốc được giữ riêng với feedback. Khi người dùng sửa chữ,
sửa nhãn rồi chọn **Duyệt đưa vào dữ liệu**, phiên bản đã duyệt nằm trong kho
dữ liệu bổ sung PostgreSQL. Xuất CSV chỉ tạo một bản dữ liệu có phiên bản/hash;
không thay model, không tự ghép vào train/validation và không đánh dấu đã train.
Mẫu AI vẫn giữ nguồn `AI_GENERATED`. Sửa mẫu đã duyệt cần duyệt lại phiên bản mới.

## 5. Cách demo dễ theo dõi

1. Chạy database, backend và frontend theo README. Mở
   `http://127.0.0.1:5173/`; kiểm tra **Backend đã sẵn sàng** và
   **PhoBERT binary_v04**. Có thể mở `http://127.0.0.1:8000/health` để đối chiếu.
2. Với live thực tế, mở ứng dụng và livestream trong **cùng Chrome/Edge**.
   Chọn **Tab livestream → Chọn tab livestream → Tab Chrome / Microsoft Edge**,
   chọn tab phát live và bật **Chia sẻ âm thanh của tab**. Không chọn cửa sổ
   hoặc toàn màn hình. Giữ tab giám sát mở trong lúc theo dõi.
3. Cho người xem quan sát transcript chốt dần, các câu cảnh báo và số audio
   đang gom/chờ/xử lý. Bật **Hiện tất cả câu giám sát** để xem cả nhãn không
   cảnh báo. Khi đối chiếu kết quả, kiểm tra transcript có giữ đúng lời nói.
4. Bấm **Kết thúc tab** hoặc **Dừng chia sẻ**, rồi chờ phiên hoàn tất và hết
   audio chưa xử lý. Mở lịch sử để xem lại phiên và tải JSON.
5. Chọn một câu cần sửa, nhập văn bản/nhãn và lý do; lưu feedback. Nếu muốn
   thu thập cho đợt nâng cấp sau, dùng thao tác duyệt rõ ràng rồi xem ở
   **Dữ liệu bổ sung**. Phân biệt bước này với huấn luyện mô hình.

## 6. Giới hạn cần nói rõ khi báo cáo

- Whisper có thể nhận sai từ, mất ý phủ định/phạm vi hoặc tự thêm chữ. Bản sửa
  hiện tại thử lại khi có timestamp bất khả thi; phần vẫn lỗi được báo **chưa
  nhận dạng đáng tin cậy** và không phân loại. Timestamp hợp lệ chưa chứng minh
  transcript đúng; không có cam kết loại hết mọi phụ đề giả.
- Chia cửa sổ và dấu câu có thể ảnh hưởng ngữ cảnh. Tồn đọng/độ trễ phụ thuộc
  độ dài lời nói, phần cứng và số lần ASR phải thử lại.
- Lịch sử lưu văn bản và kết quả, không có audio nghe lại mặc định. Muốn điều
  tra lỗi phải chủ động bật **Lưu audio debug cho phiên thử** trước khi bắt đầu;
  WAV debug và kết quả thô là bằng chứng chẩn đoán, không tự thành dữ liệu train.
- Kết quả validation đã dùng phát triển không phải độ chính xác trên mọi live
  thực tế. Demo thành công một phiên không thay thế đánh giá độc lập.
- Chưa có đăng nhập/phân quyền, chưa triển khai công khai hoặc extension Store.
  Nâng cấp model đang tạm dừng ở binary_v04; dữ liệu bổ sung được tích lũy riêng.

Các mốc và lỗi đã biết nằm trong [PROJECT_STATUS](../PROJECT_STATUS.md). Các
báo cáo thử nghiệm cũ phục vụ truy vết, không mô tả cấu hình đang chạy hiện tại.
