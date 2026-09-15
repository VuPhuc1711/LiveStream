# Livestream Violation Detection

Project scaffold for livestream speech transcription and violation detection.

## Structure

- `speech/`: audio transcription utilities
- `audio/`: sample audio files
- `nlp/`: natural language processing logic
- `models/`: trained or downloaded model files
- `data/`: datasets and processed data
- `main.py`: application entry point

## Chạy demo trên file âm thanh

Demo chạy toàn bộ file qua **Whisper small một lần**, lấy các segment `start`, `end`, `text`,
rồi tách mỗi segment thành các câu theo dấu `.`, `?`, `!` để tạo **classification unit**.
HybridDetector dự đoán từng unit riêng, không dùng một dự đoán chung cho hai câu có dấu phân cách.
Whisper và PhoBERT mỗi model chỉ
nạp một lần. Whisper được giải phóng trước khi nạp PhoBERT để giảm mức dùng VRAM.

Hybrid trong demo bắt buộc dùng `models/phobert_v02_20260914_162150_178711/best`.
Chương trình đối chiếu SHA-256 với bản khóa checkpoint, in đường dẫn thực tế sau khi nạp
và lưu đường dẫn đó vào JSON. Các API detector cũ vẫn giữ mặc định riêng; demo truyền
checkpoint v02 tường minh, không dùng mặc định của chúng.

Chạy bằng PowerShell, không cần activate môi trường:

```powershell
Set-Location 'D:\livestream-violation-detection'
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\main.py '.\audio\voicetiengviet.mp3'
```

Đổi đối số cuối thành đường dẫn audio khác, đặt trong dấu nháy nếu có khoảng trắng.
Đường dẫn audio tương đối được tính từ thư mục PowerShell hiện tại; checkpoint và thư mục
reports luôn được tính từ vị trí `main.py`.

Kiểm tra môi trường riêng, không chạy audio hay nạp model:

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\main.py --check-environment
ffmpeg -version
```

Mặc định `--device auto` chọn CUDA nếu khả dụng. Có thể dùng `--device cpu` khi không
muốn dùng GPU hoặc thiếu VRAM; CPU có thể chạy chậm hơn:

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\main.py '.\audio\voicetiengviet.mp3' --device cpu
```

### Môi trường đã kiểm tra

Python 3.11 trong `.venv-nlp`, PyTorch `2.13.0+cu126` (CUDA 12.6), Transformers `5.16.1`,
Underthesea `9.5.0`, `openai-whisper==20250625`, FFmpeg `9.0.1` trong PATH.
Đợt tích hợp chỉ bổ sung `openai-whisper==20250625`, `numba==0.67.0`, `llvmlite==0.49.0`,
`more-itertools==11.1.0`, `tiktoken==0.14.0`; giữ nguyên PyTorch CUDA và các gói có sẵn.

Nếu thiếu gói, `--check-environment` sẽ báo tên cụ thể. Không chạy nâng cấp toàn bộ môi
trường hoặc cài lại PyTorch khi CUDA đang hoạt động. Trên một bản sao của đúng môi trường
này, có thể bổ sung các gói vừa nêu bằng lệnh sau (các dependency nền phải có sẵn):

```powershell
.\.venv-nlp\Scripts\python.exe -m pip install --no-deps openai-whisper==20250625 numba==0.67.0 llvmlite==0.49.0 more-itertools==11.1.0 tiktoken==0.14.0
.\.venv-nlp\Scripts\python.exe -m pip check
```

Whisper dùng cache mặc định `%USERPROFILE%\.cache\whisper` (hoặc theo `XDG_CACHE_HOME`).
Máy hiện có `small.pt`; nếu cache chưa có, Whisper cần tải trọng số trong lần chạy đầu.
PhoBERT chỉ nạp từ checkpoint cục bộ. FFmpeg là chương trình riêng, phải chạy được
`ffmpeg -version` trong cùng cửa sổ PowerShell.

### Kết quả demo

Mỗi lần chạy tạo thư mục mới `reports/audio_demo_<timestamp>/`:

- `whisper.json`: toàn bộ kết quả Whisper, lưu ngay sau bước nhận dạng.
- `result.json`: thời gian chạy UTC, đường dẫn/hash audio và checkpoint, môi trường,
  toàn văn transcript, raw segment, kết quả từng classification unit và thống kê nhãn cuối. Nội dung cũng được in ra terminal.
- `error.json`: chỉ xuất hiện nếu quá trình sau khi tạo thư mục bị lỗi; không ghi đè báo cáo cũ.

`result.json` phiên bản 2 có:

- `whisper_segments`: giữ nguyên toàn bộ segment và metadata từ Whisper để truy vết.
- `classification_units`: mỗi câu có `unit_id`, `source_segment_id`, `source_segment_index`,
  `text`, `start`, `end`, `timestamp_is_approximate` và các kết quả dưới đây.
- `segments`: alias tương thích của `classification_units`, vẫn có `id/start/end/text/status`
  và kết quả theo cấu trúc cũ. `summary.total_segments` là độ dài alias này;
  dùng `summary.whisper_segment_count` và `summary.classification_unit_count` để phân biệt hai số lượng.

Mỗi unit giữ nguyên nội dung/dấu câu của phần tương ứng trong raw text, chỉ bỏ khoảng trắng
ở hai đầu. Không tạo unit cho đoạn rỗng hoặc chỉ có dấu câu. Phần cuối chưa có dấu kết thúc
vẫn được giữ thành một unit. Không dùng nhãn/câu tham chiếu để tách và không ép số unit bằng 9.
Nếu Whisper bỏ hẳn dấu phân cách giữa hai câu, cách tách theo dấu câu không tự suy ra ranh giới đó.

`start/end` tính bằng giây từ đầu file. Khi một segment chứa nhiều câu, khoảng thời gian được
chia theo tỷ lệ độ dài ký tự; tất cả unit trong segment đó có `timestamp_is_approximate=true`.
Đây không phải word timestamp. `false` chỉ có nghĩa dùng nguyên khoảng segment Whisper,
không xác nhận mốc ASR đã chính xác theo bản thu.

- `phobert_result.label/confidence/probabilities`: nhãn và xác suất softmax của PhoBERT.
- `rule_result.needs_review/findings`: Rules có khớp hay không, các cụm từ/câu và nhóm khớp.
- `final_label`: quyết định của Hybrid theo ngưỡng đã có: `CANH_BAO` ≥ 0.55,
  `KHONG_CANH_BAO` ≥ 0.65; các trường hợp còn lại chuyển `CAN_XAC_MINH`.
- `final_confidence`: xác suất **PhoBERT dành cho nhãn cuối**, có thể khác confidence
  của nhãn PhoBERT ban đầu; không phải độ tin cậy tổng hợp hoặc confidence của Whisper.
- `explanation`: giải thích bằng tiếng Việt của Hybrid.

Demo không có nhãn thật để tính Accuracy/F1. Các quyết định phụ thuộc văn bản ASR và
ranh giới segment; cần xem transcript khi đối chiếu với audio. Đây là xử lý file âm thanh,
chưa có giao diện web hoặc thu/phân tích livestream trực tiếp.

Lần chạy thử trước khi đổi phân đoạn ngày 14/09/2026 trên `audio/voicetiengviet.mp3` (khoảng 164 giây) hoàn tất
trong 28.13 giây trên RTX 3050 Ti Laptop: 32 segment, 9 `KHONG_CANH_BAO`,
23 `CAN_XAC_MINH`, 0 `CANH_BAO`. Kết quả ở
`reports/audio_demo_20260914_231336_035723/result.json`.
Transcript nhận được là bản tin truy nã, khác nội dung bán hàng dùng để xây dựng model;
thống kê trên chỉ mô tả đầu ra demo, không xác nhận nhãn đúng/sai của các đoạn.

Kiểm tra luồng bằng dữ liệu giả, không dùng dataset/final test hoặc nạp trọng số model:

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 -m unittest test_audio_demo -v
```

Chạy bản thu có nhãn hiện có và xem [hướng dẫn đánh giá](AUDIO_TEST_GUIDE.md):

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\main.py '.\audio\audio_test_9_sentences.m4a'
```

`evaluate_audio_pipeline.py` ưu tiên `classification_units` khi có; báo cáo cũ vẫn được
## Model checkpoint

Trọng số PhoBERT không được lưu trên GitHub do dung lượng lớn.

Đặt checkpoint tại:

models/phobert_v03_20260915_145458_712323/best
chấm theo các segment/dự đoán cũ để so sánh. `whisper.json` chưa có nhãn cũng được tách câu
bằng cùng thuật toán trước khi phân loại.

Phần demo không train, chỉnh Rules/ngưỡng, sửa dữ liệu hay checkpoint. Các báo cáo và
khóa đánh giá final test v03 đã có được giữ nguyên; không chạy lại đánh giá cuối.
