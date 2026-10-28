# Livestream Violation Detection

Ứng dụng hỗ trợ kiểm duyệt lời nói trong livestream bán hàng: nhận âm thanh từ
**microphone hoặc tab Chrome/Edge**, hiển thị transcript, đánh dấu câu cần chú ý,
lưu lịch sử và thu thập phản hồi. Đây là bản thử nghiệm theo quy ước của dự án.

**Đang dùng:** Whisper Turbo CUDA/FP16 + Silero VAD → PhoBERT **binary_v04** CPU
→ PostgreSQL Docker. Hai nhãn: `KHONG_CANH_BAO`, `CANH_BAO`. Không chạy Rules/Hybrid.
Nâng cấp model đang tạm dừng; xuất/duyệt CSV không tự train.

**Dữ liệu hiện quản lý theo hai phần:** [train.csv](data/current/train.csv)
**584 câu**, [test.csv](data/current/test.csv) **182 câu**. Test là toàn bộ validation
cũ đã dùng chọn checkpoint v04, nên đây là bộ hồi quy đã xem, **không phải test
độc lập**. Giữ nguyên nhãn/ID/family và membership; chưa train lại. Hồ sơ nguồn/hash
ở [current_dataset.json](config/data/current_dataset.json). Metadata checkpoint cũ
vẫn giữ đúng lịch sử train/validation, không đổi tên để làm sai bằng chứng.

Ngày28/10/2026, dữ liệu/checkpoint/công cụ lịch sử đã được chuyển ra ngoài dự án:
**`D:\lưu trữ\livestream-violation-detection_20261028`**. Vị trí và manifest ở
[bản mẫu cấu hình lưu trữ](config/storage.example.json); `config/storage.json`
thật là cấu hình local được Git bỏ qua. Giữ nguyên model đang chọn, cache
Whisper, database, feedback và audio debug trong dự án/volume Docker.

Trên Windows CUDA/FP16, backend chọn nhánh classic cuBLAS cho biased `addmm`
(`DISABLE_ADDMM_CUDA_LT=1` trong riêng tiến trình) để tránh lỗi heuristic cuBLASLt
đã ghi nhận. Không đổi model/precision. Sau khi cả hai model đã nạp, startup chạy
phép chiếu encoder thật trong worker; lỗi thì không báo ready. Health hiển thị
`whisper_config.cuda_runtime`, `runtime_self_check` và `asr_runtime`.
Muốn đối chiếu nhánh gốc phải đặt `$env:DISABLE_ADDMM_CUDA_LT='0'` **trước khi
khởi động tiến trình mới**; không đổi giữa phiên vì PyTorch cache cờ này.
Chi tiết: [bản sửa CUDA](docs/BAO_CAO_SUA_CUDA_20261024.md).

## Đọc dự án từ đâu?

1. [Hướng dẫn đọc và trình bày](docs/HUONG_DAN_DU_AN.md): chức năng, kiến trúc, luồng xử lý.
2. [Quy ước hai nhãn](BINARY_LABELING_GUIDE.md): phạm vi cảnh báo và ví dụ.
3. [Trạng thái hiện tại](PROJECT_STATUS.md): phiên bản, kết quả kiểm tra và hạn chế.
4. Đọc `backend/app.py` và `frontend/src/App.tsx` để đi vào mã nguồn.

## Bố cục

```text
backend/       API, realtime, database, lịch sử và phản hồi
frontend/      giao diện React/Vite, thu microphone và audio tab
speech/        Whisper, cấu hình ASR và đơn vị câu
nlp/           PhoBERT, mapping và dự đoán hai nhãn
config/        manifest dữ liệu hiện hành và vị trí lưu trữ bên ngoài
tests/         test ứng dụng hiện tại; test giao diện ở frontend/src/
tools/         công cụ bảo trì/chẩn đoán, không tự train
docs/          hướng dẫn đọc dự án và báo cáo ngắn
data/current/  chỉ hai CSV: train.csv và test.csv
models/        chỉ checkpoint binary_v04 đang chọn, giữ nguyên đường dẫn
runtime/       audio debug, CSV xuất và dữ liệu phát sinh local
backups/       bản sao lưu database/cấu hình riêng tư
```

Hai file cấu hình **đang có hiệu lực** ở gốc: `asr_config.json` và
`pipeline_model.json`. `compose.yaml` quản lý PostgreSQL; `alembic.ini` quản lý
migration. `.env` chứa thông tin kết nối riêng tư, không đưa vào Git.

## Phạm vi đưa lên Git

Giữ mã ứng dụng, migrations, tests, tài liệu, hai CSV hiện hành, manifest/checksum,
`pipeline_model.json`, `asr_config.json`, các file dependency và lockfile frontend.
Chia sẻ `.env.example`, `frontend/.env.example` và `config/storage.example.json`
để mỗi người tạo cấu hình local khi cần.

`.env`, `config/storage.json`, môi trường Python, node_modules, runtime/debug,
backups/database và trọng số lớn được Git bỏ qua. Không cần xóa những file này
khỏi máy để chia sẻ source. Lịch sử/feedback trong PostgreSQL không tự đồng bộ
qua Git. Kho lưu trữ ở ổ D là riêng của máy hiện tại, không phải điều kiện chạy app.

Môi trường phát triển Docker dùng dependency Linux được khóa riêng. Checkpoint
được xác minh bằng định danh tương đối và đầy đủ SHA-256, hỗ trợ clone khác thư
mục/Windows/Linux mà không sửa bản khóa gốc. Máy mới vẫn phải nhận model riêng
và có GPU tương thích; xem [hướng dẫn Docker](docker/README.md).

`main.py` và `model_config.py` gốc đã chuyển vào kho ngoài dự án, tại
`archive/legacy_cli_20261028/`. Web dùng `backend/model_runtime.py` và
`backend/model_config.py`; đây là các file hiện hành cần giữ. Hai file CLI cũ
được bảo toàn byte/hash trong `manifest_legacy_cli_20261028.json` ở root lưu trữ.
Muốn chạy lại công cụ lịch sử phải khôi phục đường dẫn gốc trước, không chạy
trực tiếp bản lưu trữ hoặc dùng CLI Whisper-small cũ để demo ứng dụng hiện tại.

Dataset/checkpoint/report, cấu hình train/rollback và script nghiên cứu cũ giữ
nguyên byte trong thư mục lưu trữ bên ngoài. **754 file/8,60GB**, có manifest hash
và đường dẫn gốc để khôi phục. Đợt tiếp theo lưu thêm hai file CLI cũ, tổng756
file; manifest đợt754 file trước giữ nguyên. Thư mục `outputs` có một junction thư viện: chỉ
đổi tên cùng ổ D, không truy cập/di chuyển nội dung thư viện được liên kết.
Không tự chạy lại script train hoặc sửa lock/hash lịch sử. Checkpoint cũ cần
khôi phục về đường dẫn gốc trước khi rollback; không sửa lock để ép nạp.

## Chạy bằng Docker Compose

Compose gồm `frontend`, `backend` và `postgres`. Giữ Whisper Turbo CUDA/FP16,
PhoBERT binary_v04 CPU và volume `livestream-detection_postgres_data` hiện có.
Trình duyệt trên Windows vẫn là nơi xin quyền microphone/chia sẻ tab.

Máy hiện tại đã có `.env` và model, chạy từ gốc repository:

```powershell
Set-Location 'D:\livestream-violation-detection'
docker compose up -d --build --wait --wait-timeout 600
docker compose ps
```

Mở http://127.0.0.1:5173/; health tại http://127.0.0.1:8000/health.
Không chạy đồng thời backend/frontend native trên cùng cổng hoặc thêm nhiều
worker model. Mã nguồn được bind mount; sửa Python thì `docker compose restart
backend`, frontend tự tải lại. Đổi dependency/Dockerfile thì build lại dịch vụ đó.

Máy bạn cùng phát triển: clone source, tạo `.env` từ bản mẫu, nhận hai bộ trọng số
riêng và kiểm tra hash trước khi chạy. Làm theo [Docker README](docker/README.md)
về GPU, assets, test, sao lưu và quay lại cách chạy Windows. Git không đồng bộ
database/feedback giữa hai máy; xuất CSV không tự đánh dấu đã train.

## Chạy trực tiếp trên Windows (phương án quay lại)

Yêu cầu môi trường `.venv-nlp` đang có của dự án, FFmpeg trong PATH, Node/npm,
Docker Desktop đang chạy và file `.env` được cấu hình theo `.env.example`.
Không cần extension hoặc đưa lên Chrome Web Store để demo local.

PowerShell thứ nhất, từ gốc repository:

```powershell
Set-Location 'D:\livestream-violation-detection'
docker compose stop backend frontend
docker compose up -d --wait postgres
.\.venv-nlp\Scripts\python.exe -m alembic upgrade head
.\.venv-nlp\Scripts\python.exe -X utf8 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --ws-max-size 65536 --ws-max-queue 8
```

PowerShell thứ hai:

```powershell
Set-Location 'D:\livestream-violation-detection\frontend'
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Nếu dependency đã có thì bỏ bước `npm.cmd install`. Chỉ chạy một backend, không
bật `--reload`/nhiều worker khi demo model. Cài các dependency backend còn thiếu
bằng `requirements.txt`; Silero dùng `requirements-vad.txt` với `--no-deps` để
không thay bộ PyTorch CUDA đang hoạt động. Không cần cài lại môi trường khi demo.

- **Frontend:** http://127.0.0.1:5173/
- **Kiểm tra backend/model/database:** http://127.0.0.1:8000/health
- **API:** http://127.0.0.1:8000/docs

## Demo một tab livestream

1. Mở livestream có tiếng trong Chrome/Edge.
2. Mở frontend trong tab khác của chính trình duyệt đó.
3. Chọn **Tab livestream → Chọn tab livestream → Tab Chrome/Edge**, chọn tab live
   và bật **Chia sẻ âm thanh của tab**. Không chọn cửa sổ/toàn màn hình.
4. Theo dõi transcript và câu cảnh báo. Để kiểm tra microphone, dùng mục riêng
   **Microphone**. Upload/thu xong thành file vẫn ở mục phân tích bản ghi.
5. Bấm kết thúc, đợi xử lý phần cuối rồi mở lịch sử/xuất JSON hoặc sửa phản hồi.

Chỉ audio được gửi tới backend; không gửi hình ảnh/URL/tiêu đề tab. Audio lịch sử
không được lưu để nghe lại. Có thể bật **Lưu audio debug cho phiên thử** trước
khi bắt đầu; file ở `runtime/live-debug/<session_id>/`, mặc định tắt.

Whisper có thể nhận sai. Đoạn ASR chưa nhận dạng đáng tin cậy được hiển thị riêng,
không coi là nhãn an toàn. Xem [báo cáo ASR](docs/BAO_CAO_SUA_ASR_20261024.md).
PhoBERT cũng còn lỗi, đặc biệt một số lời mời đặt qua Zalo. Kết quả hỗ trợ rà soát,
chưa bao phủ mọi chính sách TikTok hoặc chứng minh tác động lên kênh/doanh thu.

## Database và dữ liệu bổ sung

PostgreSQL chạy riêng trong Docker, cổng local `127.0.0.1:5433`; backend/frontend
chạy trực tiếp trên Windows. `.env` quy định database và tài khoản. Migration ở
`backend/migrations/`. Không xóa volume khi dừng/chạy lại.

Phản hồi tách khỏi transcript/dự đoán gốc. Mẫu được **Duyệt đưa vào dữ liệu** lưu
trong PostgreSQL; xuất CSV tạo snapshot/hash và bản sao ở `runtime/data-exports/`.
Xuất file không đánh dấu đã dùng train. Chưa tự ghép train/validation hoặc train
binary_v05. Chi tiết sao lưu/khôi phục đã có trong
tài liệu đầy đủ tại `archive/docs/README_before_cleanup_20261024.md` dưới thư mục
lưu trữ ngoài dự án (xem `config/storage.json`).

## Kiểm tra kỹ thuật

```powershell
Set-Location 'D:\livestream-violation-detection'
.\.venv-nlp\Scripts\python.exe -m unittest discover -s tests -t .
Set-Location .\frontend
npm.cmd test -- --run
npm.cmd run build
```

Test backend dùng database tạm. PostgreSQL integration chỉ chạy khi đặt
`$env:RUN_POSTGRES_TESTS='1'`; các database test có tên riêng, không xóa dữ liệu thật.
Các test/benchmark ba nhãn và script cũ nằm trong archive, không nằm trong bộ test
ứng dụng hiện hành; chúng đã ở thư mục lưu trữ ngoài dự án.
Giữ mọi biến thể cùng family một split khi có yêu cầu train sau này; không dùng
`test.csv` để chọn epoch hoặc điều chỉnh mô hình.

Kiểm tra kho lịch sử là **tùy chọn**, chỉ dành cho người đã nhận kho lưu trữ và
cấu hình `config/storage.json` theo [hướng dẫn cấu hình](config/README.md):

```powershell
.\.venv-nlp\Scripts\python.exe -m tools.verify_archive
# Tùy chọn đọc thêm khoảng8,6GB để kiểm tra toàn bộ754 file đã chuyển:
.\.venv-nlp\Scripts\python.exe -m tools.verify_archive --all
```
