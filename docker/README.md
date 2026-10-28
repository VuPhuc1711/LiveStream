# Môi trường phát triển Docker Compose

Ba dịch vụ: React/Vite `frontend` (5173), FastAPI/WebSocket `backend` (8000),
PostgreSQL `postgres` (cổng Windows5433, trong mạng Compose5432). Các cổng chỉ
bind127.0.0.1. Microphone/tab vẫn được chọn và cấp quyền trong Chrome/Edge trên
máy người dùng; container không trực tiếp mở microphone, không lấy hình tab.

Whisper giữ Turbo CUDA/FP16, LayerNorm FP32, gợi ý từ vựng/vi/transcribe và Silero
hiện tại. PhoBERT giữ binary_v04 CPU và hai nhãn. Không có train hoặc tải model
ngầm khi khởi động. Backend chỉ có một worker, không bật auto-reload model.

## Máy mới của người cùng phát triển

Yêu cầu Docker Desktop Linux containers/WSL2, NVIDIA GPU và driver hỗ trợ GPU
trong Docker. Cấu hình này dành cho Linux x86_64, chưa có chế độ CPU thay thế tự
động. Máy không có NVIDIA GPU cần thống nhất phương án chạy riêng trước.

Trên máy Linux, backend chạy UID10001, frontend UID1000. Thư mục runtime bind
mount phải cho UID10001 ghi (có thể cấp ACL đúng UID cho runtime và các thư mục
ghi bên trong); source frontend/volume dependency phải cho UID1000 ghi. Không
đổi quyền toàn repository hoặc dùng chmod777. Windows Docker Desktop quản lý
quyền bind mount theo cơ chế chia sẻ file của Windows.

1. Clone repository vào thư mục tùy chọn. Không copy venv/node_modules Windows.
2. Chỉ khi chưa có `.env`, sao chép `.env.example` thành `.env`, đặt mật khẩu
   PostgreSQL riêng. Giữ mật khẩu đang dùng nếu volume PostgreSQL đã tồn tại;
   sửa `.env` không tự đổi mật khẩu bên trong database cũ.
3. Nhận toàn bộ thư mục model dưới đây từ người đang giữ trọng số và cache Turbo.
   Giữ cấu trúc đường dẫn tương đối; không đổi byte của lock/tokenizer/trọng số:

```text
models/phobert_binary_v04_20261003_092559_043789/
  seed42/checkpoint_lock.json
  seed42/best/                         # đủ14 artifact, gồm5 shard safetensors
runtime/asr-models/large-v3-turbo.pt
```

PhoBERT là model đã fine-tune của dự án, cần nhận riêng; không tải PhoBERT-base
để thay vào. Turbo có URL chính thức và SHA-256 tại `config/runtime_assets.json`.
Nếu chưa có cache Turbo có thể tải đúng file từ URL đó. Không đưa trọng số vào
Git hoặc build context. `runtime/` là thư mục riêng của máy, chứa debug/CSV phát
sinh; không sao chép database thật chỉ để dựng môi trường phát triển.

`.gitattributes` giữ nguyên byte của CSV và metadata/tokenizer/lock model, tránh
Git đổi LF/CRLF làm sai hash khi clone Windows/Linux. Không dùng normalize hoặc
renormalize các artifact đã khóa. Các file shell Docker được giữ LF.

PowerShell tại thư mục repository:

```powershell
# Chỉ thực hiện Copy-Item nếu chưa có .env, sau đó sửa mật khẩu trong file.
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps backend python -m tools.check_container_assets
docker compose up -d --wait --wait-timeout 600
docker compose ps
```

Lần build đầu tải thư viện CUDA nên mất thời gian/dung lượng. Lệnh kiểm tra
assets chỉ đọc hash, không nạp model, không train hay nối database. Khi thiếu
thư mục mount, Compose báo lỗi thay vì tự tạo một thư mục model rỗng.

Cần dự trù khoảng15–20GB trống tại ổ chứa đĩa Docker cho lần dựng đầu và cache
build. Đây là dung lượng Docker, không phụ thuộc ổ chứa mã nguồn. Máy hiện tại
đã được người dùng cho phép chuyển đĩa Docker từ C sang D; vị trí này là cấu
hình riêng máy, không gắn cứng D vào Compose để máy khác phải theo.

Trên máy hiện tại, đĩa hoạt động ở D:\DockerDesktopData; backup VHDX đầy đủ ở
D:\lưu trữ\DockerDesktop_backup_20261028. Không copy VHDX lên Git. Tài liệu chính
thức về [GPU trên Windows/WSL2](https://docs.docker.com/desktop/features/gpu/) và
[vị trí lưu trữ WSL2](https://docs.docker.com/desktop/features/wsl/).

Mở http://127.0.0.1:5173/. Dùng trình duyệt cùng cửa sổ/profile với tab livestream
khi chia sẻ audio tab. Health http://127.0.0.1:8000/health phải báo ready, database
postgresql, binary_v04 và Whisper turbo/cuda/fp16. Không coi model chỉ nạp được
là bằng chứng chất lượng nhận dạng hay phân loại đã đạt.

## Làm việc hàng ngày

```powershell
docker compose up -d --wait --wait-timeout 600
docker compose logs --tail 80 backend frontend
docker compose restart backend
# Sau khi đổi Dockerfile hoặc dependency:
docker compose up -d --build --wait --wait-timeout 600
# Dừng phiên giám sát trên giao diện trước, rồi dừng các dịch vụ:
docker compose stop
```

Frontend bind source và dùng volume Linux riêng cho node_modules. Khi lockfile
đổi, entrypoint chạy lại npm ci; Windows node_modules không được dùng trong
container. Backend bind các thư mục source; cần restart sau sửa Python. Đổi
entrypoint/Dockerfile/dependency thì build lại. Checkpoint và cache ASR được
mount chỉ đọc; debug và CSV mới được ghi vào `runtime/` trên máy.

Giữ tên Compose `livestream-detection` và volume `postgres_data`: volume thật
là `livestream-detection_postgres_data`. Đổi project name sẽ tạo một database
khác và có thể khiến lịch sử trông như bị mất. Không xóa volume để reset app.
Git chỉ chia sẻ code/cấu hình/migration; mỗi máy có database local riêng.

`DATABASE_URL` trong `.env` dành cho chạy native Windows. Backend container tạo
URL riêng từ POSTGRES_* với host postgres:5432, escape đúng mật khẩu có ký tự
đặc biệt; không ghi URL có mật khẩu vào log/health. Không chạy `docker compose
config` rồi chia sẻ toàn bộ đầu ra vì nó có thể chứa giá trị môi trường đã giải.

Các lệnh Python bảo trì/migration dùng `docker compose run --rm --no-deps backend
...` để đi qua entrypoint dựng URL đúng. Không gọi trực tiếp Alembic bằng exec
trong container rồi dựa vào DATABASE_URL của phiên shell khác.

## Kiểm tra

```powershell
docker compose run --rm --no-deps backend python -m unittest discover -s tests -t .
docker compose run --rm --no-deps -e RUN_POSTGRES_TESTS=1 backend python -m unittest tests.backend.test_postgres
docker compose run --rm --no-deps frontend npm test
docker compose run --rm --no-deps frontend npm run build
```

Test ứng dụng dùng database tạm. Test PostgreSQL opt-in tạo/xóa các database
livestream_test_* của chính test, không xóa database sử dụng thật. Không chạy
script đánh giá/train lịch sử trong đợt đóng gói này.

Backend Linux có lockfile80 dependency kèm hash, giữ phiên bản thư viện model
và preprocessing như môi trường native. Silero6.2.1 cài riêng --no-deps và nạp
JIT đóng gói trực tiếp, không dùng torchaudio. Vì metadata Silero vẫn khai báo
torchaudio, pip check sau bước đó có thể báo dependency khai báo thiếu; Docker
chỉ chạy pip check trước khi cài Silero và kiểm tra resource/JIT riêng. WebRTC
lịch sử không nằm trong môi trường Docker; chẩn đoán hiện hành dùng Silero.

## Sao lưu và quay lại cách chạy native

Sao lưu nhất quán, không redirect file nhị phân qua PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path backups | Out-Null
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f /tmp/livestream_backup.dump'
docker compose cp postgres:/tmp/livestream_backup.dump backups/livestream_backup.dump
```

Chọn tên bản sao lưu mới mỗi lần để không ghi đè bản trước. Bản dump chỉ chứa
database; muốn sao lưu cả audio debug/CSV thì sao chép runtime riêng. Cách khôi
phục PostgreSQL có trong README gốc; ưu tiên khôi phục vào database riêng để
kiểm tra trước khi thay dữ liệu hiện tại.

Quay lại backend/frontend Windows, giữ cùng database PostgreSQL:

```powershell
docker compose stop backend frontend
docker compose up -d --wait postgres
.\.venv-nlp\Scripts\python.exe -X utf8 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --ws-max-size 65536 --ws-max-queue 8
# Terminal khác:
Set-Location frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Không chạy hai backend trên cùng dữ liệu khi đang có phiên giám sát. Dừng phiên
trên giao diện trước để xử lý phần cuối. Việc đổi cách chạy không nâng chất lượng
model và không đồng bộ các mẫu bổ sung giữa hai máy.
