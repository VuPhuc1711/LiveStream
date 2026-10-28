# Trạng thái dự án — 28/10/2026

## Chia sẻ mã nguồn lên Git

Người dùng đã cho phép commit/push toàn bộ bản ứng dụng đã kiểm chứng và bỏ
file cũ khỏi cây Git hiện tại. Giữ lịch sử Git, không force-push hoặc xóa bản
lưu ngoài dự án. 116 file cũ có bản lưu khớp hash; test backend còn lại đã chuyển
vào tests/backend, giữ đủ test cũ. Source hiện hành khoảng3,1MB; các trọng số,
runtime/debug/database/backups và thông tin đăng nhập không nằm trong bản chia sẻ.
Máy cộng tác cần model riêng, GPU phù hợp và .env local theo docker/README.md.
Quyền commit/push của yêu cầu này thay các ghi chú không commit/push ở mốc trước;
không cấp quyền train, sửa nhãn hoặc thay checkpoint.

## Đóng gói Docker Compose — đã kiểm chứng

Người dùng đã yêu cầu triển khai cả frontend/backend/PostgreSQL bằng Compose.
Đã bổ sung Dockerfile, Linux lockfile, entrypoint, source bind mount, cache/model
chỉ đọc và kiểm tra assets/hash; giữ Turbo CUDA/FP16 và binary_v04 CPU. Validator
hỗ trợ chuyển repository Windows/Linux bằng định danh model tương đối, không
sửa checkpoint_lock hoặc SHA artifact. Metadata CUDA báo đúng cờ explicit trên
Linux, còn chính sách tự đặt mặc định Windows không đổi.

Frontend container61 test và build đạt; backend Linux158 test,157 đạt/1 kiểm tra
kho lịch sử ngoài bỏ qua, gồm32 test PostgreSQL thật trên database tạm. Test
native158,126 đạt/32 PostgreSQL opt-in bỏ qua. GPU nhận RTX3050Ti4GB. Đã sao lưu
PostgreSQL bằng pg_dump:90 phiên/491 đoạn/17 feedback; còn15 mẫu bổ sung/27 phiên
bản/3 bản xuất. Sau khi chạy Docker chính, hash toàn bộ bảng/số bản ghi vẫn khớp
nguyên bản. Không đưa các phiên thử vào database thật.

Hai image đã build. Frontend/backend/postgres đều healthy trên5173/8000/5433;
health xác nhận Turbo/cuda/fp16, binary_v04 CPU, PostgreSQL và worker CUDA check
đạt. Frontend HTTP200, URL API và CORS đúng; lịch sử cũ đọc được. Checkpoint lock
và14 artifact giữ nguyên hash. Model/cache không nằm trong image/Git và mount
chỉ đọc; source bind mount để cùng phát triển. Hướng dẫn: docker/README.md.
`.gitattributes` ngăn Git tự đổi LF/CRLF của CSV và metadata model đã khóa;
không sửa byte/hash hiện có hoặc renormalize artifact.

Replay thật trên database PostgreSQL tạm: upload2câu đã xác nhận giữ Zalo/TikTok/
rẻ nhất và2từ“không”; im lặng94,749s không sinh chữ qua upload/realtime. Realtime
tab114,331s hoàn tất114,942s,24unit/1retry không prompt/0unresolved, hàng chờ tối
đa1/tồn đọng cuối0. Giọng giảm gain0,1 và hai phiên nguồn khác nhau đạt. Feedback
không ghi đè dự đoán, bấm lại không trùng, export khớp; còn nguyên sau restart
backend thử. Đã dọn riêng database/container thử có marker. Chưa kiểm chứng lại
microphone vật lý hoặc thao tác chia sẻ tab thủ công. Đây không phải đánh giá
độ chính xác model. Log/JSON: runtime/docker-compose-20261028.

Lúc xuất image CUDA, ổ C gần đầy. Theo phép người dùng, đã dừng Docker và chuyển
đĩa dữ liệu sang D:\DockerDesktopData; giữ container/volume của cả hai dự án.
Hai bản VHDX sao lưu/chuyển được kiểm tra SHA256 trước restart. Database thật
khớp hash sau chuyển; đĩa D đang dùng, bản C không còn dùng được bỏ sau xác minh.
Backup đầy đủ: D:\lưu trữ\DockerDesktop_backup_20261028. Bản cấu hình trước chuyển
và pg_dump giữ trong backups; distro engine nhỏ trên C được giữ. C trống lại
khoảng28,8GB. Vị trí đĩa là cấu hình máy, không hardcode D vào Compose.
Không train, đổi dữ liệu/nhãn/checkpoint hoặc stage/commit/push.

## Chuẩn bị cộng tác Git theo từng bước

Bước1: `config/storage.json` là cấu hình riêng của máy, được Git bỏ qua và giữ
nguyên nội dung. Có `config/storage.example.json` cho người đã nhận kho lịch sử;
không cần kho đó để chạy ứng dụng. README tách kiểm tra archive thành tùy chọn.
Chưa stage/commit/push, không đổi checkpoint/dữ liệu. Đường dẫn khóa checkpoint
và dependency Docker đã được xử lý trong đợt đóng gói phía trên; trọng số vẫn
cần bàn giao riêng cho máy cùng phát triển.

## Dữ liệu hai phần và lưu trữ ngoài dự án — 28/10/2026

Theo xác nhận người dùng, giữ584 câu train và chuyển toàn bộ182 validation cũ
sang test; tổng766, giữ nguyên text/nhãn/ID/family/membership. Chỉ còn
`data/current/train.csv` (420 không cảnh báo/164 cảnh báo) và `test.csv`
(134 không cảnh báo/48 cảnh báo). Test đã từng chọn checkpoint v04, nên là
**hồi quy đã xem**, không phải test độc lập. Chưa train lại/đổi model. Hồ sơ
hash và nguồn: `config/data/current_dataset.json`. Metadata train/validation
trong checkpoint vẫn giữ nguyên lịch sử thật, không bị đổi tên hoặc sửa hash.

Đã chuyển754 file cũ/59 nhóm,8.603.100.934 byte sang
`D:\lưu trữ\livestream-violation-detection_20261028`. Giữ nguyên byte, thuộc tính
và hash sau chuyển. Bao gồm dataset gốc/cũ, checkpoint không chạy, archive/config
lịch sử, reports và outputs; toàn bộ nằm ngoài repository. Bảng ánh xạ ở
`config/storage.json`; manifest chuyển nằm ở root lưu trữ. Junction thư viện
trong outputs được giữ như con trỏ, không truy cập/copy/xóa target C: dùng chung.
Model v04 đang chọn, cache Turbo, database, feedback, debug, backups và audio
mẫu vẫn ở vị trí cũ. Backend không chạy khi bắt đầu chuyển (health từ chối kết
nối); chưa dùng việc dọn file để tuyên bố backend đang chạy hay chất lượng model.

## Ứng dụng hiện tại

- Hai chức năng giám sát riêng: **Microphone** và **Tab livestream**; giữ upload/thu thành file.
- ASR: Whisper Turbo, CUDA/FP16, LayerNorm FP32, tiếng Việt/transcribe, Silero VAD.
- PhoBERT: **binary_v04**, CPU, PhoBERT-only, không Rules/Hybrid.
- Checkpoint: `models/phobert_binary_v04_20261003_092559_043789/seed42/best`.
- Mapping: `0=KHONG_CANH_BAO`, `1=CANH_BAO`.
- Lịch sử, feedback và mẫu bổ sung: PostgreSQL Docker local, cấu hình trong `.env`.
- Đây là bản thử nội bộ, chưa được xác nhận đạt chất lượng triển khai thực tế.

## Ưu tiên hiện tại: bố cục dễ đọc

Theo yêu cầu người dùng, dừng nâng cấp model/dataset để sắp xếp repository.
Đợt24/10, script train/đánh giá/duyệt cũ và test nghiên cứu chuyển vào `archive/research/`;
73 file được lưu nguyên bản, có đường dẫn gốc → mới và SHA256 trong
`archive/manifest_20261024.json`. Cấu hình thử/rollback/train được nhóm trong
`config/`. Đợt28/10, các thư mục lịch sử này đã ra ngoài repository như ghi ở trên.
Hai cấu hình đang chọn vẫn ở gốc repository, checkpoint không đổi.

Code ứng dụng ở `backend/`, `frontend/`, `speech/`, `nlp/`; test hiện hành ở
`tests/` và `frontend/src/`; công cụ chẩn đoán ở `tools/diagnostics/`.
Backend không còn phụ thuộc script CLI `main.py`: các hàm nạp model nằm trong
`backend/model_runtime.py`, cấu hình trong `backend/model_config.py`.
Ngày28/10, theo yêu cầu tiếp theo, `main.py`/`model_config.py` gốc đã chuyển vào
`D:\lưu trữ\livestream-violation-detection_20261028\archive\legacy_cli_20261028`.
Hai file không được ứng dụng hiện tại import. Giữ nguyên nội dung, thuộc tính,
thời gian sửa và SHA256; manifest riêng `manifest_legacy_cli_20261028.json`
ghi đường dẫn để khôi phục, không sửa manifest lịch sử cũ. Tổng kho có756 file
gốc sau hai đợt. `tools.verify_archive` kiểm tra thêm manifest này qua registry
local. Chưa thấy file nguồn nào khác đủ căn cứ để chuyển thêm; giữ test, công cụ
chẩn đoán, tài liệu, audio mẫu và toàn bộ thành phần ứng dụng.

Kiểm tra riêng đợt chuyển hai CLI:13 test dữ liệu/cấu trúc/kho lưu trữ đạt;
73 hash lịch sử và2 hash mới khớp. Hash cấu hình ASR/PhoBERT, dữ liệu hiện hành
và hai module nạp model không đổi. Lượt chạy test tổng chưa trả kết quả và đã
dừng; không coi đây là xác nhận lại toàn bộ ứng dụng. Không đổi server/model/DB,
không stage/commit/push trong đợt này.

Không xóa mất dataset/checkpoint/report; bản cũ đã chuyển ngoài theo yêu cầu28/10.
Không di chuyển checkpoint hiện hành, audio debug, database, lịch sử hoặc feedback.
Không train, đổi nhãn hoặc tự ghép dữ liệu bổ sung. Không commit/push.
Kiểm tra sau sắp xếp:94 test ứng dụng +32 test PostgreSQL database riêng +61 test
frontend và build đạt. FE5173 trả HTTP200; BE8000 health ready với Turbo CUDA/FP16,
binary_v04 CPU và PostgreSQL. Đối chiếu nguyên nội dung/hash83 phiên,438 đoạn và16
feedback trước/sau không đổi. Manifest73 file lưu trữ đều khớp. Hai server đã bật lại.
Tài liệu dài và mọi mốc cũ được giữ tại
`archive/docs/PROJECT_STATUS_before_cleanup_20261024.md` và
`archive/docs/README_before_cleanup_20261024.md` dưới root lưu trữ bên ngoài.

## Lỗi đã biết và phạm vi kiểm chứng

### ASR

24/10/2026: ba phiên thật lỗi ở encoder F.linear/cuBLASLt với
`CUBLAS_STATUS_NOT_SUPPORTED`, trong đó39cfc442-4d56-4e51-862f-40cd956d3cb8
nhận15,5s nhưng còn12,6s chưa xử lý. Đã chọn classic biased addmm bằng biến
process-local `DISABLE_ADDMM_CUDA_LT=1` cho Windows CUDA/FP16; không đổi model,
trọng số, precision, VAD hoặc decoding. Startup kiểm tra query encoder thật trong
worker sau khi PhoBERT đã nạp; GPU lỗi thì không báo ready. Runtime lỗi CUDA cũng
được phản ánh riêng trong health, không sửa snapshot cấu hình startup.

Kiểm chứng thật trên Uvicorn/WebSocket và SQLite tạm: upload bản xác nhận2câu giữ
đúng Zalo/TikTok/rẻ nhất và2từ“không”; im lặng94,749s không có chữ; ba phiên voice
liên tiếp (có114,331s audio tab và bản giảm gain0,1) hoàn tất, hàng chờ tối đa1,
tồn đọng cuối0; không lỗi CUDA. Phiên tab ra24câu,1lượt retry không prompt,0đoạn
chưa nhận dạng; không thêm caption subscribe. Vẫn có từ ASR sai, không phải WER.
103 test ứng dụng đạt;32 test PostgreSQL opt-in bỏ qua trong lượt này.
Nguồn: `runtime/cuda-fix-20261024/`, `docs/BAO_CAO_SUA_CUDA_20261024.md`.

Đã tái hiện hai WAV 12s sinh caption với timestamp 29,98s. Code cũ cắt thời gian
nhưng giữ chữ, rồi ghép vào lời nói thật. Bản sửa kiểm tra timestamp trước khi
phân loại; chỉ khi sai nghiêm trọng mới thử lại một lần nguyên cửa sổ không prompt.
Lượt bình thường giữ gợi ý từ vựng. Không dùng blacklist/thay từ cứng/căn chỉnh từ.
Retry rỗng/sai/cùng câu hoặc đuôi quá ngắn được ghi rõ chưa nhận dạng, không an toàn.

Hai WAV lỗi được nhận dạng lại thành nội dung bán hàng. Upload im lặng thật94,75s
và replay nhanh cùng PCM không sinh chữ; upload bản hai câu đã xác nhận giữ cả hai
“không”, Zalo/TikTok/rẻ nhất. Replay toàn phiên tab trước đó bị gián đoạn khi đổi
ưu tiên; đã hoàn tất lại cùng audio với nhánh CUDA classic như ghi ở trên. Chưa thử microphone vật lý
hoặc thao tác chia sẻ tab thủ công sau sửa. Không bảo đảm loại mọi hallucination.
Chi tiết: `docs/BAO_CAO_SUA_ASR_20261024.md`; bằng chứng giữ ở runtime.

### PhoBERT: vẫn còn lỗi điều hướng Zalo

- Phiên `0e2e7df3-5c5d-441e-a8d1-77cfdee88aa3`:
  “Bạn nào muốn đặt số lượng nhiều thì nhắn qua Zalo để shop tư vấn và lên đơn nha.”
  và “Đơn số lượng lớn các bác trao đổi với Shop qua Zalo.” phải CANH_BAO;
  v04 chọn KHONG_CANH_BAO, xác suất cảnh báo8,78% và0,46%.
- Phiên `9fbeec0d-0cbd-4e3e-abba-6d93783d6528`:
  “Mọi người đặt hàng qua Zalo cho rẻ.” phải CANH_BAO; v04 chọn KHONG_CANH_BAO99,78%.
  “Đặt qua Zalo cho rẻ hơn.” trong báo cáo lại CANH_BAO92,92%; chưa ổn định với biến thể.

Giữ nguyên bằng chứng/nhãn, không thêm ngoại lệ ép dự đoán. Điểm validation đã xem
không phải độ chính xác livestream thực tế. Khi người dùng yêu cầu nâng cấp trở lại,
đối chiếu `reports/binary_v04/`, lỗi thực tế và các phiên bản feedback đã duyệt.

## Dữ liệu và việc sẽ làm sau

Dataset binary_v04 và các bản trước giữ nguyên. Mẫu bổ sung trong PostgreSQL có
phiên bản/nguồn/duyệt riêng; đã xuất CSV không đồng nghĩa đã dùng train. Không tự
chuẩn bị binary_v05, chia split hoặc train khi đạt số lượng. Tiếp tục thu dữ liệu
livestream thật khi được người dùng yêu cầu; giữ family/buổi thu cùng phía và không
đưa test đã khóa vào train. Không phục hồi17 câu nguy hiểm/54 câu không thực tế đã loại.

Hướng dẫn chạy và kiểm tra hiện hành ở `README.md`; hướng dẫn đọc kiến trúc ở
`docs/HUONG_DAN_DU_AN.md`.
