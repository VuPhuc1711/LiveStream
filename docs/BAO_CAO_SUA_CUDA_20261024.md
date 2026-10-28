# Sửa lỗi suy luận GPU — 24/10/2026

Giữ Whisper Turbo CUDA/FP16, LayerNorm FP32, prompt/decoding/Silero, PhoBERT
binary_v04 CPU và PostgreSQL. Không train, sửa dữ liệu, Rules hoặc checkpoint.

## Bằng chứng và thay đổi

Backend thật nhiều lần lỗi `CUBLAS_STATUS_NOT_SUPPORTED` tại encoder query
`F.linear`, trong heuristic `cublasLtMatmulAlgoGetHeuristic`. Phiên39cfc442 nhận
15,5s;12,6s chưa xử lý; chưa tới PhoBERT. `/health` cũ chỉ kiểm tra model đã nạp.
Log gốc giữ ở `runtime/cuda-fix-20261024/backend_before.stderr.log`.

Trên Windows CUDA/FP16, đặt mặc định **trong riêng tiến trình**
`DISABLE_ADDMM_CUDA_LT=1` trước CUDA addmm đầu tiên: chọn nhánh classic cho
biased addmm thay vì nhánh Lt đã lỗi. Không cài lại PyTorch/CUDA/driver, đổi model,
CPU fallback hoặc precision. Trọng số/checkpoint và nhãn giữ nguyên.

Sau khi cả hai model nạp, startup chạy query encoder thật shape1×1500×1280,
đúng CUDA/FP16 trong worker, kiểm tra output hữu hạn và synchronize. Không gọi
ASR trên audio giả. Lỗi thì startup không sẵn sàng. `whisper_config.cuda_runtime`
và `runtime_self_check` giữ snapshot startup; `asr_runtime` phản ánh lỗi thực thi
phát sinh sau đó, không tiếp tục báo ready khi GPU lỗi.

PyTorch cài2.13.0+cu126. Nguồn chính thức: [cờ tắt fused Lt trong đúng commit
PyTorch](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Blas.cpp#L107-L142),
[NVIDIA CUDA12.6 heuristic](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublasltmatmulalgogetheuristic).
Mã lỗi này không tự chứng minh hết VRAM. Baseline mới sau restart cũng qua được
worker/audio; chưa chốt tác nhân sâu hơn khiến lỗi xuất hiện gián đoạn. Bản sửa
chủ động tránh nhánh gây lỗi và thêm kiểm chứng thực thi, không gọi đó là bằng
chứng lỗi phần cứng hoặc bảo đảm không bao giờ gặp lỗi GPU khác.

## Kiểm chứng

Uvicorn thật, HTTP/WebSocket thật và SQLite tạm riêng, không thêm mẫu test vào
PostgreSQL sử dụng thật:

| Trường hợp | Kết quả |
| --- | --- |
| Upload bản hai câu người dùng xác nhận | Giữ nguyên transcript, Zalo/TikTok/rẻ nhất và cả hai “không”;2,01s |
| Upload im lặng thật94,749s | Transcript rỗng, không phân loại;1,79s |
| Realtime bản xác nhận + im lặng đầu/cuối |2câu đúng nguyên văn; tồn đọng cuối0 |
| Realtime bản tab114,331s theo tốc độ thật | Chạy115,35s;24câu,1retry không prompt,0unresolved; queue tối đa1, cuối0 |
| Phiên tiếp theo dùng bản giảm gain0,1 |2câu nguyên văn, không lẫn nội dung phiên trước; cuối0 |
| Realtime im lặng thật, truyền tăng tốc | Không ASR window/transcript; cuối0 |

Bản tab không còn caption Ghiền Mì Gõ/La La School trong lượt này; vẫn có sai
từ vựng/đơn vị nên không gọi là bản chép chuẩn hoặc tính WER. Timestamp câu đều
có end>start. Export JSON khớp lịch sử; các phiên tạm tách ID và audio accounting
đầy đủ.103 test ứng dụng đạt;32 test PostgreSQL opt-in bỏ qua vì schema/DB không
đổi (đã kiểm tra32 test đó ở bước tổ chức trước). Frontend không đổi trong lượt này.

Bằng chứng: `runtime/cuda-fix-20261024/api_verification.json`, `pipeline_unset.json`,
`pipeline_default.json` và log cùng thư mục. Chưa thử lại microphone vật lý hoặc
thao tác chia sẻ tab thủ công sau bản sửa. Các phiên thất bại cũ giữ nguyên, không
thể tự chạy lại phần audio không được lưu; cần tạo phiên mới trên frontend.

## Chạy lại / đối chiếu

Chạy backend bằng lệnh README; policy Windows CUDA/FP16 tự áp dụng. Health cần
`cuda_runtime.environment_value="1"`, `runtime_self_check.status="passed"`,
`ready=true`, `model_version="binary_v04"`, database PostgreSQL.
Đối chiếu kỹ thuật nhánh cũ: đặt `$env:DISABLE_ADDMM_CUDA_LT='0'` trước khi chạy
**tiến trình mới**. PyTorch cache cờ này nên không đổi nóng giữa phiên. Bỏ override
bằng `Remove-Item Env:DISABLE_ADDMM_CUDA_LT -ErrorAction SilentlyContinue` rồi
restart để trở về mặc định mới. Không dùng đối chiếu này khi demo.
