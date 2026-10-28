# Báo cáo ngắn: sửa phụ đề tự sinh — 24/10/2026

Giữ **Whisper Turbo CUDA/FP16, PhoBERT binary_v04, hai nhãn và PostgreSQL**. Không train, đổi dữ liệu, thêm luật phân loại hoặc xóa lịch sử/audio debug.

## Nguyên nhân có bằng chứng

Phiên tab `84ed1fa1-16cc-4056-b3a0-e45af59bc0c1`: cửa sổ đầu có **12,012 giây audio**, Silero đánh dấu 10,72 giây lời nói, nhưng Whisper sinh riêng câu “Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn” với timestamp **0–29,98 giây**. Code cũ cắt timestamp về độ dài audio nhưng vẫn giữ chữ; cơ chế giữ câu cuối (`carry`) ghép câu giả vào lời bán hàng của cửa sổ sau. Đây là đầu ra Whisper của chính phiên, không phải kết quả trả muộn từ phiên trước.

Phiên này không bật debug nên không thể nghe lại chính audio đó. Hai cửa sổ lỗi khác có WAV đã lưu tái hiện cùng hiện tượng. Chúng đúng PCM16 mono/16kHz, không clipping hoặc lệch DC đáng kể. Chỉ bỏ gợi ý từ vựng trên cùng WAV, giữ các tham số khác, đã thay caption giả bằng lời bán hàng với timestamp hợp lệ: prompt ảnh hưởng đến lỗi ở các trường hợp này. Không suy ra mọi lỗi ASR đều do prompt. `no_speech_prob` rất thấp vẫn đi kèm caption giả, nên không dùng riêng chỉ số này để xác nhận lời nói.

## Bản sửa

- Kiểm tra timestamp theo thời lượng PCM trước khi chốt chữ hoặc đưa vào PhoBERT; chỉ dung sai làm tròn 0,5 giây.
- Chỉ khi đầu ra có thời gian bất hợp lý, thử lại **một lần trên nguyên cửa sổ**, cùng Turbo/FP16/vi/transcribe, bỏ prompt; giữ timestamp nguyên bản, không chia đôi hoặc ép căn chỉnh từ.
- Không chấp nhận lại cùng nội dung chỉ vì timestamp thay đổi. Nếu vẫn lỗi, rỗng, hoặc là đuôi quá ngắn thiếu bằng chứng lời nói mới, hiển thị đoạn chưa nhận dạng đáng tin cậy và không phân loại; không ghép câu xuyên qua khoảng thiếu.
- Giữ raw hai lần, lý do và thông báo trên giao diện/lịch sử. Không blacklist “subscribe”, không thay từ cứng. Lượt nhận dạng thông thường vẫn dùng prompt hiện tại.
- Giảm đỉnh RAM khi nạp CUDA/FP16 bằng checkpoint mmap đã kiểm SHA256; giữ nguyên trọng số và LayerNorm FP32, không fallback model âm thầm.

## Đối chiếu trên cùng audio

Cả hai bản trước dưới đây đều sinh đúng câu “Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn”, timestamp 0–29,98 giây.

| Audio | Sau retry không prompt, nguyên văn ASR |
| --- | --- |
| `00b6725c…/window_2.wav`, 12 giây | Đây là mức giá rẻ nhất toàn sản ngày hôm nay Mọi người chốt nhanh để khỏi bỏ lỡ Bạn nào muốn đặt số lượng nhiều thì nhắn qua Zalo để cho shop tư vấn và lên đơn nha Cảm ơn cả nhà đưa dõi |
| `e0428626…/window_9.wav`, 12,012 giây | Có bánh to mà không có trứng không? Bánh to không trứng á Bánh to không trứng |

Thời gian riêng lượt không prompt lần lượt **1,77 giây và 0,44 giây**; tổng xử lý phục hồi còn gồm lượt nhận dạng ban đầu. Đây không phải bản chép chuẩn đã duyệt: vẫn có từ sai như “toàn sản”, “đưa dõi”, chưa tính WER/CER.

Audio hai câu người dùng đã xác nhận vẫn giữ cả hai từ “không” ở baseline. Bỏ prompt toàn cục lại làm “Zalo” thành “gia lô”, nên **không áp dụng thay đổi đó cho mọi cửa sổ**. Thử căn chỉnh từ và thử chia đôi vẫn sinh caption giả; chia đôi còn làm hỏng câu phủ định đã biết. Hai phương án này không dùng trong production.

## Kiểm tra và giới hạn

**90 test backend, 61 test frontend và build đạt.**

Hai cửa sổ lỗi đã đi qua Turbo/PhoBERT thật với retry không prompt. Upload bản im
lặng94,75s và replay nhanh cùng PCM không sinh transcript; upload bản hai câu đã
xác nhận giữ nguyên cả hai từ phủ định, Zalo/TikTok/rẻ nhất. Phép replay toàn phiên
tab theo nhịp thật bị gián đoạn khi chuyển ưu tiên công việc sang tổ chức repository;
chưa kết luận đạt bước này. Kết quả một phần lưu ở
`runtime/asr-recovery-20261024/real_pipeline_verification.json`.

Chưa thử lại bằng microphone vật lý hoặc thao tác chia sẻ tab thủ công sau bản sửa. Timestamp hợp lệ không bảo đảm transcript đúng; bản sửa chặn/phục hồi cơ chế lỗi đã tái hiện, không bảo đảm loại hết mọi hallucination. Phiên có đoạn chưa nhận dạng phải thể hiện thiếu nội dung; không diễn giải “0 cảnh báo” thành an toàn.

Bằng chứng: [phiên gốc](/D:/livestream-violation-detection/runtime/tab-live-check/session_84ed_original.json), [đối chiếu không prompt](/D:/livestream-violation-detection/runtime/asr-recovery-20261024/no_prompt_comparison.json), [thử căn chỉnh từ](/D:/livestream-violation-detection/runtime/asr-recovery-20261024/decode_comparison.json), [thử chia đôi](/D:/livestream-violation-detection/runtime/asr-recovery-20261024/partition_comparison.json).
