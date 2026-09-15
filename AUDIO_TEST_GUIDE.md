# Thu âm và kiểm thử pipeline với 9 câu có nhãn

Nguồn đọc là `data/audio_test_ground_truth.csv` (UTF-8), gồm đúng 9 câu và 3 câu mỗi nhãn.
Các nhãn/lý do do trợ lý đề xuất để kiểm thử; bạn có thể xem lại trong CSV trước khi thu.
Đây là bộ audio riêng, không phải final test v03 đã khóa và không dùng để train hoặc chỉnh ngưỡng.

## Cách thu một file MP3

1. Đọc lần lượt **AT001 → AT009** đúng nội dung bên dưới. Chỉ đọc câu trong dấu trích dẫn;
   không đọc ID, số thứ tự, nhãn hoặc lý do. Không thêm lời chào, giới thiệu hay bình luận.
2. Đọc rõ, tốc độ nói bình thường như đang giới thiệu sản phẩm trong livestream.
   Giữ nguyên các từ phủ định và thông tin trong câu. Đọc số bằng chữ như văn bản đã viết.
3. Để yên khoảng 2 giây trước câu đầu, nghỉ **khoảng 3 giây giữa hai câu**, rồi để yên
   khoảng 3 giây sau câu cuối. Khoảng nghỉ giúp Whisper chia đoạn nhưng không bảo đảm đúng 9 segment.
4. Dùng nơi ít tiếng ồn, giữ khoảng cách với micro ổn định và không bật nhạc nền.
   Nếu đọc nhầm hoặc bỏ câu, thu lại cả lượt để file cuối có mỗi câu đúng một lần và đúng thứ tự.
5. Xuất **một file MP3** tại `D:\livestream-violation-detection\audio\audio_test_9_sentences.mp3`.
   Nếu ứng dụng chỉ xuất WAV/M4A, phải chuyển đổi thật sang MP3; không chỉ đổi đuôi tên file.

Ví dụ chuyển bản WAV đã thu bằng FFmpeg, không ghi đè file MP3 có sẵn:

```powershell
Set-Location 'D:\livestream-violation-detection'
ffmpeg -n -i '.\audio\audio_test_9_sentences.wav' -ac 1 -ar 44100 -codec:a libmp3lame -b:a 128k '.\audio\audio_test_9_sentences.mp3'
```

## Nội dung đọc — không đọc các ID

**AT001**

> Cả nhà nhìn giúp em mặt sau chiếc ba lô, ngăn khóa kéo nhỏ nằm ngay dưới quai xách.

**AT002**

> Tấm rèm này cản được chín mươi phần trăm ánh sáng, bên em có phiếu thử nghiệm để khách kiểm tra.

**AT003**

> Chiếc máy xay này đã qua sử dụng nhưng em sẽ đóng hộp lại và nói với khách là hàng mới chưa từng dùng.

**AT004**

> Bộ thìa gỗ này không dùng trong lò vi sóng, mọi người đọc hướng dẫn sử dụng trước khi mua nhé.

**AT005**

> Tai nghe này dùng liên tục được tám giờ sau một lần sạc đầy, em sẽ gửi bảng thông số cho mọi người.

**AT006**

> Em biết lọ mật ong này có pha đường nhưng vẫn ghi nguyên chất để bán giá cao cho khách.

**AT007**

> Em đang mở hộp để mọi người xem các phụ kiện đi kèm chiếc đèn bàn.

**AT008**

> Túi đựng thực phẩm này được giới thiệu là dùng được trong ngăn đông, chị xem mức nhiệt cho phép trên bao bì.

**AT009**

> Lô bánh này đã hết hạn mà em vẫn dán ngày sản xuất mới rồi giới thiệu là hàng vừa ra lò.

## Chạy sau khi có bản thu

Kiểm tra CSV trước khi chạy; lệnh này chỉ đọc dữ liệu và đối chiếu text với v02/v03:

```powershell
Set-Location 'D:\livestream-violation-detection'
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\evaluate_audio_pipeline.py --validate-data
```

Chạy pipeline hiện có trên bản thu mới:

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\main.py '.\audio\audio_test_9_sentences.mp3'
```

File M4A cũng được hỗ trợ trực tiếp. Với bản thu hiện có, dùng
`audio/audio_test_9_sentences.m4a` thay cho đường dẫn MP3, không cần đổi đuôi hoặc chuyển đổi.

Cuối lệnh sẽ in đường dẫn `reports/audio_demo_<timestamp>/result.json`.
Thay `<timestamp>` dưới đây bằng đúng thư mục **của bản thu 9 câu vừa chạy**; không dùng
kết quả `voicetiengviet.mp3` cũ, vì đó là bản tin truy nã và không chứa bộ câu này.

```powershell
$demoResult = 'D:\livestream-violation-detection\reports\audio_demo_<timestamp>\result.json'
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\evaluate_audio_pipeline.py $demoResult --alignment-only
```

Mở `report.md` và `alignment.csv` ở thư mục kết quả được in ra. Nghe lại các đoạn có
`MISSING`, `NEEDS_REVIEW`, `DIFFERENT_NEEDS_LISTEN` hoặc `has_shared_segment=true`.
Không sửa lời nhận dạng hay nhãn chỉ để tăng điểm. Nếu bản thu đọc sai câu, có thể thu một
lượt mới đúng kịch bản và giữ báo cáo cũ làm dấu vết; đừng coi chênh lệch do đọc nhầm là lỗi Whisper.

Chạy đối chiếu nhãn và chẩn đoán lỗi bằng cùng JSON đã lưu:

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 .\evaluate_audio_pipeline.py $demoResult
```

Có thể thêm `--device cpu` khi không dùng GPU. Script không chạy lại Whisper.
Đầu vào `result.json` được kiểm tra checkpoint/hash, ngưỡng và hash mã detector trước khi
tái sử dụng nhãn. Script ưu tiên `classification_units` trong báo cáo mới, vẫn đọc được
`segments` của báo cáo cũ. Với `whisper.json` gốc chưa có nhãn, nó cũng tách câu theo dấu câu
trước khi nạp Hybrid một lần để phân loại từng unit và làm các đối chứng văn bản.

Nếu transcript toàn bản thu có độ giống token dưới 60%, script ghi báo cáo ghép cần xem lại
và không chạy model. Đây là điều kiện xem lại **phép ghép**, không phải chỉnh ngưỡng Hybrid.

## Cách đọc kết quả

Mỗi lần đánh giá tạo thư mục mới `reports/audio_pipeline_eval_<timestamp>/` với:

- `report.md`: bảng 9 câu, các loại lỗi và giải thích.
- `report.json`: số đo đầy đủ, phép ghép, các đoạn văn bản, nhãn/confidence/explanation,
  confusion matrix (hàng thật, cột dự đoán), hash nguồn và số lượt suy luận mới.
- `alignment.csv`: 9 dòng đối chiếu câu gốc, transcript, segment, mốc thời gian và nhãn.

**Nhận dạng:** so sánh sau chuẩn hóa NFC, chữ thường, khoảng trắng và dấu câu; giữ dấu tiếng Việt
và các từ phủ định. Báo độ giống token, độ giống ký tự, WER cùng từ bị thay/thêm/mất.
WER được tính trên token cách nhau bằng khoảng trắng (trong tiếng Việt thường là âm tiết),
không phải đánh giá từ đã tách bằng Underthesea. WER có thể vượt 100% nếu transcript thêm nhiều từ.
`tám` và `8` vẫn được ghi nhận khác chữ; cần nghe lại để phân biệt cách ghi số và nghe sai thực sự.

**Ghép:** dùng thứ tự câu và nội dung, không dùng nhãn hoặc confidence để quyết định ghép.
Câu khớp toàn bộ token ở một vị trí duy nhất được dùng làm mốc; các khoảng còn lại được ghép
bằng khoảng cách Levenshtein. Với báo cáo mới, đơn vị ghép/chấm là `classification_units`,
được tách trước đó bằng dấu `.`, `?`, `!` và hoàn toàn không dùng câu tham chiếu.
Một câu gốc có thể tương ứng nhiều unit nếu Whisper chèn thêm dấu kết thúc câu.
Đơn vị chứa nhiều câu gốc vẫn được đánh dấu dùng chung; không sao chép nhãn cho từng câu.

Raw Whisper segments được giữ nguyên trong `whisper_segments`. Unit có `unit_id` và
`source_segment_id` để truy vết. Các unit chia từ cùng một segment có timestamp ước lượng
theo độ dài ký tự và `timestamp_is_approximate=true`, không phải mốc từng từ chính xác.
Phần cuối không có dấu kết thúc vẫn được phân loại. Nếu không có dấu phân cách giữa hai
câu, không tự dùng ground truth để sửa ranh giới trong pipeline.

Đầu báo cáo ghi rõ số câu mong đợi, số raw segment, số unit phân loại, câu không ghép được
và câu chưa chấm riêng được. Không ép số unit thành 9: một câu gốc có thể bị ASR tách thành hai câu ngắn.

**Phân loại:** báo ba phép đo riêng, với số mẫu được chấm và tỷ lệ bao phủ:

1. `pipeline_classification_units`: chấm quyết định Hybrid của unit khi ghép được với đúng
   một câu tham chiếu. Đơn vị chung, thiếu hoặc ghép chưa chắc bị loại khỏi chỉ số này,
   nhưng vẫn có trong báo cáo. Với báo cáo cũ, chỉ số mang tên `pipeline_original_segments`
   và dùng các dự đoán nguyên segment cũ, không đổi kết quả lịch sử.
2. `clean_reference_text_diagnostic`: chạy Hybrid trên 9 câu gốc, bỏ qua Whisper. Sai ở đây
   cho thấy lỗi phân loại tồn tại ngay trên văn bản đúng.
3. `reconstructed_asr_text_diagnostic`: chạy Hybrid trên lời ASR được đối chiếu về từng câu.
   Phép ghép sử dụng câu tham chiếu để khôi phục ranh giới, nên đây là chẩn đoán,
   **không thay thế điểm pipeline thực tế**.

Ngoài ra, `pipeline_sentence_strict` tính một câu đúng khi mọi đơn vị riêng của câu đó đều
đúng nhãn. Luôn báo số đạt trên **tổng 9 câu**; câu thiếu/đơn vị chung/ghép chưa chắc không
được tính đúng. Không dùng bỏ phiếu hay tạo chính sách tổng hợp nhãn mới cho Hybrid.

Nếu Hybrid đúng trên câu gốc nhưng sai sau ASR, đó là dấu hiệu nghi ảnh hưởng nhận dạng hoặc
ranh giới ghép, chưa chứng minh chắc chắn nguyên nhân. Báo riêng danh sách khác transcript,
lỗi phân loại câu gốc, lỗi phân loại câu ASR ghép và lỗi trên segment thực tế.

Tất cả dự đoán dùng checkpoint v02 đã chốt và ngưỡng Hybrid cũ (0.55/0.65).
Confidence PhoBERT và xác suất PhoBERT của nhãn cuối vẫn được ghi riêng.
Bộ 9 câu nhỏ này dùng kiểm thử chức năng, không thay thế tập final test v03 hoặc đánh giá triển khai thực tế.

## Kiểm tra script bằng dữ liệu giả

```powershell
.\.venv-nlp\Scripts\python.exe -B -X utf8 -m unittest test_evaluate_audio_pipeline -v
```

Các kiểm tra không nạp model hoặc đọc final test, có các tình huống tách/gộp segment,
thiếu câu giữa, mất từ phủ định, transcript không liên quan và tách lỗi phân loại khỏi lỗi chữ.
