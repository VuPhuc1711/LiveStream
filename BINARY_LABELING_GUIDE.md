# Hướng dẫn gán nhãn binary — cập nhật 06/10/2026

Mapping giữ nguyên: 0=KHONG_CANH_BAO, 1=CANH_BAO. Áp dụng quy ước dự án do người dùng chốt, không phải kết luận về chính sách pháp lý hoặc chính sách nền tảng được xác minh bên ngoài.

## Điều hướng khỏi TikTok trong livestream

- CANH_BAO: người bán mời hoặc hướng dẫn khách sang nền tảng/kênh khác để tư vấn, liên hệ phục vụ mua hàng, đặt hàng hoặc thanh toán. Bao gồm Shopee, Lazada, Zalo, Facebook/Messenger và các kênh ngoài TikTok khác.
- Tư vấn ngoài TikTok rồi quay lại TikTok đặt hàng/thanh toán vẫn CANH_BAO. Không yêu cầu giao dịch cuối cùng xảy ra ngoài sàn mới cảnh báo.
- Người bán giới thiệu gian hàng/kênh mua hàng của mình trên nền tảng ngoài TikTok cũng là CANH_BAO, dù chưa mời đặt hàng rõ ràng: “Shop cũng có gian hàng Shopee.” Không coi vế “thanh toán tại TikTok” là bác bỏ việc giới thiệu kênh mua hàng khác.
- Không gán nhãn chỉ vì tên nền tảng. Lời nhắc trung lập, câu hỏi thật, phủ định hoặc bác bỏ việc điều hướng cần đọc toàn câu; giữ KHONG_CANH_BAO nếu không có nội dung cảnh báo khác.
- Một câu từ chối nhận đơn qua Zalo không tự động bác bỏ lời mời tư vấn qua Zalo ở vế khác. Xét phạm vi phủ định và toàn bộ hành vi được mời/hướng dẫn.
- Nếu câu chỉ nói nhắn/liên hệ mà không xác định kênh ngoài TikTok, không tự bổ sung ngữ cảnh đó. Đánh dấu cần người dùng làm rõ khi có nghi vấn, không tự đổi nhãn đã duyệt.
- Link chỉ để đọc tài liệu sử dụng sau mua chưa được người dùng chốt là điều hướng phục vụ mua hàng hay không; không tự mở rộng quy ước sang mọi đường dẫn tài liệu.

Phân biệt: “Shop cũng có gian hàng Shopee.” là khẳng định/giới thiệu (CANH_BAO);
“Shop có bán trên Shopee không?” là câu hỏi thật (KHONG_CANH_BAO nếu không có ý
khẳng định khác); “Shop không có gian hàng Shopee.” là phủ định. Nhắc tên nền tảng
trung lập hoặc trích dẫn bị bác bỏ không tự thành cảnh báo. Đây là quy ước riêng
của dự án do người dùng xác nhận, không phải kết luận pháp lý về nền tảng.

## Feedback và dữ liệu bổ sung

- Transcript, dự đoán và xác suất gốc giữ nguyên trong lịch sử; văn bản sửa/nhãn
  xác nhận là feedback riêng. Feedback chưa duyệt không được tự đưa vào train.
- Chỉ thao tác “Duyệt đưa vào dữ liệu” cho phiên bản cụ thể mới tạo trạng thái
  DA_DUYET, kèm tên người thực sự duyệt và thời gian. Mẫu AI giữ AI_GENERATED dù
  đã được người dùng duyệt; mẫu chưa duyệt không được gán tên người duyệt.
- Mỗi sửa đổi tạo lịch sử và đưa mẫu trở lại CHO_DUYET; CSV cũ vẫn là snapshot
  bất biến. CSV bổ sung chưa phải split hoặc lệnh train.
- Mẫu lời nói thực tế chỉ ghi USER_CONFIRMED_SPEECH khi người dùng đã xác nhận
  văn bản. Không nhận đã nghe/đối chiếu audio nếu phiên không lưu debug.
- Cùng buổi thu/cặp đối chiếu/biến thể phải giữ cùng family khi chuẩn bị split.
  Trước train phải kiểm tra giao nhau với test đã khóa; lượt thu thập này không
  tự đọc lại test, chia split hay ghép bộ binary_v04 đang dùng.

Quyết định cụ thể của người dùng: “Mẫu đặt riêng chưa có trong giỏ, chị gửi yêu cầu qua Zalo để tư vấn; shop tạo mã trên sàn rồi chị mới đặt và thanh toán.” là CANH_BAO (trước đề xuất KHONG_CANH_BAO). Đây là sửa đề xuất REVIEW_F03, chưa thuộc train/validation.

## Các quyết định binary tiếp tục giữ

- Lời mời mua thông thường, cảm ơn, mô tả và thông số/thành phần/bảo hành thông thường giữ phạm vi đã duyệt.
- Khẳng định rẻ nhất/đứng đầu không giới hạn hoặc toàn sàn/thị trường là CANH_BAO; giới hạn rõ trong danh mục riêng của người bán là KHONG_CANH_BAO. Câu hỏi thật, phủ định và trích dẫn bác bỏ khác với khẳng định/tán thành.
- Giữ 17 ID hướng dẫn nguy hiểm đã được người dùng loại khỏi bộ phát triển hiện tại; không phục hồi bằng cập nhật quy ước này.
- Không đổi nhãn theo dự đoán model. Đề xuất mới là CHUA_DUYET; chỉ ghi DA_DUYET và reviewer Vũ Hồng Phúc khi có quyết định cụ thể của người dùng.
- Dữ liệu binary_v03 đã train và các checkpoint/báo cáo giữ nguyên để tái lập. Rà nhãn sau cập nhật quy ước không phải lệnh train hay chuyển backend.
