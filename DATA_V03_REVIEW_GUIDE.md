# Hướng dẫn duyệt dữ liệu ứng viên v03

60 câu tổng hợp mới, 20 family, đang CHUA_DUYET; chưa dùng huấn luyện hoặc chia tập.

## Tiêu chí nhãn

- KHONG_CANH_BAO: mô tả, thao tác, thông tin thông thường hoặc hướng dẫn giới hạn sử dụng; không khẳng định hiệu năng hay quyền lợi cần kiểm chứng.
- CAN_XAC_MINH: có khẳng định khách quan về hiệu suất, thành phần, xuất xứ hoặc quyền lợi cần bằng chứng. Nói có phiếu thử không đồng nghĩa phiếu đã được xác thực.
- CANH_BAO: gian dối rõ ràng, tự thừa nhận che giấu hoặc nói sai, dàn dựng bằng chứng hay cam kết nguy hiểm.

Không chọn nhãn chỉ vì xuất hiện số phần trăm, từ “chứng nhận”, “chính hãng” hoặc “không”. Đọc toàn câu và xem người bán đang khẳng định, phủ định hay chỉ yêu cầu kiểm tra.

## Cách duyệt

1. Mở data/candidates_v03_review.csv bằng UTF-8; đọc ba câu trong từng family cùng lý do và cue.
2. Sửa proposed_label nếu cần; giữ một trong ba nhãn. Điền tên người duyệt vào reviewer và giải thích trong review_note, kể cả khi đồng ý.
3. Chỉ chuyển review_status sang DA_DUYET sau khi đã xác nhận cả câu và lý do. Câu còn tranh luận giữ CHUA_DUYET.
4. Nếu đổi nhãn, ghi nhãn cũ → nhãn mới và lý do trong review_note. Không cố giữ cân bằng nhãn bằng cách duyệt sai; chạy kiểm tra với --review-progress để báo lại phân bố.
5. Giữ ID/family ổn định; nếu sửa text hãy ghi nội dung sửa trong review_note và chạy lại kiểm tra trùng. Chưa đưa ứng viên vào split hoặc train.

## Cặp ASR

Có 10 dòng ASR_LIKE. Mỗi dòng lưu text nhiễu và clean_reference_text là câu sạch mới tương ứng; cùng family_id và proposed_label của chính dòng đó. Câu sạch tham chiếu không phải dòng thứ 61 trở đi và không tự động trở thành dữ liệu train.
Nhiễu là giả lập gần âm/dấu thanh, chưa được xác nhận là lỗi của Whisper trên bản thu thật. Đọc cả hai để xác nhận giữ nguyên ý nghĩa; nếu không chắc, giữ CHUA_DUYET. Không sửa mất phủ định, con số hoặc dấu hiệu gian dối.

Lệnh kiểm tra ban đầu: python -B -X utf8 validate_candidates_v03.py
Trong quá trình duyệt: python -B -X utf8 validate_candidates_v03.py --review-progress
Script chỉ đọc dữ liệu và in kết quả, không train, không tự duyệt và không ghi đè báo cáo ban đầu.

## Điểm cần lưu ý từ dữ liệu cũ

Xem reports/data_candidates_v03/report.md và near_similarity_review.json. Gần chủ đề không đồng nghĩa trùng câu; người duyệt cần quyết định mức độc lập. Final test v03 và audio test tiếp tục là dữ liệu giữ riêng, không dùng làm nguồn train.
Audio AT008 đã được Vũ Hồng Phúc chốt KHONG_CANH_BAO. CSV audio có 4/2/3 mẫu và thêm thông tin duyệt; báo cáo cũ vẫn phản ánh nhãn trước sửa. Bộ đọc evaluate_audio_pipeline.py hiện yêu cầu 4 cột và 3/3/3, cần một thay đổi riêng được cho phép trước lần đánh giá tiếp theo.

## 20 nhóm ứng viên

### CV03_F001 — PIN_QUAT

**CV03_001 — KHONG_CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Em xoay chiếc quạt kẹp sang bên để cả nhà thấy nút chỉnh gió ở cạnh chân để.

Lý do đề xuất: Thao tác giới thiệu vị trí nút bấm, không tuyên bố hiệu năng.

Câu sạch cùng nhãn/family: Em xoay chiếc quạt kẹp sang bên để cả nhà thấy nút chỉnh gió ở cạnh chân đế.

Nhiễu ASR giả lập: chân đế → chân để; không đổi dấu hiệu quyết định nhãn.

**CV03_002 — CAN_XAC_MINH — CLEAN**

Quạt kẹp chạy được sáu giờ ở nấc gió thấp sau khi sạc, ai cần em gửi bản đo của hãng.

Lý do đề xuất: Thời lượng định lượng cần đối chiếu bản đo và điều kiện pin; hứa gửi chưa phải bằng chứng đã kiểm tra.

**CV03_003 — CANH_BAO — CLEAN**

Bản đo quạt kẹp chỉ ghi hai giờ mà trên live em đọc thành sáu giờ, cứ giấu bản đo đi cho dễ bán.

Lý do đề xuất: Chủ động xuyên tạc số đo và giấu tài liệu khi bán hàng.

### CV03_F002 — PIN_MIC

**CV03_004 — KHONG_CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Cổng cắm sạc của micro nằm dưới nắp cao xu, em chỉ chỗ này để các bác khỏi tìm lâu.

Lý do đề xuất: Hướng dẫn tìm cổng sạc, không hứa thời lượng hay khả năng chống nước.

Câu sạch cùng nhãn/family: Cổng cắm sạc của micro nằm dưới nắp cao su, em chỉ chỗ này để các bác khỏi tìm lâu.

Nhiễu ASR giả lập: cao su → cao xu; không đổi dấu hiệu quyết định nhãn.

**CV03_005 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Một lượt nạp đầy cho micro thu tiếng liên tục mười giờ ở chế độ thường, số liệu này có trong hồ sơ thử pin.

Lý do đề xuất: Khẳng định thời lượng sử dụng cần kiểm tra chế độ đo và hồ sơ.

Chú ý: Có hồ sơ thử pin là lời người bán, chưa xác thực hồ sơ; không nâng thành gian dối chỉ vì số giờ.

**CV03_006 — CANH_BAO — CLEAN**

Micro đang bị chai pin nhưng em sửa ảnh thông số thành pin mới, khách xem ảnh sẽ tưởng chưa có lỗi.

Lý do đề xuất: Cố tình sửa thông tin để che giấu tình trạng pin.

### CV03_F003 — HIEU_QUA_LOC_BUI

**CV03_007 — KHONG_CANH_BAO — CLEAN**

Em nhấc tấm lọc ra cho mọi người thấy chiều lắp, mũi tên phải quay về phía thân máy.

Lý do đề xuất: Mô tả thao tác lắp bộ lọc thông thường.

**CV03_008 — CAN_XAC_MINH — ASR_LIKE** — CẦN CHÚ Ý, DUYỆT CẶP ASR

Bộ lọc hút được chín mươi lăm phần trăm bụi trong buồn thử của hãng, mình cần xem cỡ hạt trong biên bản.

Lý do đề xuất: Hiệu suất định lượng phụ thuộc cỡ hạt và buồng thử; chưa có dấu hiệu thừa nhận gian dối.

Câu sạch cùng nhãn/family: Bộ lọc hút được chín mươi lăm phần trăm bụi trong buồng thử của hãng, mình cần xem cỡ hạt trong biên bản.

Nhiễu ASR giả lập: buồng thử → buồn thử; không đổi dấu hiệu quyết định nhãn.

Chú ý: Tỷ lệ lọc có phạm vi buồng thử; không suy diễn thành hiệu quả ở mọi phòng.

**CV03_009 — CANH_BAO — CLEAN**

Kết quả lọc bụi thấp quá nên em lấy biên bản của máy khác gắn vào mẫu đang bán để khách tin hiệu suất cao.

Lý do đề xuất: Dùng tài liệu không thuộc sản phẩm để gây hiểu lầm về hiệu suất.

### CV03_F004 — HIEU_QUA_VOI_NUOC

**CV03_010 — KHONG_CANH_BAO — CLEAN**

Đầu vòi đang tháo rời trên khay, các bác kiểm tra kiểu ren nhà mình trước khi chọn nhé.

Lý do đề xuất: Hướng dẫn lựa chọn cấu hình phù hợp, không khẳng định tiết kiệm.

**CV03_011 — CAN_XAC_MINH — CLEAN**

Đầu vòi này giảm được bốn mươi phần trăm lượng nước so với đầu thường ở cùng áp lực, em có bảng đo lưu lượng.

Lý do đề xuất: So sánh định lượng cần bảng đo, đầu đối chứng và áp lực thử.

**CV03_012 — CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Lúc quay so vòi nướt em cố ý khóa bớt van bên mẫu của shop rồi bảo khách là tự nó tiết kiệm hơn.

Lý do đề xuất: Dàn dựng phép so sánh để đánh lừa người xem.

Câu sạch cùng nhãn/family: Lúc quay so vòi nước em cố ý khóa bớt van bên mẫu của shop rồi bảo khách là tự nó tiết kiệm hơn.

Nhiễu ASR giả lập: vòi nước → vòi nướt; không đổi dấu hiệu quyết định nhãn.

### CV03_F005 — CHONG_NUOC_LOA

**CV03_013 — KHONG_CANH_BAO — CLEAN**

Đang vệ sinh loa thì rút dây nguồn trước, đừng xịt nước vào các khe phía sau nhé.

Lý do đề xuất: Giới hạn sử dụng thận trọng, không khẳng định sản phẩm chịu nước.

**CV03_014 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Loa dã ngoại chịu được tia nước theo mức IPX5 mà hãng công bố, hồ sơ kiểm tra cần đúng mã loa này.

Lý do đề xuất: Tuyên bố cấp bảo vệ cần đối chiếu chứng từ và đúng mã sản phẩm.

Chú ý: Đối chiếu đúng IPX5 và tia nước, không suy diễn thành ngâm nước.

**CV03_015 — CANH_BAO — CLEAN**

Loa vừa vào nước đã chập mà em vẫn lấy đoạn quay loa khác ngâm bể để bán như thể mẫu này ngâm được.

Lý do đề xuất: Biết sản phẩm hỏng do nước nhưng dùng video khác để đánh lừa khách.

### CV03_F006 — CHIU_NHIET_KHAY

**CV03_016 — KHONG_CANH_BAO — CLEAN** — CẦN CHÚ Ý

Khay này có phần nắp rời, trước khi cho vào lò các bác đọc riêng hướng dẫn của nắp và thân khay.

Lý do đề xuất: Chỉ yêu cầu kiểm tra điều kiện sử dụng từng bộ phận, chưa bảo đảm mức chịu nhiệt.

Chú ý: Chỉ yêu cầu đọc hướng dẫn từng bộ phận; phân biệt với câu 17 khẳng định mức nhiệt.

**CV03_017 — CAN_XAC_MINH — ASR_LIKE** — DUYỆT CẶP ASR

Thân khay chịu được hai trăm độ C trong nửa giờ theo công bố của bên sản xuất, em sẽ cung cấp phiếu thư.

Lý do đề xuất: Giới hạn nhiệt và thời gian là thông số khách quan cần xác minh.

Câu sạch cùng nhãn/family: Thân khay chịu được hai trăm độ C trong nửa giờ theo công bố của bên sản xuất, em sẽ cung cấp phiếu thử.

Nhiễu ASR giả lập: phiếu thử → phiếu thư; không đổi dấu hiệu quyết định nhãn.

**CV03_018 — CANH_BAO — CLEAN**

Nắp khay đã chảy trong lần thử nóng nhưng em cắt đoạn đó khỏi video rồi khuyên khách cứ cho cả nắp vào lò.

Lý do đề xuất: Che giấu thử nghiệm thất bại và hướng dẫn sử dụng có nguy cơ gây hại.

### CV03_F007 — CAN_SANG_TAM_CHE

**CV03_019 — KHONG_CANH_BAO — CLEAN**

Tấm che cửa xe gấp lại theo đường nếp này, em làm chậm để các bác nhìn rõ cách cất.

Lý do đề xuất: Hướng dẫn gấp và cất sản phẩm, không nêu mức cản sáng.

**CV03_020 — CAN_XAC_MINH — ASR_LIKE** — CẦN CHÚ Ý, DUYỆT CẶP ASR

Tấm che kính xe giảm bảy mươi lăm phần trăm ánh sáng trong phép đo này, cần đối chiếu loại kín thử cùng nó.

Lý do đề xuất: Tuyên bố hiệu quả cản sáng có điều kiện vẫn cần xác minh phép đo.

Câu sạch cùng nhãn/family: Tấm che kính xe giảm bảy mươi lăm phần trăm ánh sáng trong phép đo này, cần đối chiếu loại kính thử cùng nó.

Nhiễu ASR giả lập: loại kính → loại kín; không đổi dấu hiệu quyết định nhãn.

Chú ý: Cản sáng định lượng vẫn cần xác minh dù đã giới hạn loại kính thử.

**CV03_021 — CANH_BAO — CLEAN**

Để bán tấm che kính em hạ sáng máy quay ở cảnh sau rồi nói mức tối đi đó hoàn toàn do sản phẩm.

Lý do đề xuất: Can thiệp điều kiện quay rồi gán sai hiệu quả cho sản phẩm.

### CV03_F008 — CHAT_LIEU_THAM

**CV03_022 — KHONG_CANH_BAO — CLEAN**

Các bác nhìn mép tấm thảm em đang cuộn, đường viền chạy vòng quanh cả bốn cạnh.

Lý do đề xuất: Mô tả bộ phận nhìn thấy, không khẳng định thành phần sợi.

**CV03_023 — CAN_XAC_MINH — CLEAN**

Sợi len chiếm tám mươi phần trăm trong tấm thảm này, em đang xin phiếu phân tích thành phần để đối chiếu.

Lý do đề xuất: Khẳng định tỷ lệ chất liệu cần phân tích; đang xin phiếu chưa đủ xác nhận.

**CV03_024 — CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Xưởng báo thảm làm từ sợi nhựa mà em cất nhản đi rồi giới thiệu là len cừu để lấy giá cao hơn.

Lý do đề xuất: Biết thành phần thực nhưng che nhãn và nói sai chất liệu.

Câu sạch cùng nhãn/family: Xưởng báo thảm làm từ sợi nhựa mà em cất nhãn đi rồi giới thiệu là len cừu để lấy giá cao hơn.

Nhiễu ASR giả lập: nhãn đi → nhản đi; không đổi dấu hiệu quyết định nhãn.

### CV03_F009 — THANH_PHAN_NGU_COC

**CV03_025 — KHONG_CANH_BAO — CLEAN**

Ai dị ứng hạt thì đọc danh sách nguyên liệu trên lon ngũ cốc trước, em không chọn hộ cho mình được.

Lý do đề xuất: Khuyến nghị kiểm tra thành phần theo nhu cầu cá nhân, không cam kết an toàn.

**CV03_026 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Trong một khẩu phần ngũ cốc có mười hai gam chất đạm theo bảng dinh dưỡng của hãng, chị đối chiếu định lượng khẩu phần nhé.

Lý do đề xuất: Hàm lượng dinh dưỡng cần đối chiếu nhãn, khẩu phần và dữ liệu kiểm nghiệm.

Chú ý: Hàm lượng dinh dưỡng là số liệu cần đối chiếu; phân biệt với hướng dẫn đọc nhãn ở câu 25.

**CV03_027 — CANH_BAO — CLEAN**

Mẻ ngũ cốc có đậu phộng nhưng em bỏ tên nó khỏi nhãn để người dị ứng cũng tưởng là dùng được.

Lý do đề xuất: Cố ý giấu thành phần gây dị ứng, tạo nguy cơ cho người mua.

### CV03_F010 — XUAT_XU_DAO

**CV03_028 — KHONG_CANH_BAO — CLEAN**

Em đưa đáy hộp dao lại gần camera để chị đọc dòng nơi sản xuất, hình lá cờ trang trí chưa nói lên nguồn gốc.

Lý do đề xuất: Hướng dẫn đọc thông tin xuất xứ, không tự xác nhận nguồn gốc qua trang trí.

**CV03_029 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Lô dao được gia công tại Đức rồi nhập về qua đơn vị này, mình có thể đối chiếu tờ khai với mã lô.

Lý do đề xuất: Tuyên bố nơi gia công và nhập khẩu cần chứng từ đúng lô.

Chú ý: Nguồn gia công và đơn vị nhập cần tài liệu; chưa có dấu hiệu nói dối trong câu.

**CV03_030 — CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Dao được làm ngay tại xưởng trong nước nhưng em thuê in địa chĩ xưởng Đức lên hộp để giả hàng nhập.

Lý do đề xuất: Chủ động làm giả thông tin nguồn gốc để bán hàng.

Câu sạch cùng nhãn/family: Dao được làm ngay tại xưởng trong nước nhưng em thuê in địa chỉ xưởng Đức lên hộp để giả hàng nhập.

Nhiễu ASR giả lập: địa chỉ → địa chĩ; không đổi dấu hiệu quyết định nhãn.

### CV03_F011 — CHINH_HANG_HOP_MUC

**CV03_031 — KHONG_CANH_BAO — CLEAN** — CẦN CHÚ Ý

Hộp mực có tem để quét, em chưa tra kết quả nên chưa xác nhận gì về hãng sản xuất nhé.

Lý do đề xuất: Thừa nhận chưa xác minh và chưa khẳng định chính hãng.

Chú ý: Có tem nhưng chưa xác nhận hãng; không gán nhãn theo riêng từ tem hoặc hãng.

**CV03_032 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Hộp mực này là hàng chính hãng, mã xác thực trên tem có thể tra ở trang của nhà sản xuất.

Lý do đề xuất: Tuyên bố chính hãng cần kiểm tra mã, nguồn tra và sản phẩm tương ứng.

Chú ý: Có khẳng định chính hãng; khả năng tra mã chưa chứng minh mã là thật.

**CV03_033 — CANH_BAO — CLEAN**

Em mua tem của hộp mực thật dán sang hộp nhái rồi quay cận tem để khách tin là hàng của hãng.

Lý do đề xuất: Dùng tem thật cho hàng nhái nhằm đánh lừa người mua.

### CV03_F012 — CHUNG_NHAN_MU

**CV03_034 — KHONG_CANH_BAO — CLEAN**

Các bác xem dây cài mũ bảo hiểm em đang luồn qua khóa, chỉnh vừa đầu rồi mới cài nhé.

Lý do đề xuất: Hướng dẫn thao tác đội mũ, không tuyên bố chứng nhận.

**CV03_035 — CAN_XAC_MINH — CLEAN**

Mẫu mũ này đã đạt chứng nhận hợp quy, khách cần đối chiếu số chứng nhận với tên mẫu trên hồ sơ.

Lý do đề xuất: Khẳng định đạt chứng nhận cần kiểm tra tính hợp lệ và phạm vi mẫu.

**CV03_036 — CANH_BAO — CLEAN**

Giấy hợp quy không có tên mẫu mũ này nên em sửa tên mẫu trong bản chụp trước khi gửi người mua.

Lý do đề xuất: Tự thừa nhận sửa chứng từ để giả phạm vi chứng nhận.

### CV03_F013 — GIA_CAN_DIEN_TU

**CV03_037 — KHONG_CANH_BAO — CLEAN**

Chiếc cân em đang cầm là bản có màn hình rời, chị chọn đúng bản đó rồi xem số tiền trước khi đặt.

Lý do đề xuất: Hướng dẫn đối chiếu phân loại và giá, không hứa giảm hay giá thấp nhất.

**CV03_038 — CAN_XAC_MINH — CLEAN**

Bản cân có màn hình rời hôm nay giảm hai mươi phần trăm so với giá niêm yết tuần trước.

Lý do đề xuất: So sánh giá theo thời điểm cần lịch sử giá đúng phân loại.

**CV03_039 — CANH_BAO — CLEAN**

Em gắn giá của cục pin lên hình chiếc cân để khách bấm mua, vào thanh toán họ mới biết giá cân cao hơn.

Lý do đề xuất: Cố tình dùng giá phụ kiện làm giá mồi cho sản phẩm khác.

### CV03_F014 — KHUYEN_MAI_SACH

**CV03_040 — KHONG_CANH_BAO — CLEAN**

Mã sách nào tham gia ưu đãi được đánh dấu trong danh sách, mình kiểm tra tên bộ trước khi thêm vào đơn.

Lý do đề xuất: Hướng dẫn xem phạm vi chương trình, không bảo đảm nhận ưu đãi.

**CV03_041 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Mua bộ sách bản bìa cứng trong đợt này được hoàn hai mươi nghìn vào ví sàn khi đơn hoàn tất.

Lý do đề xuất: Cam kết quyền lợi khuyến mãi cần kiểm tra điều kiện, thời hạn và đối tượng.

Chú ý: Hoàn vào ví có điều kiện là quyền lợi cần đối chiếu, không mặc định là lừa đảo.

**CV03_042 — CANH_BAO — CLEAN**

Bộ sách này bị loại khỏi chương trình hoàn tiền rồi nhưng em vẫn treo thông báo được hoàn để dụ đặt đơn.

Lý do đề xuất: Biết không đủ điều kiện nhưng vẫn quảng cáo quyền lợi không có.

### CV03_F015 — HAN_DUNG_TRAI_CAY

**CV03_043 — KHONG_CANH_BAO — CLEAN**

Với túi trái cây sấy, mọi người xem ngày hết hạn in ở mép hàn trước khi mở ăn nhé.

Lý do đề xuất: Hướng dẫn đọc hạn dùng, không khẳng định thời gian còn sử dụng.

**CV03_044 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Túi trái cây sấy chưa mở bảo quản được chín tháng ở nơi khô mát theo công bố của xưởng.

Lý do đề xuất: Thời hạn bảo quản có điều kiện cần kiểm tra nhãn và căn cứ công bố.

Chú ý: Thời hạn bảo quản chưa mở khác ngày hết hạn cụ thể; cần xem điều kiện.

**CV03_045 — CANH_BAO — CLEAN**

Túi trái cây đã quá hạn nên em xóa mốc hết hạn bằng dung môi rồi trộn vào lô còn mới để bán tiếp.

Lý do đề xuất: Cố ý xóa hạn dùng và che giấu hàng quá hạn.

### CV03_F016 — BAO_HANH_MAY_IN

**CV03_046 — KHONG_CANH_BAO — CLEAN**

Khi cần gửi máy in đi kiểm tra, chị giữ phiếu mua và mô tả lỗi cho bộ phận tiếp nhận nhé.

Lý do đề xuất: Hướng dẫn thủ tục tiếp nhận, không cam kết kết quả hay thời hạn bảo hành.

**CV03_047 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Máy in nhãn được thay đầu in miễn phí trong mười tám tháng nếu lỗi từ nhà sản xuất.

Lý do đề xuất: Cam kết bảo hành có điều kiện cần chính sách và phạm vi linh kiện.

Chú ý: Cam kết thay linh kiện miễn phí cần chính sách, dù đã nêu lỗi từ nhà sản xuất.

**CV03_048 — CANH_BAO — CLEAN**

Chính sách loại trừ đầu in nhưng em xóa dòng loại trừ trên ảnh gửi khách để họ nghĩ thay đầu in cũng miễn phí.

Lý do đề xuất: Cố ý che điều khoản bảo hành khiến người mua hiểu sai quyền lợi.

### CV03_F017 — GIAO_HANG_CAY

**CV03_049 — KHONG_CANH_BAO — CLEAN**

Trước khi gửi chậu cây em sẽ nhắn chị kiểm tra lại số nhà, tránh để bên giao tìm sai địa chỉ.

Lý do đề xuất: Mô tả bước xác nhận địa chỉ, không hứa tốc độ hay bảo đảm nhận hàng.

**CV03_050 — CAN_XAC_MINH — CLEAN** — CẦN CHÚ Ý

Chậu cây đặt trước trưa sẽ đến trong ngày ở các phường có trong danh sách giao của shop.

Lý do đề xuất: Cam kết giao hàng theo thời gian và khu vực cần kiểm tra điều kiện vận hành.

Chú ý: Đã giới hạn khu vực nhưng vẫn hứa giao trong ngày; cần kiểm tra khả năng thực hiện.

**CV03_051 — CANH_BAO — CLEAN**

Xe chở cây hôm nay đã hủy chuyến mà em vẫn gửi ảnh giao hàng của hôm qua để báo đơn mới đang tới nơi.

Lý do đề xuất: Dùng bằng chứng giao hàng cũ để nói sai trạng thái đơn mới.

### CV03_F018 — GIOI_HAN_MAY_XONG

**CV03_052 — KHONG_CANH_BAO — ASR_LIKE** — CẦN CHÚ Ý, DUYỆT CẶP ASR

Máy xông mặt này dùng theo hướng dẫn về khoản cách, nếu thấy rát thì dừng lại chứ đừng cố dùng tiếp.

Lý do đề xuất: Nêu giới hạn và khuyên dừng khi khó chịu, không cam kết hiệu quả điều trị.

Câu sạch cùng nhãn/family: Máy xông mặt này dùng theo hướng dẫn về khoảng cách, nếu thấy rát thì dừng lại chứ đừng cố dùng tiếp.

Nhiễu ASR giả lập: khoảng cách → khoản cách; không đổi dấu hiệu quyết định nhãn.

Chú ý: Khuyên dừng khi rát là giới hạn an toàn, không phải lời chữa bệnh.

**CV03_053 — CAN_XAC_MINH — CLEAN**

Vòi máy xông giữ hơi ở bốn mươi độ C tại khoảng cách hai mươi xăng ti mét, mình cần xem cách hãng đo.

Lý do đề xuất: Thông số nhiệt tại vị trí xác định cần đối chiếu phép đo.

**CV03_054 — CANH_BAO — CLEAN**

Cứ dí sát mặt vào vòi hơi đang bỏng cũng không sao, em bảo đảm da ai bị viêm cũng khỏi sau một lần xông.

Lý do đề xuất: Khuyến khích tiếp xúc hơi bỏng và hứa chữa bệnh tuyệt đối, có nguy cơ rõ ràng.

### CV03_F019 — THAO_TAC_HOP_KIM_CHI

**CV03_055 — KHONG_CANH_BAO — ASR_LIKE** — DUYỆT CẶP ASR

Em kéo ngăn hộp kim chĩ ra để chị thấy các cuộn xếp đứng, lát nữa em đậy nắp lại cho dễ quan sát.

Lý do đề xuất: Thao tác giới thiệu cách sắp xếp đồ, không tuyên bố cần bằng chứng.

Câu sạch cùng nhãn/family: Em kéo ngăn hộp kim chỉ ra để chị thấy các cuộn xếp đứng, lát nữa em đậy nắp lại cho dễ quan sát.

Nhiễu ASR giả lập: kim chỉ → kim chĩ; không đổi dấu hiệu quyết định nhãn.

**CV03_056 — CAN_XAC_MINH — CLEAN**

Chốt hộp kim chỉ chịu được năm nghìn lần đóng mở trong bài thử của xưởng, em gửi bảng đếm để mình đối chiếu.

Lý do đề xuất: Độ bền định lượng cần đối chiếu phương pháp và bảng thử.

**CV03_057 — CANH_BAO — CLEAN**

Chốt hộp đã gãy sẵn nên em giữ bằng tay lúc quay, không cho khách thấy để vẫn bán như hộp nguyên vẹn.

Lý do đề xuất: Cố ý che bộ phận bị hỏng trong video bán hàng.

### CV03_F020 — CANH_BAO_SU_DUNG_DEN

**CV03_058 — KHONG_CANH_BAO — CLEAN**

Bóng đèn này cần ngắt điện trước khi tháo, em chờ công tắc tắt rồi mới chỉ cách xoay đui.

Lý do đề xuất: Hướng dẫn thao tác thận trọng, không cam kết tính năng điện.

**CV03_059 — CAN_XAC_MINH — CLEAN**

Bóng này cho độ sáng một nghìn lumen theo phiếu đo của hãng, cần đọc cả công suất và điều kiện đo.

Lý do đề xuất: Thông số quang học cần đối chiếu chứng từ và điều kiện thử.

**CV03_060 — CANH_BAO — CLEAN**

Đui đèn đã rò điện mà em quấn băng che vết cháy rồi vẫn nói khách cứ chạm lúc bật để thử độ chắc.

Lý do đề xuất: Biết lỗi điện nhưng che giấu và hướng dẫn nguy hiểm.
