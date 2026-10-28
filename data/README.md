# Dữ liệu train/test — 28/10/2026

Chỉ hai phần CSV hiện hành, tổng766 câu đã duyệt:

| File | Không cảnh báo | Cảnh báo | Tổng |
| --- | ---: | ---: | ---: |
| `current/train.csv` |420|164|584|
| `current/test.csv` |134|48|182|

584 câu train giữ nguyên. Toàn bộ182 câu validation trước đây chuyển sang test
theo xác nhận người dùng, giữ text/nhãn/ID/family và cùng membership. Các trường
`original_split`, `evaluation_role`, `seen_by_development` giữ lịch sử nguồn;
chúng không phải nhãn phân loại mới. Không có `validation.csv` trong bộ hiện hành.

**Test này đã từng dùng chọn binary_v04**, không phải test độc lập; chỉ đối chiếu
hồi quy. Chưa train lại; checkpoint vẫn là binary_v04 đã train theo quy trình cũ.
Không chọn epoch/tham số bằng test khi có một lượt train mới được phép.

Manifest/hash/nguồn: `config/data/current_dataset.json`. Toàn bộ bản gốc binary_v01
đến v04, splits, test/holdout và quyết định duyệt cũ giữ nguyên tại
`D:\lưu trữ\livestream-violation-detection_20261028\data`.
Không tự phục hồi17 câu nguy hiểm hoặc54 câu không thực tế đã loại.
Mẫu feedback mới vẫn riêng trong PostgreSQL/runtime/data-exports, chưa ghép vào
766 câu này và chưa được dùng train. Không tự mở binary_v05.
