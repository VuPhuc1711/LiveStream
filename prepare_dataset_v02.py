"""Chia v02 theo family_id và kiểm tra tập cuối v03, không chạy model.

Chạy: python prepare_dataset_v02.py
Kiểm tra lại mà không ghi file: python prepare_dataset_v02.py --check-only
Chỉ dùng thư viện chuẩn Python. Không nhập module Rules/PhoBERT/Hybrid.
"""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "dataset_livestream_v02.csv"
FINAL_TEST = ROOT / "data" / "final_test_v03_chua_duyet.csv"
SPLIT_DIR = ROOT / "data" / "splits_v02"
REPORT_DIR = ROOT / "reports" / "data_preparation_v02_v03"
SEED = 42
LABELS = ("KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO")
FINAL_COLUMNS = ["id", "text", "label", "category", "reason", "family_id", "review_status", "needs_context"]
# Đóng dấu hai phiên bản đã được đọc để các nhận xét định tính không bị tái dùng
# âm thầm khi người dùng sửa nguồn hoặc duyệt lại v03 ở giai đoạn sau.
REVIEWED_HASHES = {
    SOURCE: "89be61f10e6c7b048b382c39b403c22bc48f57c463dc9b0a7cc40e978bb00d2e",
    FINAL_TEST: "7b6114a9e7d6e569fd9e28d09e3f655becd8297fed7771e5704741766523b253",
}

# Rà soát ngữ nghĩa định tính của trợ lý trên toàn bộ 30 câu v03 và 150 câu v02.
# Các mức dưới đây không phải xác suất, cosine similarity hoặc kết quả model.
# Không dùng nhận xét, nhãn hay nội dung v03 trong thuật toán chia v02.
SEMANTIC_REVIEW = (
    (1, "CUNG_CHU_DE", "LS0119", "Cùng mô tả định lượng thông thường; số lượng ly/dung tích khác khối lượng túi. Chưa xem là diễn đạt lại cùng tình huống."),
    (2, "CHUA_THAY_CAP_GAN", "", "Chưa thấy câu v02 tương đương về sai khác màu do ánh sáng; câu nêu sở thích màu có ý nghĩa khác."),
    (3, "CUNG_CHU_DE", "LS0054", "Cùng giới hạn phát biểu ai cũng dùng/phù hợp và nhắc xem hướng dẫn; v03 xét kích thước, v02 xét đối tượng sử dụng."),
    (4, "CHUA_THAY_CAP_GAN", "", "Chưa thấy câu v02 tương đương với danh sách size S/M/L và bảng số đo."),
    (5, "CUNG_CHU_DE", "LS0071", "Cùng báo giá một sản phẩm; khác giá và cách dẫn vị trí trên giỏ. Đây là mẫu câu thông thường, chưa coi là gần nghĩa cao."),
    (6, "CUNG_CHU_DE", "LS0051", "Cùng thận trọng với kích ứng; v03 khuyên đọc thành phần trước mua, v02 khuyên ngừng dùng khi kích ứng và hỏi chuyên môn."),
    (7, "CUNG_CHU_DE", "LS0117", "Cùng làm rõ phụ kiện trong giỏ; v03 hướng dẫn xem phân loại, v02 xác nhận pin/sạc bán riêng. Khác thông tin được khẳng định."),
    (8, "CHUA_THAY_CAP_GAN", "", "Chưa thấy tình huống v02 tương đương với xác nhận đơn trước giao mà không hứa thời điểm giao."),
    (9, "CAO", "LS0033", "Cùng giải thích mỹ phẩm chỉ làm sáng/nâng tông tạm thời và giới hạn công dụng thật; v03 thêm phủ định điều trị da."),
    (10, "CUNG_CHU_DE", "LS0012", "Cùng nói về mẫu đang trưng bày/cầm trên tay; v03 dẫn tới tồn kho trên giỏ, v02 chỉ nói một chiếc dùng quay cận cảnh."),
    (11, "CAO", "LS0107", "Cùng tuyên bố áo 100% cotton/full cotton; v03 thêm yêu cầu đối chiếu nhãn phụ, cần cân nhắc ảnh hưởng của lời thận trọng."),
    (12, "CHUA_THAY_CAP_GAN", "", "Chưa thấy câu v02 tương đương về giữ nóng 12 giờ và thông số thử nghiệm."),
    (13, "CUNG_CHU_DE", "LS0146", "Cùng so sánh hiệu năng giữa hai mẫu, nhưng tiết kiệm điện có tỷ lệ khác so sánh tốc độ trong bài thử."),
    (14, "CUNG_CHU_DE", "LS0050", "Cùng thông tin phù hợp/không kích ứng kèm đề nghị kiểm tra. Nhãn nguồn khác nhau; cần người dùng rà soát ranh giới phát biểu, không tự đổi nhãn."),
    (15, "CUNG_CHU_DE", "LS0125", "Cùng lời hứa bảo hành; hai năm kèm điều kiện khác bảo hành trọn đời, không coi là cùng mệnh đề."),
    (16, "CUNG_CHU_DE", "LS0131", "Cùng lời hứa thời gian giao; v03 có nội thành, có thể và khoảng hai giờ, khác lời chốt tối nay mai nhận."),
    (17, "CAO", "LS0092", "Cùng khẳng định hàng chính hãng/auth; v03 thêm cam kết gửi hóa đơn hoặc thông tin phân phối."),
    (18, "CUNG_CHU_DE", "LS0065", "Cùng quảng cáo phần trăm giảm giá; v03 nêu mức tối đa và đề nghị kiểm tra điều kiện, khác lời giá sập sàn giảm 90%."),
    (19, "CAO", "LS0113", "Cùng khẳng định khả năng chống nước trong điều kiện sử dụng; v03 đổi đồng hồ đi bơi thành bộ sạc ngoài trời và thêm hoàn toàn. Khác phạm vi tuyệt đối cần người duyệt lưu ý."),
    (20, "RAT_CAO", "LS0101", "Gần diễn đạt lại cùng tuyên bố sản phẩm được bác sĩ khuyên dùng; v03 thêm nguồn nhãn hàng và chuyên khoa da liễu."),
    (21, "RAT_CAO", "LS0097", "Cùng tự thừa nhận thiếu chứng nhận/kiểm định nhưng vẫn quảng cáo chuẩn quốc tế."),
    (22, "CAO", "LS0019", "Cùng cam kết sau ba ngày khỏi hẳn bệnh; đổi sản phẩm và bệnh, v03 thêm không cần gặp bác sĩ."),
    (23, "CAO", "LS0064", "Cùng thừa nhận dùng giá gốc/niêm yết không thật để tạo giảm giá giả; khác con số và cách mô tả lịch sử giá."),
    (24, "RAT_CAO", "LS0106", "Cùng khuôn tình huống biết chất liệu thật nhưng vẫn nói 100% một chất liệu khác; thay sợi pha/cotton bằng polyester/lụa."),
    (25, "CUNG_CHU_DE", "LS0085", "Cùng tự thừa nhận tạo áp lực mua; v03 giả số lượng tồn kho, v02 giả chương trình/thời gian Flash Sale, khác cơ chế cụ thể."),
    (26, "RAT_CAO", "LS0022", "Cùng khuyên bỏ thuốc đang dùng để thay bằng viên bổ/sản phẩm khác và cam kết thay điều trị."),
    (27, "RAT_CAO", "LS0130", "Cùng biết không đáp ứng giao hàng nhưng vẫn hứa chắc mọi người nhận sáng mai."),
    (28, "CHUA_THAY_CAP_GAN", "", "Chưa thấy v02 tự thừa nhận tạo đánh giá năm sao giả; các câu bàn bình luận khách và bằng chứng có hành vi khác."),
    (29, "RAT_CAO", "LS0124", "Gần diễn đạt lại tình huống bảo hành hãng thực tế sáu tháng nhưng quảng cáo trọn đời; khác chủ thể được nêu trong lời quảng cáo."),
    (30, "CAO", "LS0139", "Cùng cáo buộc đối thủ bán hàng giả để kéo khách; v03 biết đối thủ có giấy tờ, v02 thừa nhận chưa kiểm tra."),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    require(columns and len(columns) == len(set(columns)), f"Thiếu/trùng tên cột: {path}")
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()), f"Sai số cột: {path}")
        require(all(row.get(key, "").strip() for key in ("id", "text", "label", "family_id")), f"Dòng thiếu dữ liệu: {path}")
        require(row["label"] in LABELS, f"Nhãn không hợp lệ: {row['id']}")
    require(len({row["id"] for row in rows}) == len(rows), f"ID trùng: {path}")
    require(len({normalize_text(row["text"]) for row in rows}) == len(rows), f"Nội dung chuẩn hóa trùng: {path}")
    return columns, rows


def check_disjoint(left, right):
    counts = {key: len({row[key] for row in left} & {row[key] for row in right}) for key in ("id", "family_id")}
    counts["normalized_text"] = len(
        {normalize_text(row["text"]) for row in left} & {normalize_text(row["text"]) for row in right}
    )
    require(not any(counts.values()), f"Phát hiện giao nhau: {counts}")
    return counts


def select_splits(rows, seed=SEED):
    """Chỉ nhận v02; quy hoạch động chọn 1/5 họ của mỗi category vào validation."""
    require(len(rows) == 150, "V02 phải có đúng 150 câu.")
    families, categories = defaultdict(list), defaultdict(list)
    for row in rows:
        require(row["review_status"] in {"DA_DUYET", "DA_DUYET_AI"}, f"Câu v02 chưa duyệt: {row['id']}")
        families[row["family_id"]].append(row)
    require(len(families) == 50 and all(len(items) == 3 for items in families.values()), "Cần 50 họ, mỗi họ 3 câu.")
    for family, items in sorted(families.items()):
        require(len({item["category"] for item in items}) == 1, f"Họ {family} có nhiều category.")
        categories[items[0]["category"]].append(family)
    require(len(categories) == 10 and all(len(ids) == 5 for ids in categories.values()), "Cần 10 category, mỗi category 5 họ.")
    totals = Counter(row["label"] for row in rows)
    require(set(totals) == set(LABELS), "V02 thiếu nhãn.")
    rng = random.Random(seed)
    states = {(0, 0, 0): ()}
    for category in sorted(categories):
        candidates = sorted(categories[category])
        rng.shuffle(candidates)
        following = {}
        for counts, selected in states.items():
            for family in candidates:
                frequencies = Counter(row["label"] for row in families[family])
                combined = tuple(counts[i] + frequencies[label] for i, label in enumerate(LABELS))
                following.setdefault(combined, selected + (family,))
        states = following

    def imbalance(counts):
        # Ưu tiên lệch bình phương tương đối theo tần suất lớp; không dùng model.
        return sum((Fraction(count * 5 - totals[label]) ** 2 / totals[label]
                    for count, label in zip(counts, LABELS)), Fraction(0))

    feasible = [counts for counts in states if all(0 < count < totals[label] for count, label in zip(counts, LABELS))]
    require(feasible, "Không tìm được cách chia đủ nhãn.")
    best_counts = min(feasible, key=lambda counts: (imbalance(counts), counts))
    validation_families = set(states[best_counts])
    splits = {name: [] for name in ("train", "validation")}
    for row in rows:
        name = "validation" if row["family_id"] in validation_families else "train"
        splits[name].append({**row, "split": name})
    for items in splits.values():
        rng.shuffle(items)
    return splits, {"objective": "sum((5 * validation_count - source_count)^2 / source_count) over labels",
                    "optimal_objective_fraction": str(imbalance(best_counts)),
                    "feasible_label_count_combinations": len(feasible),
                    "category_constraint": "Exactly one validation family and four train families per category",
                    "tie_break": "Seeded shuffle of family candidates; first path for identical label counts"}


def validate_splits(source_rows, splits):
    original = {row["id"]: row for row in source_rows}
    for name, expected_count in (("train", 120), ("validation", 30)):
        rows = splits[name]
        require(len(rows) == expected_count and len({row["id"] for row in rows}) == expected_count, f"Sai số câu/ID ở {name}.")
        require(set(row["label"] for row in rows) == set(LABELS), f"Thiếu nhãn ở {name}.")
        for row in rows:
            require(row == {**original[row["id"]], "split": name}, f"Thay đổi nội dung nguồn: {row['id']}")
    overlap = check_disjoint(splits["train"], splits["validation"])
    require({row["id"] for items in splits.values() for row in items} == set(original), "Thiếu/thừa câu v02.")
    return overlap


def validate_final(columns, rows):
    require(columns == FINAL_COLUMNS and len(rows) == 30, "V03 sai cấu trúc hoặc số câu.")
    for index, row in enumerate(rows, 1):
        require(row["id"] == f"FTV03_{index:03}" and row["family_id"] == f"FTV03_F{index:02}", "Sai ID/thứ tự/family v03.")
        require(row["label"] == LABELS[(index - 1) // 10], f"Sai nhãn được cung cấp: {row['id']}")
        require(row["category"] == "FINAL_TEST_V03" and row["review_status"] == "CHUA_DUYET"
                and row["needs_context"] == "no" and row["reason"].strip(), f"Sai metadata v03: {row['id']}")


def summary(rows):
    return {"rows": len(rows), "families": len({row["family_id"] for row in rows}),
            "labels": {label: sum(row["label"] == label for row in rows) for label in LABELS},
            "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
            "review_status": dict(sorted(Counter(row["review_status"] for row in rows).items()))}


def semantic_rows(source_rows, final_rows):
    source_by_id = {row["id"]: row for row in source_rows}
    require([item[0] for item in SEMANTIC_REVIEW] == list(range(1, 31)), "Thiếu câu rà soát ngữ nghĩa.")
    return [
        {"v03_id": final_rows[index - 1]["id"], "v03_text": final_rows[index - 1]["text"],
         "v03_label": final_rows[index - 1]["label"], "similarity_level": level,
         "v02_id": other_id, "v02_text": source_by_id[other_id]["text"] if other_id else "",
         "v02_label": source_by_id[other_id]["label"] if other_id else "",
         "reason": reason, "needs_user_review": "yes" if level in {"CAO", "RAT_CAO"} else "optional",
         "review_method": "Codex qualitative content comparison; no embedding/classifier execution"}
        for index, level, other_id, reason in SEMANTIC_REVIEW
    ]


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    read_columns, read_rows = read_csv(path) if "id" in columns else (None, None)
    if read_columns is not None:
        require(read_columns == columns and read_rows == rows, f"CSV không khớp sau ghi: {path}")


def render_report(report, reviews):
    lines = ["# Chuẩn bị v02 và tập cuối v03 chưa duyệt", "",
             "V02 chia theo family_id, seed 42. Giữ nguyên 21 cột; chỉ cập nhật cột split trong hai file đầu ra.",
             "Không dùng v03 để chọn split, huấn luyện, chọn epoch, sửa Rules hoặc ngưỡng.", "",
             "| Tập | Câu | Họ câu | KHONG_CANH_BAO | CAN_XAC_MINH | CANH_BAO |", "|---|---:|---:|---:|---:|---:|"]
    for name, details in report["datasets"].items():
        lines.append(f"| {name} | {details['rows']} | {details['families']} | "
                     + " | ".join(str(details["labels"][label]) for label in LABELS) + " |")
    lines += ["", "Mỗi category v02 có 12 câu train và 3 câu validation. Không tạo test.csv trong splits_v02.",
              "Cách chọn họ: quy hoạch động tìm phân bố nhãn gần nguồn nhất theo sai lệch bình phương có trọng số, trong ràng buộc một họ validation mỗi category. Seed xử lý các lựa chọn tương đương.",
              "", "## Kiểm tra", "",
              "- Đủ 150 ID nguồn, không mất/thêm câu; mỗi câu thuộc đúng một split.",
              "- Không giao ID, text chuẩn hóa hoặc family_id giữa train và validation.",
              "- Không giao ID, text chuẩn hóa hoặc family_id giữa v03 và v02/các split.",
              "- Chuẩn hóa khi so trùng: Unicode NFC, casefold và gộp khoảng trắng; giữ dấu và dấu câu.",
              "- V03 có đúng 8 cột, 30 ID và 30 family_id riêng; 10 câu mỗi nhãn, toàn bộ CHUA_DUYET.",
              "- CSV UTF-8 được đọc lại và đối chiếu đầy đủ cột, thứ tự, giá trị.",
              "- SHA-256 của nguồn v02, v01, splits cũ, checkpoint và mã nguồn cũ không đổi.",
              "- Chưa chạy train, suy luận hoặc đánh giá model.",
              "", "## Rà soát gần nghĩa", "",
              "Đây là đối chiếu nội dung định tính do trợ lý thực hiện trên 30 câu v03 và 150 câu v02; không phải điểm embedding hay phép đo tự động về ngữ nghĩa. Các nhận xét vẫn cần người dùng duyệt.",
              "Không trùng chữ/ID không đồng nghĩa độc lập hoàn toàn về nội dung. Cột CUNG_CHU_DE chỉ cùng chủ đề với khác biệt đáng kể, không được tính là gần nghĩa cao.", "",
              f"Có {report['semantic_review']['high_or_very_high_count']} câu v03 được đề nghị xem lại vì gần nghĩa cao/rất cao.", "",
              "| V03 | V02 gần nghĩa | Mức | Lý do xem lại |", "|---|---|---|---|"]
    for row in reviews:
        if row["similarity_level"] in {"CAO", "RAT_CAO"}:
            lines.append(f"| {row['v03_id']} | {row['v02_id']} | {row['similarity_level']} | {row['reason']} |")
    lines += ["", "Xem semantic_review.csv để đọc nguyên văn hai phía và nhận xét cho đủ 30 câu, gồm cả những cặp chỉ cùng chủ đề.",
              "", "## Điều kiện sử dụng tập cuối", "",
              "1. Người dùng duyệt dữ liệu v03 trước khi đánh giá; hiện vẫn CHUA_DUYET.",
              "2. Chỉ train và chọn epoch bằng train/validation trong data/splits_v02.",
              "3. Chốt checkpoint v02 và cấu hình đánh giá trước khi mở đánh giá tập cuối.",
              "4. Không dùng v03 để chỉnh Rules, ngưỡng Hybrid hoặc PhoBERT. Việc rà soát ở đây không phải đánh giá model.",
              "5. Các câu gần nghĩa vẫn được giữ nguyên theo yêu cầu; chưa chứng nhận v03 độc lập về ngữ nghĩa.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for path, expected_hash in REVIEWED_HASHES.items():
        require(sha256(path) == expected_hash, f"Nguồn đã đổi; cần rà soát lại trước khi chạy: {path}")
    columns, source_rows = read_csv(SOURCE)
    require(len(columns) == 21 and "split" in columns, "V02 phải giữ đủ 21 cột.")
    splits, selection = select_splits(source_rows)
    split_overlap = validate_splits(source_rows, splits)
    # Đọc v03 sau khi chọn split; dữ liệu này không tham gia quyết định chia.
    final_columns, final_rows = read_csv(FINAL_TEST)
    validate_final(final_columns, final_rows)
    overlaps = {"train_vs_validation": split_overlap, "v03_vs_v02": check_disjoint(final_rows, source_rows)}
    overlaps.update({f"v03_vs_{name}": check_disjoint(final_rows, items) for name, items in splits.items()})
    split_files = [SPLIT_DIR / f"{name}.csv" for name in splits]
    if args.check_only:
        require({path.name for path in SPLIT_DIR.iterdir()} == {"train.csv", "validation.csv"}, "splits_v02 phải chỉ có hai file.")
        for name in splits:
            actual_columns, actual_rows = read_csv(SPLIT_DIR / f"{name}.csv")
            require(actual_columns == columns and actual_rows == splits[name], f"Không tái lập được split {name} theo seed 42.")
        report = json.loads((REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        for relative, digest in report["protected_file_sha256"].items():
            require(sha256(ROOT / relative) == digest, f"File được bảo vệ đã thay đổi: {relative}")
        print("PASS: tái lập đúng seed 42; đủ dòng, nhãn, cột, họ câu; không rò rỉ và file cũ không đổi.")
        return
    require(not SPLIT_DIR.exists() and not REPORT_DIR.exists(), "Thư mục đích đã có; dùng --check-only để kiểm tra, không ghi đè.")
    protected_paths = sorted(set(
        [SOURCE, FINAL_TEST, ROOT / "data" / "dataset_livestream_v01.csv", ROOT / "data" / "dataset_livestream_v02.xlsx"]
        + list((ROOT / "data" / "splits").glob("*.csv"))
        + [path for path in (ROOT / "models").rglob("*") if path.is_file()]
        + [path for path in ROOT.glob("*.py") if path != Path(__file__).resolve()]
        + list((ROOT / "nlp").glob("*.py")) + list((ROOT / "speech").glob("*.py"))
        + [ROOT / "README.md", ROOT / "requirements.txt"]
    ))
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected_paths}
    reviews = semantic_rows(source_rows, final_rows)
    SPLIT_DIR.mkdir(parents=True)
    for name, items in splits.items():
        write_csv(SPLIT_DIR / f"{name}.csv", columns, items)
    require({path.name for path in SPLIT_DIR.iterdir()} == {"train.csv", "validation.csv"}, "Có file không mong muốn trong splits_v02.")
    require(before == {str(path.relative_to(ROOT)): sha256(path) for path in protected_paths}, "File được bảo vệ đã thay đổi.")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "source": str(SOURCE), "source_sha256": before[str(SOURCE.relative_to(ROOT))],
        "selection": selection, "normalization": "NFC + casefold + collapse whitespace; preserve accents/punctuation",
        "datasets": {"v02_source": summary(source_rows), **{name: summary(items) for name, items in splits.items()},
                     "final_test_v03_chua_duyet": summary(final_rows)},
        "split_ids": {name: [row["id"] for row in items] for name, items in splits.items()},
        "split_family_ids": {name: sorted({row["family_id"] for row in items}) for name, items in splits.items()},
        "overlap_counts": overlaps, "preserved_columns": columns,
        "output_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in split_files + [FINAL_TEST]},
        "protected_file_sha256": before,
        "semantic_review": {"method": "Qualitative assistant review, not numerical similarity or model inference",
                            "reviewed_v03_rows": len(reviews), "source_v02_rows": len(source_rows),
                            "levels": dict(Counter(row["similarity_level"] for row in reviews)),
                            "high_or_very_high_count": sum(row["similarity_level"] in {"CAO", "RAT_CAO"} for row in reviews),
                            "used_to_select_splits": False},
        "final_test_policy": {"status": "CHUA_DUYET", "training_allowed": False,
                              "validation_or_epoch_selection_allowed": False, "threshold_or_rules_tuning_allowed": False,
                              "evaluation_performed": False,
                              "evaluation_requires": ["User review of v03", "v02 training completed", "v02 checkpoint and evaluation configuration frozen"]},
    }
    REPORT_DIR.mkdir(parents=True)
    write_csv(REPORT_DIR / "semantic_review.csv", list(reviews[0]), reviews)
    (REPORT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "README.md").write_text(render_report(report, reviews), encoding="utf-8")
    print(json.dumps({"datasets": report["datasets"], "overlaps": overlaps, "semantic_review": report["semantic_review"],
                      "protected_files_unchanged": len(before), "report_dir": str(REPORT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
