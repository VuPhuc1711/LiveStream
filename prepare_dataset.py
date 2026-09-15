import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "data" / "dataset_livestream_v01.csv"
    output_dir = project_dir / "data" / "splits"

    with input_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    allowed_labels = {
        "CANH_BAO",
        "CAN_XAC_MINH",
        "KHONG_CANH_BAO",
    }
    allowed_statuses = {"DA_DUYET", "DA_DUYET_AI"}

    # Kiểm tra dữ liệu trước khi chia.
    for row in rows:
        if row["label"] not in allowed_labels:
            raise ValueError(f"Nhãn không hợp lệ: {row['id']}")

        if row["review_status"] not in allowed_statuses:
            raise ValueError(f"Câu chưa được rà soát: {row['id']}")

        if not row["text"].strip() or not row["family_id"].strip():
            raise ValueError(f"Thiếu text hoặc family_id: {row['id']}")

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset có id bị trùng.")

    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)

    # Gom các họ câu theo nhóm nội dung.
    categories = defaultdict(list)
    for family_id, items in families.items():
        category_names = {row["category"] for row in items}

        if len(category_names) != 1:
            raise ValueError(f"Họ câu có nhiều category: {family_id}")

        categories[items[0]["category"]].append(family_id)

    # Với bộ hiện tại: mỗi category có 5 họ, mỗi họ 3 câu.
    if len(rows) != 150 or len(categories) != 10:
        raise ValueError("Script này dành cho bộ khởi đầu 150 câu.")

    if any(len(ids) != 5 for ids in categories.values()):
        raise ValueError("Mỗi category cần có đúng 5 họ câu.")

    if any(len(items) != 3 for items in families.values()):
        raise ValueError("Mỗi họ cần có đúng 3 câu.")

    rng = random.Random(42)
    splits = {"train": [], "validation": [], "test": []}

    for category in sorted(categories):
        family_ids = sorted(categories[category])
        rng.shuffle(family_ids)

        allocation = {
            "train": family_ids[:3],
            "validation": family_ids[3:4],
            "test": family_ids[4:],
        }

        for split_name, selected_ids in allocation.items():
            for family_id in selected_ids:
                for row in families[family_id]:
                    splits[split_name].append({
                        **row,
                        "split": split_name,
                    })

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        rng.shuffle(items)
        output_path = output_dir / f"{split_name}.csv"

        with output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)

        print(f"\n{split_name}: {len(items)} câu")
        print("Nhãn:", dict(Counter(row["label"] for row in items)))
        print("Đã lưu:", output_path)


if __name__ == "__main__":
    main()