import csv
from pathlib import Path

from nlp.keyword_detector import KeywordDetector


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "data" / "dataset_livestream_v01.csv"

    with dataset_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    stats = {
        "CANH_BAO": {"total": 0, "flagged": 0},
        "CAN_XAC_MINH": {"total": 0, "flagged": 0},
        "KHONG_CANH_BAO": {"total": 0, "flagged": 0},
    }

    # Kiểm tra dữ liệu trước khi đánh giá.
    for row in rows:
        if row["label"] not in stats:
            raise ValueError(f"Nhãn không hợp lệ: {row['label']}")
        if not row["text"].strip():
            raise ValueError(f"Câu bị trống: {row['id']}")

    detector = KeywordDetector()
    missed = []
    false_alarms = []

    for row in rows:
        label = row["label"]
        result = detector.detect(row["text"])
        flagged = result["needs_review"]

        stats[label]["total"] += 1
        stats[label]["flagged"] += int(flagged)

        # Mục tiêu sàng lọc: đưa cả hai nhóm này ra kiểm tra.
        should_review = label in {"CANH_BAO", "CAN_XAC_MINH"}

        if should_review and not flagged:
            missed.append(row)

        if not should_review and flagged:
            keywords = [
                finding["keyword"]
                for finding in result["findings"]
            ]
            false_alarms.append((row, keywords))

    print("\n===== KẾT QUẢ THEO NHÃN =====")

    for label, counts in stats.items():
        total = counts["total"]
        flagged = counts["flagged"]
        print(
            f"{label}: {total} câu | "
            f"Đánh dấu: {flagged} | "
            f"Không đánh dấu: {total - flagged}"
        )

    print(f"\nBỏ sót câu cần kiểm tra: {len(missed)}")
    print(f"Cảnh báo nhầm: {len(false_alarms)}")

    print("\n===== TỐI ĐA 5 CÂU BỊ BỎ SÓT =====")
    for row in missed[:5]:
        print(f"\n{row['id']} | Nhãn: {row['label']}")
        print(row["text"])

    print("\n===== TẤT CẢ CÂU CẢNH BÁO NHẦM =====")
    for row, keywords in false_alarms:
        print(f"\n{row['id']} | Nhãn: {row['label']}")
        print(row["text"])
        print("Cụm từ khớp:", ", ".join(keywords))

    print("\nLưu ý: Kết quả so với nhãn mô phỏng v0.1 chưa duyệt.")


if __name__ == "__main__":
    main()