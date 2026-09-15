import re
import unicodedata



RULES = {
    "vi_the_tuyet_doi": {
        "label": "Khẳng định vị thế cần xác minh",
        "keywords": [
            "top 1",
            "top một",
            "tóp một",
            "số một",
            "số 1",
            "bán chạy nhất",
            "tốt nhất thế giới",
            "không có đối thủ",
            "rẻ nhất sàn",
            "rẻ nhất toàn",
        ],
    },

    "cam_ket_cong_dung": {
        "label": "Cam kết công dụng cần kiểm tra",
        "keywords": [
            "chữa khỏi hoàn toàn",
            "chữa khỏi bệnh",
            "chữa được bệnh",
            "trị dứt điểm",
            "khỏi bệnh 100%",
            "bao khỏi",
            "khỏi hẳn",
            "thay được toàn bộ thuốc",
            "bỏ thuốc bác sĩ kê",
        ],
    },

    "cam_ket_hieu_qua": {
        "label": "Cam kết hiệu quả cần kiểm tra",
        "keywords": [
            "đảm bảo giảm 10kg",
            "trắng da sau một lần",
            "trắng vĩnh viễn",
            "bay sạch mỡ",
            "giảm mười ký",
            "giảm 10kg",
            "xóa sạch mọi nếp nhăn",
            "tóc mọc phủ kín đầu",
            "không có tác dụng phụ",
            "an toàn tuyệt đối",
        ],
    },

    "gia_va_khuyen_mai": {
        "label": "Thông tin giá và ưu đãi cần xác minh",
        "keywords": [
            "giá sập sàn",
            "giảm chín mươi phần trăm",
            "giảm 90%",
            "lỗi giá",
            "nhập nhầm giá",
            "freeship toàn quốc",
            "freeship không điều kiện",
            "chắc chắn có quà",
        ],
    },
}
    
def normalize_text(text):
    """Chuẩn hóa Unicode, chữ hoa/thường và khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


class KeywordDetector:
    def __init__(self):
        self.patterns = []

        for category, rule in RULES.items():
            for keyword in rule["keywords"]:
                # Cho phép một hoặc nhiều khoảng trắng giữa các từ.
                phrase = r"\s+".join(
                    re.escape(word)
                    for word in normalize_text(keyword).split()
                )

                # Tránh khớp một phần nằm bên trong từ khác.
                pattern = re.compile(r"(?<!\w)" + phrase + r"(?!\w)")

                self.patterns.append(
                    (category, rule["label"], keyword, pattern)
                )

    def detect(self, text):
        findings = []

        # Tách câu đơn giản theo dấu câu hoặc xuống dòng.
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            normalized = normalize_text(sentence)

            for category, label, keyword, pattern in self.patterns:
                if pattern.search(normalized):
                    findings.append({
                        "category": category,
                        "label": label,
                        "keyword": keyword,
                        "sentence": sentence,
                    })

        return {
            "needs_review": bool(findings),
            "findings": findings,
        }


if __name__ == "__main__":
    detector = KeywordDetector()

    examples = [
        "Áo này chất cotton, giá 150 nghìn đồng.",
        "Sản phẩm này chữa khỏi hoàn toàn và trị dứt điểm.",
        "Không nên tin quảng cáo chữa khỏi hoàn toàn.",
    ]

    for text in examples:
        result = detector.detect(text)

        print(f"\nNội dung: {text}")

        if not result["needs_review"]:
            print("Kết quả: Không tìm thấy cụm từ trong bộ quy tắc.")
            continue

        print("Kết quả: Có nội dung cần kiểm tra.")

        for finding in result["findings"]:
            print(f"  - Nhóm: {finding['label']}")
            print(f"    Cụm từ: {finding['keyword']}")
            print(f"    Câu: {finding['sentence']}")