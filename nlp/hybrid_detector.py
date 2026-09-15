"""Kết hợp Rules và PhoBERT theo chính sách ưu tiên của Giai đoạn 2."""

from functools import lru_cache
from typing import TypedDict

from nlp.keyword_detector import KeywordDetector
from nlp.phobert_detector import PhoBertDetector, Prediction


WARNING_THRESHOLD = 0.55
SAFE_THRESHOLD = 0.65


class HybridPrediction(TypedDict):
    final_label: str
    final_confidence: float
    phobert_result: Prediction
    rule_result: dict
    explanation: str


class HybridDetector:
    """Nạp hai bộ phát hiện một lần rồi dùng predict(text) cho từng câu.

    final_confidence = phobert_result['probabilities'][final_label].
    Đây là xác suất PhoBERT của nhãn cuối, chưa được hiệu chỉnh cho hybrid.
    Khi chuyển sang CAN_XAC_MINH theo chính sách, giá trị này có thể thấp hơn
    confidence của nhãn PhoBERT ban đầu; không tự tạo điểm tin cậy từ Rules.
    """

    def __init__(
        self,
        *,
        phobert_detector: PhoBertDetector | None = None,
        rule_detector: KeywordDetector | None = None,
    ):
        self.phobert_detector = (
            phobert_detector if phobert_detector is not None else PhoBertDetector()
        )
        self.rule_detector = rule_detector if rule_detector is not None else KeywordDetector()

    def predict(self, text: str) -> HybridPrediction:
        """Giữ nguyên kết quả hai bộ phát hiện và giải thích quyết định cuối."""
        # PhoBERT kiểm tra kiểu dữ liệu và từ chối câu rỗng trước khi chạy Rules.
        phobert_result = self.phobert_detector.predict(text)
        rule_result = self.rule_detector.detect(text)
        label = phobert_result["label"]
        confidence = phobert_result["confidence"]
        has_rule_match = rule_result["needs_review"]

        keywords = list(dict.fromkeys(item["keyword"] for item in rule_result["findings"]))
        rule_summary = (
            "Rules khớp các cụm từ: " + ", ".join(f'“{keyword}”' for keyword in keywords) + "."
            if has_rule_match
            else "Rules không khớp cụm từ nào."
        )
        phobert_summary = f"PhoBERT dự đoán {label} với confidence {confidence:.2%}."

        if label == "CANH_BAO" and confidence >= WARNING_THRESHOLD:
            final_label = "CANH_BAO"
            reason = (
                f"Đạt ngưỡng cảnh báo {WARNING_THRESHOLD:.0%}; "
                "ưu tiên kết quả PhoBERT và chọn CANH_BAO."
            )
        elif label == "KHONG_CANH_BAO" and confidence >= SAFE_THRESHOLD:
            final_label = "KHONG_CANH_BAO"
            reason = (
                f"Đạt ngưỡng không cảnh báo {SAFE_THRESHOLD:.0%}; chọn KHONG_CANH_BAO "
                "theo chính sách ưu tiên PhoBERT. Câu có thể là câu phủ định hoặc "
                "thuộc ngữ cảnh an toàn."
            )
            if has_rule_match:
                reason += " Các cụm từ Rules khớp không làm thay đổi quyết định này."
        elif has_rule_match or label == "CAN_XAC_MINH":
            final_label = "CAN_XAC_MINH"
            reason = (
                "Chưa thỏa điều kiện chốt CANH_BAO hoặc KHONG_CANH_BAO. "
                "Có cụm từ Rules khớp hoặc PhoBERT chọn CAN_XAC_MINH; "
                "chuyển người kiểm duyệt xem lại."
            )
        else:
            final_label = "CAN_XAC_MINH"
            reason = (
                "PhoBERT chưa đạt ngưỡng tin cậy cho nhãn dự đoán. "
                "Dù không có Rules khớp, vẫn chọn CAN_XAC_MINH "
                "để người kiểm duyệt xem lại."
            )

        final_confidence = phobert_result["probabilities"][final_label]
        if final_label != label:
            reason += (
                f" Nhãn cuối được chuyển theo chính sách; final_confidence {final_confidence:.2%} "
                f"là xác suất PhoBERT dành cho {final_label}, không phải điểm tin cậy tổng hợp."
            )
        return {
            "final_label": final_label,
            "final_confidence": final_confidence,
            "phobert_result": phobert_result,
            "rule_result": rule_result,
            "explanation": f"{phobert_summary} {rule_summary} {reason}",
        }


@lru_cache(maxsize=1)
def _get_default_detector() -> HybridDetector:
    # Chỉ nạp checkpoint khi predict() được gọi lần đầu.
    return HybridDetector()


def predict(text: str) -> HybridPrediction:
    """API tiện dụng: from nlp.hybrid_detector import predict."""
    return _get_default_detector().predict(text)
