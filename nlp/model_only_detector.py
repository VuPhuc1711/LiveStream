"""Adapter for serving PhoBERT predictions without Rules or Hybrid thresholds."""

from nlp.phobert_detector import PhoBertDetector


class ModelOnlyDetector:
    def __init__(self, phobert_detector: PhoBertDetector | None = None):
        self.phobert_detector = phobert_detector or PhoBertDetector()

    def predict(self, text: str) -> dict:
        result = self.phobert_detector.predict(text)
        return {
            "final_label": result["label"],
            "final_confidence": result["confidence"],
            "phobert_result": result,
            "explanation": "PhoBERT phân loại câu này dựa trên nội dung đã nhận dạng; người kiểm duyệt cần xem lại khi cần.",
        }
