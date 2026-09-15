"""Dự đoán một câu bằng checkpoint PhoBERT đã huấn luyện của dự án.

Dùng môi trường .venv-nlp. Mô hình và tokenizer chỉ được nạp từ ổ đĩa.
"""

import json
import unicodedata
from pathlib import Path
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize
from model_config import get_default_checkpoint


LABELS = ("KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO")
DEFAULT_CHECKPOINT = get_default_checkpoint()


class Prediction(TypedDict):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PhoBertDetector:
    """Nạp mô hình một lần; gọi predict(text) cho từng câu cần phân loại."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.checkpoint_path = (
            get_default_checkpoint() if checkpoint_path is None else Path(checkpoint_path).resolve()
        )
        if not self.checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục checkpoint: {self.checkpoint_path}"
            )

        with (self.checkpoint_path / "preprocessing.json").open(
            encoding="utf-8"
        ) as stream:
            preprocessing = json.load(stream)
        if (
            preprocessing.get("normalization")
            != "NFC and collapse whitespace; preserve case and punctuation"
            or preprocessing.get("segmenter")
            != "underthesea.word_tokenize(format='text')"
        ):
            raise ValueError("Checkpoint yêu cầu cách tiền xử lý chưa được hỗ trợ.")
        self.max_length = preprocessing.get("max_length")
        if type(self.max_length) is not int or not 8 <= self.max_length <= 256:
            raise ValueError("max_length của checkpoint phải là số nguyên từ 8 đến 256.")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, use_fast=False, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            attn_implementation="eager",
        )
        expected_id2label = dict(enumerate(LABELS))
        expected_label2id = {label: index for index, label in enumerate(LABELS)}
        if (
            self.model.config.id2label != expected_id2label
            or self.model.config.label2id != expected_label2id
        ):
            raise ValueError("Label mapping của checkpoint không khớp train_phobert.py.")
        self.labels = tuple(self.model.config.id2label[i] for i in range(len(LABELS)))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Prediction:
        """Trả về nhãn, confidence và ba xác suất softmax trong khoảng [0, 1].

        Giới hạn token và cách cắt câu dài giống lúc huấn luyện.
        Confidence là xác suất của nhãn được chọn, chưa được hiệu chỉnh.
        """
        if not isinstance(text, str):
            raise TypeError("Nội dung dự đoán phải là chuỗi tiếng Việt.")

        # Giống normalize_text() trong train_phobert.py; không casefold/lower.
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        if not normalized:
            raise ValueError("Nội dung dự đoán không được để trống.")
        segmented = word_tokenize(normalized, format="text")
        inputs = self.tokenizer(
            segmented,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Mô hình trả về NaN/Inf; không thể tính xác suất.")
            scores = torch.softmax(logits, dim=-1).cpu().tolist()

        probabilities = dict(zip(self.labels, scores))
        label = max(probabilities, key=probabilities.get)
        return {
            "label": label,
            "confidence": probabilities[label],
            "probabilities": probabilities,
        }
