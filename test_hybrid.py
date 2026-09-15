r"""Kiểm tra chính sách hybrid và chạy 8 câu với checkpoint PhoBERT thật.

PowerShell, tại thư mục dự án:
    .\.venv-nlp\Scripts\python.exe -B -X utf8 test_hybrid.py
Thêm --policy-only để chỉ kiểm tra chính sách bằng kết quả PhoBERT giả lập.
Thêm --device cpu để chạy các câu ví dụ trên CPU.
"""

import argparse
import math
from unittest.mock import Mock, patch

from nlp.hybrid_detector import HybridDetector, predict
from nlp.keyword_detector import KeywordDetector
from nlp.phobert_detector import LABELS, PhoBertDetector


EXAMPLES = (
    "Chiếc áo này làm từ cotton, có ba màu và giá 150 nghìn đồng.",
    "Sản phẩm này chữa khỏi hoàn toàn bệnh tiểu đường, không cần dùng thuốc bác sĩ kê.",
    "Đây là sản phẩm bán chạy nhất Việt Nam, giá rẻ nhất thị trường.",
    "Không nên tin lời quảng cáo rằng sản phẩm này chữa khỏi hoàn toàn mọi bệnh.",
    "Mã giảm giá 20% áp dụng cho đơn từ 300 nghìn đồng đến hết hôm nay.",
    "Kem này giúp trắng da sau một lần sử dụng và không có tác dụng phụ.",
    "Shop có hai kích cỡ M và L, khách có thể đổi trong bảy ngày nếu chưa sử dụng.",
    "Sản phẩm này không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh.",
)


def check_result(result):
    assert set(result) == {
        "final_label", "final_confidence", "phobert_result", "rule_result", "explanation"
    }, result
    assert result["final_label"] in LABELS, result
    assert isinstance(result["explanation"], str) and result["explanation"].strip(), result
    phobert = result["phobert_result"]
    probabilities = phobert["probabilities"]
    assert set(probabilities) == set(LABELS), probabilities
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6), probabilities
    assert phobert["label"] == max(probabilities, key=probabilities.get), phobert
    assert phobert["confidence"] == probabilities[phobert["label"]], phobert
    assert result["final_confidence"] == probabilities[result["final_label"]], result
    assert result["rule_result"]["needs_review"] == bool(result["rule_result"]["findings"])


def test_policy():
    """Kiểm tra chính xác ranh giới >= và quyền ưu tiên, độc lập chất lượng model."""
    below_warning = math.nextafter(0.55, 0.0)
    below_safe = math.nextafter(0.65, 0.0)
    # Thứ tự xác suất: KHONG_CANH_BAO, CAN_XAC_MINH, CANH_BAO.
    cases = (
        ("Cảnh báo đúng ngưỡng", (0.20, 0.25, 0.55), "CANH_BAO"),
        ("Cảnh báo ngay dưới ngưỡng", (0.20, 0.80 - below_warning, below_warning), "CAN_XAC_MINH"),
        ("Không cảnh báo đúng ngưỡng", (0.65, 0.20, 0.15), "KHONG_CANH_BAO"),
        ("Không cảnh báo ngay dưới ngưỡng", (below_safe, 0.20, 0.80 - below_safe), "CAN_XAC_MINH"),
        ("PhoBERT cần xác minh", (0.20, 0.60, 0.20), "CAN_XAC_MINH"),
        ("Cảnh báo thiếu tin cậy", (0.30, 0.20, 0.50), "CAN_XAC_MINH"),
        ("Không cảnh báo thiếu tin cậy", (0.60, 0.20, 0.20), "CAN_XAC_MINH"),
    )
    rules = KeywordDetector()
    count = 0
    for description, scores, expected_label in cases:
        for should_match, text in (
            (False, "Áo cotton màu xanh."),
            (True, "Không nên tin quảng cáo chữa khỏi hoàn toàn."),
        ):
            probabilities = dict(zip(LABELS, scores))
            label = max(probabilities, key=probabilities.get)
            phobert_result = {
                "label": label,
                "confidence": probabilities[label],
                "probabilities": probabilities,
            }
            phobert = Mock(spec=PhoBertDetector)
            phobert.predict.return_value = phobert_result
            detector = HybridDetector(phobert_detector=phobert, rule_detector=rules)
            result = detector.predict(text)
            check_result(result)
            assert result["final_label"] == expected_label, (description, should_match, result)
            assert result["phobert_result"] is phobert_result
            assert result["rule_result"] == rules.detect(text)
            assert result["rule_result"]["needs_review"] is should_match
            phobert.predict.assert_called_once_with(text)
            if expected_label == "KHONG_CANH_BAO":
                assert "phủ định" in result["explanation"]
                assert "ngữ cảnh an toàn" in result["explanation"]
            count += 1

    # API hàm cấp module chuyển nguyên câu và trả nguyên kết quả của detector.
    default_detector = Mock(spec=HybridDetector)
    default_detector.predict.return_value = result
    with patch("nlp.hybrid_detector._get_default_detector", return_value=default_detector):
        assert predict("Câu mới.") is result
    default_detector.predict.assert_called_once_with("Câu mới.")
    print(f"PASS: {count} trường hợp chính sách/ngưỡng và API predict(text).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--policy-only", action="store_true")
    args = parser.parse_args()
    test_policy()
    if args.policy_only:
        return

    print("Đang nạp checkpoint PhoBERT để kiểm tra tích hợp...", flush=True)
    phobert = PhoBertDetector(device=args.device)
    detector = HybridDetector(phobert_detector=phobert)
    print(f"Thiết bị: {phobert.device} | Checkpoint: {phobert.checkpoint_path}")
    print("final_confidence = xác suất PhoBERT của nhãn cuối; chưa hiệu chỉnh cho hybrid.")
    print("\n| STT | Câu | Rules khớp | PhoBERT | Confidence PhoBERT | Nhãn cuối | final_confidence |")
    print("|---|---|---|---|---:|---|---:|", flush=True)
    results = []
    for index, text in enumerate(EXAMPLES, start=1):
        result = detector.predict(text)
        check_result(result)
        results.append(result)
        keywords = list(dict.fromkeys(item["keyword"] for item in result["rule_result"]["findings"]))
        matches = ", ".join(keywords) if keywords else "Không"
        prediction = result["phobert_result"]
        print(
            f"| {index} | {text} | {matches} | {prediction['label']} | "
            f"{prediction['confidence']:.2%} | {result['final_label']} | "
            f"{result['final_confidence']:.2%} |",
            flush=True,
        )

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['explanation']}", flush=True)

    for invalid, expected_error in (("", ValueError), (" \n\t ", ValueError), (None, TypeError)):
        try:
            detector.predict(invalid)
        except expected_error:
            pass
        else:
            raise AssertionError(f"Phải từ chối đầu vào không hợp lệ: {invalid!r}")
    print(f"\nPASS: {len(EXAMPLES)} câu với model thật và kiểm tra đầu vào; không đánh giá độ chính xác.")


if __name__ == "__main__":
    main()
