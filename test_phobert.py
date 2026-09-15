r"""Chạy thử PhoBERT trên câu mới, không huấn luyện hay đọc tập test CSV.

PowerShell, tại thư mục dự án:
    .\.venv-nlp\Scripts\python.exe -B -X utf8 test_phobert.py
Thêm --device cpu nếu muốn kiểm tra trên CPU.
"""

import argparse
import math
import unicodedata

from nlp.phobert_detector import LABELS, PhoBertDetector


EXAMPLES = (
    "Chiếc áo này làm từ cotton, có ba màu và giá 150 nghìn đồng.",
    "Sản phẩm này chữa khỏi hoàn toàn bệnh tiểu đường, không cần dùng thuốc bác sĩ kê.",
    "Đây là sản phẩm bán chạy nhất Việt Nam, giá rẻ nhất thị trường.",
    "Không nên tin lời quảng cáo rằng sản phẩm này chữa khỏi hoàn toàn mọi bệnh.",
    "Mã giảm giá 20% áp dụng cho đơn từ 300 nghìn đồng đến hết hôm nay.",
    "Kem này giúp trắng da sau một lần sử dụng và không có tác dụng phụ.",
)


def check_prediction(result):
    """Kiểm tra giao diện kết quả, không áp đặt nhãn đúng cho câu minh họa."""
    assert set(result) == {"label", "confidence", "probabilities"}, result
    probabilities = result["probabilities"]
    assert set(probabilities) == set(LABELS), probabilities
    assert all(
        isinstance(value, float) and math.isfinite(value) and 0 <= value <= 1
        for value in probabilities.values()
    ), probabilities
    assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6), probabilities
    assert result["label"] == max(probabilities, key=probabilities.get), result
    assert result["confidence"] == probabilities[result["label"]], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    print("Đang nạp checkpoint PhoBERT đã lưu...", flush=True)
    detector = PhoBertDetector(device=args.device)
    print(f"Checkpoint: {detector.checkpoint_path}")
    print(f"Thiết bị: {detector.device} | Giới hạn: {detector.max_length} token")
    print("Kết quả minh họa của mô hình, không phải đánh giá độ chính xác.\n", flush=True)

    results = []
    for index, sentence in enumerate(EXAMPLES, start=1):
        result = detector.predict(sentence)
        check_prediction(result)
        results.append(result)
        print(f"[{index}] {sentence}")
        print(f"Nhãn: {result['label']} | Confidence: {result['confidence']:.2%}")
        for label, probability in result["probabilities"].items():
            print(f"  {label}: {probability:.2%}")
        print(flush=True)

    # Cùng câu với Unicode tổ hợp và khoảng trắng khác vẫn phải có cùng kết quả.
    variant = " \n" + unicodedata.normalize("NFD", EXAMPLES[0]).replace(" ", "  \t") + " "
    equivalent = detector.predict(variant)
    check_prediction(equivalent)
    assert equivalent["label"] == results[0]["label"]
    assert all(
        math.isclose(equivalent["probabilities"][label], results[0]["probabilities"][label],
                     rel_tol=1e-5, abs_tol=1e-6)
        for label in LABELS
    ), "Chuẩn hóa Unicode/khoảng trắng cho kết quả không nhất quán."

    for invalid, expected_error in (("", ValueError), (" \n\t ", ValueError), (None, TypeError)):
        try:
            detector.predict(invalid)
        except expected_error:
            pass
        else:
            raise AssertionError(f"Phải từ chối đầu vào không hợp lệ: {invalid!r}")

    print(f"PASS: {len(EXAMPLES)} câu ví dụ, xác suất hợp lệ, chuẩn hóa và kiểm tra đầu vào.")


if __name__ == "__main__":
    main()
