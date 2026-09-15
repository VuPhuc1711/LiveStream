"""Tách câu theo dấu câu mà không dùng ground truth, nhãn hoặc model."""

import math
import re


SENTENCE_END = re.compile(r'[.!?。！？]+[\"\u201d\u2019\u00bb)\]]*')
SEGMENTATION_VERSION = "punctuation_v1"


def sentence_spans(text):
    """Giữ dấu kết thúc và phần cuối chưa có dấu; không tách dấu thập phân 3.5."""
    if not isinstance(text, str):
        raise TypeError("Text của Whisper phải là chuỗi.")
    spans = []
    begin = 0
    boundaries = []
    for match in SENTENCE_END.finditer(text):
        if (match.group() == "." and match.start() > 0 and match.end() < len(text)
                and text[match.start() - 1].isdigit() and text[match.end()].isdigit()):
            continue
        boundaries.append(match.end())
    for end in boundaries + [len(text)]:
        left, right = begin, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(char.isalnum() for char in text[left:right]):
            spans.append((left, right))
        begin = end
    return spans


def build_classification_units(whisper_segments):
    """Mỗi câu là một unit; chỉ phân chia thời lượng bằng tỷ lệ số ký tự.

    timestamp_is_approximate=False nghĩa là giữ nguyên khoảng của một segment;
    không có nghĩa timestamp Whisper đã được kiểm chứng chính xác bằng nghe tay.
    """
    units = []
    for index, segment in enumerate(whisper_segments):
        start, end = float(segment["start"]), float(segment["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f"Segment Whisper {index} có thời gian không hợp lệ.")
        text = segment["text"]
        spans = sentence_spans(text)
        total_length = sum(right - left for left, right in spans)
        consumed = 0
        for part, (left, right) in enumerate(spans):
            unit_start = start + (end - start) * consumed / total_length
            consumed += right - left
            unit_end = end if part == len(spans) - 1 else start + (end - start) * consumed / total_length
            unit_id = f"U{len(units) + 1:04d}"
            units.append({"unit_id": unit_id, "id": unit_id,
                          "source_segment_id": segment.get("id", index), "source_segment_index": index,
                          "text": text[left:right], "source_text_start": left, "source_text_end": right,
                          "start": unit_start, "end": unit_end,
                          "timestamp_is_approximate": len(spans) > 1,
                          "timestamp_method": "character_length_proportion" if len(spans) > 1 else "whisper_segment_bounds"})
    return units
