from typing import Literal
from pydantic import BaseModel, Field

Label = Literal['KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO']


class PhoBertResult(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[Label, float]


class ClassificationUnit(BaseModel):
    unit_id: str
    source_segment_id: int | str
    text: str
    start: float
    end: float
    timestamp_is_approximate: bool
    final_label: Label
    final_confidence: float = Field(ge=0, le=1)
    phobert_result: PhoBertResult
    rule_result: dict
    explanation: str


class Summary(BaseModel):
    whisper_segment_count: int
    classification_unit_count: int
    final_label_counts: dict[Label, int]


class AudioResponse(BaseModel):
    filename: str
    processing_time_seconds: float
    transcript: str
    summary: Summary
    classification_units: list[ClassificationUnit]
