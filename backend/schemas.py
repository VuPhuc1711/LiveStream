from typing import Literal
from pydantic import BaseModel, Field, model_validator
from nlp.label_mapping import labels_for_mode

Label = Literal['KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO']


class PhoBertResult(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[Label, float]


class ClassificationUnit(BaseModel):
    segment_id: str | None = None
    unit_id: str
    source_segment_id: int | str
    text: str
    start: float
    end: float
    timestamp_is_approximate: bool
    final_label: Label
    final_confidence: float = Field(ge=0, le=1)
    phobert_result: PhoBertResult
    explanation: str


class Summary(BaseModel):
    whisper_segment_count: int
    classification_unit_count: int
    final_label_counts: dict[Label, int]


class AudioResponse(BaseModel):
    session_id: str | None = None
    classification_mode: Literal['three_class', 'binary'] = 'three_class'
    filename: str
    processing_time_seconds: float
    transcript: str
    summary: Summary
    classification_units: list[ClassificationUnit]
    asr_windows: list[dict] = Field(default_factory=list)
    audio_accounting: dict[str, float] = Field(default_factory=dict)
    asr_quality_warnings: list[dict] = Field(default_factory=list)
    asr_unresolved_windows: int = 0
    asr_recovered_windows: int = 0

    @model_validator(mode='after')
    def check_task_labels(self):
        expected = set(labels_for_mode(self.classification_mode))
        if set(self.summary.final_label_counts) != expected:
            raise ValueError('Summary labels do not match classification_mode')
        for unit in self.classification_units:
            if (unit.final_label not in expected or unit.phobert_result.label not in expected or
                    set(unit.phobert_result.probabilities) != expected):
                raise ValueError('Prediction labels do not match classification_mode')
        return self


class FeedbackInput(BaseModel):
    label: Literal['KHONG_CANH_BAO', 'CANH_BAO']
    note: str = Field(default='', max_length=2000)
    corrected_text: str | None = Field(default=None, min_length=1, max_length=4000)
    situation: str = Field(default='', max_length=200)


class ReviewContent(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    label: Literal['KHONG_CANH_BAO', 'CANH_BAO']
    family_id: str = Field(min_length=1, max_length=120)
    situation: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def reject_blank(self):
        for name in ('text', 'family_id', 'situation', 'reason'):
            value = getattr(self, name).strip()
            if not value:
                raise ValueError('Trường bắt buộc không được chỉ chứa khoảng trắng.')
            setattr(self, name, value)
        return self


class ReviewCreate(ReviewContent):
    source_type: Literal['AI_GENERATED', 'USER_ENTERED'] = 'USER_ENTERED'
    source_sample_id: str | None = None


class ReviewUpdate(ReviewContent):
    expected_revision: int = Field(ge=1)
    editor: str = Field(default='', max_length=120)


class ReviewApproval(BaseModel):
    expected_revision: int = Field(ge=1)
    reviewer: str = Field(min_length=1, max_length=120)
    confirm_text: Literal[True]

    @model_validator(mode='after')
    def require_reviewer(self):
        self.reviewer = self.reviewer.strip()
        if not self.reviewer:
            raise ValueError('Cần tên người thực sự duyệt mẫu.')
        return self
