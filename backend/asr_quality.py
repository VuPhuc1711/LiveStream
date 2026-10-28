"""Audio-duration checks for ASR output, independent of words and class labels."""
import math
import re
import unicodedata

REVISION = 'audio_timestamp_no_prompt_recovery_v1'


def timestamp_issues(raw, duration):
    """Allow normal frame rounding, never turn a 30s decode into a valid 12s one."""
    if duration is None:
        return []
    tolerance = .5  # frame rounding, not a percentage of a long upload window
    issues = []
    for index, segment in enumerate(raw.get('segments', [])):
        reason = None
        try:
            start, end = float(segment['start']), float(segment['end'])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                reason = 'invalid_timestamp'
            elif start >= duration:
                reason = 'no_timestamp_overlap_with_received_audio'
            elif end > duration + tolerance:
                reason = 'timestamp_exceeds_audio_duration'
        except (TypeError, ValueError, KeyError):
            reason = 'invalid_timestamp'
        if reason:
            issues.append(dict(segment_id=segment.get('id', index), reason=reason))
    return issues


def raw_snapshot(raw):
    # Preserve evidence and optional word alignment, never giant token/logit arrays.
    keys = ('id', 'start', 'end', 'text', 'no_speech_prob', 'avg_logprob',
            'compression_ratio', 'temperature', 'words')
    return json_safe(dict(text=raw.get('text', ''), segments=[
        {key: segment[key] for key in keys if key in segment}
        for segment in raw.get('segments', [])]))


def json_safe(value):
    # PostgreSQL JSON rejects NaN/Infinity. Keep invalid evidence explicitly as
    # text rather than failing the whole session or inventing a valid timestamp.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def same_decoded_text(left, right):
    """A new timestamp alone cannot rehabilitate the same invalid decode."""
    def normalize(raw):
        text = ' '.join(str(s.get('text', '')) for s in raw.get('segments', []))
        return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))
    return normalize(left) == normalize(right)


def guard_configuration():
    return dict(revision=REVISION, max_retries=1,
                timestamp_tolerance_seconds=.5,
                unrecovered_action='explicit_unrecognized_window_no_classification',
                retry_policy='same_full_audio_without_vocabulary_prompt_native_timestamps',
                word_alignment_retry=False, partition_retry=False,
                invalid_short_tail_retry=False,
                text_blacklist=False)
