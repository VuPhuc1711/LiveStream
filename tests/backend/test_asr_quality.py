"""Failure geometry/carry isolation tests; model mocks are explicit here."""
import unittest
from unittest.mock import Mock
from backend.pipeline_service import PipelineService
from backend.asr_quality import timestamp_issues


def raw(text, end, start=0):
    return dict(text=text, segments=[dict(id=0,start=start,end=end,text=text,no_speech_prob=1e-10)])


class QualityTests(unittest.TestCase):
    def service(self, first, retry=None):
        service=PipelineService();service.transcriber=Mock();service.detector=Mock()
        service.transcriber.transcribe.return_value=first
        service.transcriber.transcribe_without_prompt.return_value=retry or {'text':'','segments':[]}
        service.detector.predict.return_value={'final_label':'KHONG_CANH_BAO'}
        return service

    def test_severe_overrun_is_unresolved_even_if_aligned_retry_would_fit(self):
        first=raw('Lời giả với timestamp sai.',29.98)
        aligned=raw(first['text'],11.5)
        service=self.service(first,aligned)
        result=service.analyze_live('same.wav',.148,False,duration=12.012,voiced_seconds=10.72)
        service.transcriber.transcribe_retry.assert_not_called()
        service.transcriber.transcribe_without_prompt.assert_called_once_with('same.wav')
        self.assertIsNone(result['pending'])
        self.assertEqual(result['whisper_raw']['text'],first['text'])
        self.assertEqual(result['asr_recovery']['state'],'unresolved')
        self.assertTrue(result['asr_quality_warning']['retried'])
        service.detector.predict.assert_not_called()
        service.transcriber.transcribe.return_value=raw('Anh chị đặt trong giỏ nhé.',1)
        final=service.analyze_live('next.wav',12.16,True,result['pending'],duration=1.2)
        text=final['units'][0]['text']
        self.assertNotIn(first['text'],text)
        self.assertEqual(text,'Anh chị đặt trong giỏ nhé.')

    def test_single_no_prompt_native_retry_keeps_original_and_recovers_distinct_text(self):
        first=raw('Lời giải mã có thời gian bất thường.',29.98)
        recovered=raw('Shop không nhận đơn qua Zalo.',11.5)
        service=self.service(first,recovered)
        output=service.analyze_live('same.wav',.148,True,duration=12.012)
        self.assertEqual(output['whisper_raw']['text'],first['text'])
        self.assertEqual(output['asr_recovery']['whisper_raw']['text'],recovered['text'])
        self.assertEqual(output['asr_recovery']['state'],'recovered')
        self.assertEqual(output['units'][0]['text'],recovered['text'])
        self.assertEqual(output['asr_used_attempt'],'without_prompt')
        service.transcriber.transcribe_without_prompt.assert_called_once_with('same.wav')

    def test_invalid_window_never_turns_into_carry_or_a_safe_label(self):
        first=raw('Một đoạn bất kỳ.',29.98)
        service=self.service(first,raw('Đầu ra khác vẫn sai thời gian.',30))
        pending=dict(text='Mã này là số một',start=1,end=12)
        result=service.analyze_live('x.wav',12,True,pending,duration=12,voiced_seconds=10)
        self.assertEqual(result['units'],[]);self.assertIsNone(result['pending'])
        self.assertEqual(result['asr_quality_warning']['start'],1)
        self.assertEqual(result['asr_quality_warning']['end'],24)
        self.assertEqual(result['asr_recovery']['state'],'unresolved')
        self.assertEqual(result['asr_recovery']['uncommitted_context'],pending)
        service.detector.predict.assert_not_called()

    def test_invalid_timestamps_preserve_json_safe_evidence(self):
        import json
        service=self.service(raw('X',float('nan')))
        result=service.analyze_live('x.wav',0,True,duration=12)
        self.assertEqual(result['asr_quality_warning']['reason'],'retry_no_usable_text')
        self.assertEqual(result['whisper_raw']['segments'][0]['end'],'nan')
        json.dumps(result,allow_nan=False)
        service.detector.predict.assert_not_called()

    def test_zero_length_caption_is_not_rendered_at_00_00(self):
        service=self.service(raw('Một câu không có thời gian hợp lệ.',0))
        result=service.analyze_live('x.wav',0,True,duration=12)
        self.assertEqual(result['units'],[])
        self.assertEqual(result['asr_quality_warning']['end'],12)
        service.detector.predict.assert_not_called()

    def test_real_subscribe_words_are_not_blacklisted_and_high_no_speech_does_not_delete_negation(self):
        for text,end,duration in [('Hãy subscribe cho kênh của shop nhé.',3,3.2),('Không.',.2,.3)]:
            first=raw(text,end);first['segments'][0]['no_speech_prob']=.99
            service=self.service(first)
            result=service.analyze_live('x.wav',0,True,duration=duration,voiced_seconds=.08)
            self.assertEqual(result['units'][0]['text'],text)
            service.transcriber.transcribe_retry.assert_not_called()
            service.transcriber.transcribe_without_prompt.assert_not_called()

    def test_frame_rounding_preserves_final_words_and_shop_scope(self):
        text='Không phải rẻ nhất thị trường, chỉ rẻ nhất trong gian hàng nhà em.'
        service=self.service(raw(text,12.2))
        result=service.analyze_live('x.wav',20,True,duration=12.012,voiced_seconds=9)
        self.assertEqual(result['units'][0]['text'],text)
        self.assertAlmostEqual(result['units'][0]['end'],32.012)
        service.transcriber.transcribe_retry.assert_not_called()
        service.transcriber.transcribe_without_prompt.assert_not_called()

    def test_short_tail_with_false_confident_decode_is_not_rescued_by_no_speech_prob(self):
        service=self.service(raw('Một từ.',10.64))
        result=service.analyze_live('x.wav',88.608,True,duration=.492,voiced_seconds=.032)
        self.assertEqual(result['units'],[])
        self.assertEqual(result['asr_recovery']['state'],'unresolved')
        service.transcriber.transcribe_retry.assert_not_called()
        service.transcriber.transcribe_without_prompt.assert_not_called()

    def test_timestamp_guard_uses_duration_not_words(self):
        self.assertTrue(timestamp_issues(raw('Bất kỳ nội dung nào.',29.98),12.012))
        self.assertFalse(timestamp_issues(raw('Cùng một nội dung.',12.2),12.012))


if __name__=='__main__':unittest.main()
