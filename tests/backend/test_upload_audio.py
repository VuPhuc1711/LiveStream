import tempfile
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from backend.pipeline_service import PipelineService, InvalidAudioError
from backend.upload_audio import decoded_pcm, UploadSpeechWindows, AudioDecodeError
from tests.backend.test_live import fake_vad, packet, result
from backend.speech_buffer import RATE


@contextmanager
def pcm_source(pcm):
    yield (pcm[i:i+16000] for i in range(0, len(pcm), 16000))


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.service = PipelineService()
        self.service.transcriber = Mock()
        self.service.detector = Mock()
        self.service.detector.predict.side_effect = lambda _: {k: v for k, v in result()['units'][0].items()
            if k in ('final_label','final_confidence','phobert_result','explanation')}
        self.service.live_vad = fake_vad()
        self.service.labels = ('KHONG_CANH_BAO', 'CANH_BAO')
        self.service.classification_mode = 'binary'
        self.paths = []
        def recognize(path):
            self.paths.append(Path(path))
            with wave.open(str(path), 'rb') as audio:
                self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()), (16000, 1, 2))
                length = audio.getnframes()/RATE
            return {'text':'Một câu thật.', 'segments':[{'id':0,'start':0,'end':length,
                'text':'Một câu thật.', 'tokens':[123], 'no_speech_prob':.99}]}
        self.service.transcriber.transcribe.side_effect = recognize

    def analyze_pcm(self, pcm):
        with patch('backend.upload_audio.decoded_pcm', side_effect=lambda _: pcm_source(pcm)):
            output = self.service.analyze(Path('input.wav'))
        self.assertTrue(all(not path.exists() and not path.parent.exists() for path in self.paths))
        return output

    def test_whole_silence_has_no_asr_classifier_or_transcript(self):
        output = self.analyze_pcm(bytes(RATE*2*5))
        self.service.transcriber.transcribe.assert_not_called()
        self.service.detector.predict.assert_not_called()
        self.assertEqual(output['transcript'], '')
        self.assertEqual(output['classification_units'], [])
        self.assertEqual(output['audio_accounting']['silence_seconds'], 5)

    def test_separate_regions_keep_original_offsets_and_unique_units(self):
        pcm=bytes(RATE*2*2)+packet(0,1)[4:]+bytes(RATE*2*3)+packet(0,1)[4:]+bytes(RATE*2*2)
        output = self.analyze_pcm(pcm)
        self.assertEqual(len(output['classification_units']),2)
        first, second = output['classification_units']
        self.assertAlmostEqual(first['start'],1.7,places=2)
        self.assertAlmostEqual(second['start'],5.7,places=2)
        self.assertLessEqual(second['end'],7.31)
        self.assertNotEqual(first['unit_id'],second['unit_id'])
        self.assertTrue(first['timestamp_is_approximate'])
        self.assertNotIn('tokens', str(output))
        self.assertEqual(output['audio_accounting']['received_seconds'],9)
        self.assertAlmostEqual(sum(output['audio_accounting'][k] for k in ('processed_audio_seconds','silence_seconds')),9)
        # High no_speech_prob alone did not erase a confirmed voice region.
        self.assertEqual(self.service.detector.predict.call_count,2)

    def test_subsecond_forced_tail_joins_contiguous_context_without_losing_pcm(self):
        pcm = packet(0,12.2)[4:]+bytes(RATE*2*2)
        source = UploadSpeechWindows(fake_vad(), max_seconds=12)
        windows = list(source.windows([pcm]))
        self.assertEqual(len(windows),1)
        self.assertTrue(windows[0].boundary)
        self.assertGreater(len(windows[0].pcm)/(RATE*2),12.2)
        self.assertEqual(windows[0].pcm,pcm[:len(windows[0].pcm)])
        self.assertEqual(len(windows[0].pcm)//2+source.buffer.silence_samples,len(pcm)//2)
        with patch('backend.upload_audio.WINDOW_SECONDS',12):
            output = self.analyze_pcm(pcm)
        self.assertEqual(len(output['classification_units']),1)
        self.service.transcriber.transcribe.assert_called_once()

    def test_empty_tail_flushes_last_sentence_once_without_another_asr(self):
        with patch('backend.upload_audio.WINDOW_SECONDS',12):
            output = self.analyze_pcm(packet(0,12)[4:]+bytes(RATE*2*2))
        self.service.transcriber.transcribe.assert_called_once()
        self.assertEqual(len(output['classification_units']),1)
        self.assertEqual(output['transcript'],'Một câu thật.')

    def test_consecutive_uploads_reset_vad_and_carry_and_keep_true_repetitions(self):
        with patch('backend.upload_audio.WINDOW_SECONDS',12):
            self.analyze_pcm(packet(0,12.2)[4:])
        output = self.analyze_pcm(packet(0,1)[4:]+bytes(RATE*2*2)+packet(0,1)[4:])
        self.assertEqual(output['transcript'],'Một câu thật. Một câu thật.')
        self.assertEqual(output['classification_units'][0]['unit_id'],'U0001')
        self.assertEqual(output['classification_units'][0]['start'],0)
        self.assertEqual(self.service.live_vad.reset.call_count,4)

    def test_decoder_failure_restores_vad_and_releases_service_lock(self):
        with patch('backend.upload_audio.decoded_pcm', side_effect=AudioDecodeError('Invalid audio')):
            with self.assertRaises(InvalidAudioError):self.service.analyze('bad.wav')
        self.service.live_vad.reset.assert_called()
        self.assertEqual(self.analyze_pcm(packet(0,1)[4:])['transcript'],'Một câu thật.')

    def test_ffmpeg_real_stereo_48khz_decode_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stereo.wav'
            signal=(1000*np.sin(np.arange(48000)*2*np.pi*240/48000)).astype('<i2')
            with wave.open(str(path),'wb') as output:
                output.setnchannels(2); output.setsampwidth(2);output.setframerate(48000)
                output.writeframes(np.column_stack([signal,signal]).tobytes())
            with decoded_pcm(path) as frames:pcm=b''.join(frames)
            self.assertEqual(len(pcm),RATE*2)
            path.write_bytes(b'corrupt')
            with self.assertRaises(AudioDecodeError):
                with decoded_pcm(path) as frames:list(frames)


if __name__=='__main__': unittest.main()
