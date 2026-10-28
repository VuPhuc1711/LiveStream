import asyncio
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from backend.app import create_app
from backend.database import AnalysisSegment, Database
from backend.live_audio import ChunkOrder, ProtocolError, UtteranceBuffer, RATE
from backend.pipeline_service import PipelineService
from backend import history


def packet(index, seconds=.5, value=1000):
    samples=round(RATE*seconds)
    signal=(value*np.sin(np.arange(samples)*2*np.pi*240/RATE)).astype('<i2')
    return struct.pack('<I', index) + signal.tobytes()


def fake_vad():
    """Only deterministic buffer/state tests use this mock, never ASR smoke proof."""
    vad=Mock(frame_samples=320, onset_threshold=.5, offset_threshold=.35, min_speech_ms=60)
    vad.probability.side_effect=lambda frame: float(bool(np.any(np.frombuffer(frame,dtype='<i2'))))
    return vad


def result(text='Xin chào.'):
    unit = dict(unit_id='local', source_segment_id=0, text=text, start=0, end=.5,
        timestamp_is_approximate=True, final_label='KHONG_CANH_BAO', final_confidence=.9,
        phobert_result=dict(label='KHONG_CANH_BAO', confidence=.9,
            probabilities={'KHONG_CANH_BAO':.9,'CANH_BAO':.1}), explanation='Mô tả.')
    return dict(units=[unit], pending=None, temporary_transcript='', whisper_count=1)


class BufferTests(unittest.TestCase):
    def test_order_duplicate_gap_and_size(self):
        order=ChunkOrder()
        order.receive(packet(0))
        self.assertIsNone(order.receive(packet(0))[1])
        self.assertEqual(order.samples,8000)
        for bad in (packet(0,value=2000),packet(2),b'',packet(1,1.1)):
            with self.assertRaises(ProtocolError): order.receive(bad)

    def test_silence_boundary_and_stop_remainder(self):
        buffer=UtteranceBuffer(vad=fake_vad())
        self.assertEqual(buffer.feed(packet(0,value=0)[4:]),[])
        buffer.feed(packet(1)[4:])
        self.assertEqual(buffer.feed(packet(2,1,value=0)[4:]),[])
        windows=buffer.feed(packet(3,value=0)[4:])
        self.assertEqual(len(windows),1)
        self.assertTrue(windows[0].boundary)
        self.assertEqual(windows[0].speech_end,1)
        buffer.feed(packet(4)[4:])
        self.assertEqual(len(buffer.finish()),1)

    def test_forced_window_exact_stop_and_true_repetitions(self):
        buffer=UtteranceBuffer(vad=fake_vad())
        windows=[]
        for _ in range(24): windows.extend(buffer.feed(packet(0)[4:]))
        self.assertEqual(len(windows),1)
        self.assertFalse(windows[0].boundary)
        self.assertTrue(buffer.finish()[0].boundary)

    def test_real_vad_silence_and_dc_never_reach_asr(self):
        for pcm in (bytes(RATE*2*3), np.full(RATE*3,1000,dtype='<i2').tobytes()):
            buffer=UtteranceBuffer()
            self.assertEqual(buffer.feed(pcm),[])
            self.assertEqual(buffer.finish(),[])
            self.assertEqual(buffer.silence_samples,RATE*3)

    def test_forced_cut_silent_tail_flushes_without_asr_and_conserves_samples(self):
        buffer=UtteranceBuffer(max_seconds=1,vad=fake_vad())
        emitted=buffer.feed(packet(0,1)[4:])
        emitted+=buffer.feed(packet(1,2,value=0)[4:])
        emitted+=buffer.finish()
        self.assertEqual(len(emitted),2)
        self.assertFalse(emitted[1].pcm)
        self.assertTrue(emitted[1].boundary)
        self.assertEqual(sum(len(w.pcm)//2 for w in emitted)+buffer.silence_samples,3*RATE)

    def test_partial_frames_and_preroll_postroll_keep_short_voice(self):
        buffer=UtteranceBuffer(vad=fake_vad())
        pcm=packet(0,.4,0)[4:]+packet(1,.12)[4:]+packet(2,2,0)[4:]
        windows=[]
        for index in range(0,len(pcm),222):windows+=buffer.feed(pcm[index:index+222])
        windows+=buffer.finish()
        self.assertEqual(len(windows),1)
        self.assertLessEqual(windows[0].start,.4)
        self.assertGreaterEqual(windows[0].start+len(windows[0].pcm)/32000,.52)
        self.assertEqual(sum(len(w.pcm)//2 for w in windows)+buffer.silence_samples,len(pcm)//2)

    def test_context_tail_not_classified_until_next_boundary(self):
        service=PipelineService(); service.transcriber=Mock();service.detector=Mock()
        service.detector.predict.return_value = result()['units'][0]
        service.transcriber.transcribe.side_effect=[
            {'segments':[{'id':0,'text':'Mã này là top một','start':0,'end':12}]},
            {'segments':[{'id':0,'text':'trong gian hàng nhà em.','start':0,'end':3}]}]
        first=service.analyze_live('one.wav',0,False)
        self.assertEqual(first['units'],[])
        service.detector.predict.assert_not_called()
        service.analyze_live('two.wav',12,True,first['pending'])
        service.detector.predict.assert_called_once_with('Mã này là top một trong gian hàng nhà em.')

    def test_unconfirmed_impulse_never_sent_to_asr_including_finish(self):
        vad=fake_vad();vad.min_speech_ms=250
        buffer=UtteranceBuffer(vad=vad)
        self.assertEqual(buffer.feed(packet(0,.16)[4:]),[])
        self.assertEqual(buffer.finish(),[])
        self.assertEqual(buffer.silence_samples,round(.16*RATE))
        buffer=UtteranceBuffer(vad=vad)
        self.assertEqual(buffer.feed(packet(0,.4,0)[4:]+packet(1,.16)[4:]+packet(2,2,0)[4:]),[])
        self.assertEqual(buffer.finish(),[])
        self.assertEqual(buffer.silence_samples,round(2.56*RATE))

    def test_confirmed_short_voice_has_original_attack_and_end_with_no_energy_floor(self):
        vad=fake_vad();vad.min_speech_ms=250
        buffer=UtteranceBuffer(vad=vad)
        voice=packet(0,.32,10)[4:]
        original=bytes(6400)+voice+bytes(64000)
        windows=buffer.feed(original)+buffer.finish()
        self.assertEqual(len(windows),1)
        start=round(windows[0].start*RATE)
        self.assertEqual(windows[0].pcm,original[start*2:start*2+len(windows[0].pcm)])
        self.assertIn(voice,windows[0].pcm)
        self.assertEqual(sum(len(w.pcm)//2 for w in windows)+buffer.silence_samples,len(original)//2)

    def test_weak_attack_before_confident_voice_counts_without_lowering_onset(self):
        vad=fake_vad();vad.frame_samples=512;vad.min_speech_ms=250
        vad.probability.side_effect=iter([0]*16+[.38,.49,.3]+[.8]*7+[0]*64)
        buffer=UtteranceBuffer(vad=vad)
        pcm=bytes(512*2*90)
        windows=buffer.feed(pcm)+buffer.finish()
        self.assertEqual(len(windows),1)
        self.assertEqual(windows[0].voiced_samples,9*512)
        self.assertLess(windows[0].start,16*512/RATE)

    @unittest.skipUnless((Path(__file__).resolve().parents[2]/'runtime/live-debug/b892bf34-35af-4d2b-89f0-846f6e571ec3/received.wav').is_file(), 'Opt-in real microphone fixture not on this machine')
    def test_confirmed_real_silent_microphone_has_no_asr_windows(self):
        import wave
        path=Path(__file__).resolve().parents[2]/'runtime/live-debug/b892bf34-35af-4d2b-89f0-846f6e571ec3/received.wav'
        with wave.open(str(path)) as audio:pcm=audio.readframes(audio.getnframes())
        buffer=UtteranceBuffer()
        windows=[]
        for i in range(0,len(pcm),16000):windows+=buffer.feed(pcm[i:i+16000])
        windows+=buffer.finish()
        self.assertEqual(windows,[])
        self.assertEqual(buffer.silence_samples,len(pcm)//2)

    def test_asr_segment_outside_audio_does_not_create_zero_length_unit(self):
        service=PipelineService();service.transcriber=Mock();service.detector=Mock()
        service.transcriber.transcribe.return_value={'segments':[{'id':0,'text':'Chữ ngoài audio.','start':4,'end':5}]}
        service.transcriber.transcribe_without_prompt.return_value={'segments':[]}
        output=service.analyze_live('x.wav',10,True,duration=1)
        self.assertEqual(output['units'],[])
        self.assertEqual(output['asr_rejected_segments'][0]['reason'],'no_timestamp_overlap_with_received_audio')
        service.detector.predict.assert_not_called()

    def test_timestamps_cannot_exceed_received_audio(self):
        service=PipelineService();service.transcriber=Mock();service.detector=Mock()
        service.transcriber.transcribe.return_value={'segments':[{'id':0,'text':'Xin chào.','start':0,'end':1.2}]}
        service.detector.predict.return_value={'final_label':'KHONG_CANH_BAO'}
        output=service.analyze_live('window.wav',60,True,duration=1)
        self.assertEqual(output['units'][0]['end'],61)
        self.assertTrue(output['units'][0]['timestamp_clamped'])

    def test_unresolved_tail_preserves_raw_without_classifying_unfinished_context(self):
        service=PipelineService();service.transcriber=Mock();service.detector=Mock()
        service.transcriber.transcribe.return_value={'text':'Một câu bất kỳ do ASR sinh.',
            'segments':[{'id':0,'text':'Một câu bất kỳ do ASR sinh.','start':0,'end':6,'no_speech_prob':.79}]}
        service.detector.predict.return_value={'final_label':'KHONG_CANH_BAO'}
        service.transcriber.transcribe_without_prompt.return_value={'text':'','segments':[]}
        carry={'text':'Câu thật cuối cùng.','start':1,'end':3,'timestamp_is_approximate':True}
        output=service.analyze_live('tail.wav',3,True,carry,duration=.299,voiced_seconds=.059)
        self.assertEqual(output['units'],[])
        self.assertEqual(len(output['asr_rejected_segments']),1)
        self.assertEqual(output['whisper_raw']['text'],'Một câu bất kỳ do ASR sinh.')
        self.assertEqual(output['asr_recovery']['uncommitted_context']['text'],'Câu thật cuối cùng.')
        self.assertFalse(output['asr_quality_warning']['retried'])
        service.detector.predict.assert_not_called()

    def test_short_negation_with_valid_timestamp_is_not_filtered(self):
        service=PipelineService();service.transcriber=Mock();service.detector=Mock()
        service.transcriber.transcribe.return_value={'segments':[{'id':0,'text':'Không.','start':0,'end':.2,'no_speech_prob':.8}]}
        service.detector.predict.return_value={}
        output=service.analyze_live('short.wav',0,True,duration=.3,voiced_seconds=.08)
        self.assertEqual(output['units'][0]['text'],'Không.')
        self.assertEqual(output['asr_rejected_segments'],[])


class LiveTests(unittest.TestCase):
    def setUp(self):
        def fake_buffer(**kwargs):
            kwargs['vad']=fake_vad()
            return UtteranceBuffer(**kwargs)
        self.vad_patch=patch('backend.live_audio.UtteranceBuffer',side_effect=fake_buffer)
        self.vad_patch.start();self.addCleanup(self.vad_patch.stop)
        self.directory=tempfile.TemporaryDirectory()
        self.url=self.database_url()
        self.service=Mock()
        self.service.health.return_value=dict(ready=True,model_version='binary_v04',checkpoint='model/best',classification_mode='binary')
        self.service.analyze_live.return_value=result()
        self.app=create_app(lambda:self.service,self.url)
        self.client=TestClient(self.app);self.client.__enter__()

    def database_url(self):
        return 'sqlite:///'+(Path(self.directory.name)/'live.sqlite3').as_posix()

    def tearDown(self):
        self.client.__exit__(None,None,None);self.directory.cleanup()

    def connect(self):
        return self.client.websocket_connect('/ws/monitor',headers={'origin':'http://127.0.0.1:5173'})

    def start(self,ws,source=None):
        config=dict(type='start',format='pcm_s16le',sample_rate=16000,channels=1)
        if source is not None:config['source']=source
        ws.send_json(config)
        m=ws.receive_json();self.assertEqual(m['type'],'started')
        self.assertEqual(m['source'],source or 'microphone_live')
        return m['session_id']

    def until(self,ws,kind):
        seen=[]
        while True:
            m=ws.receive_json();seen.append(m)
            if m['type']==kind:return seen
            if m['type']=='error':self.fail(m['message'])

    def test_stop_remainder_persist_feedback_and_restart(self):
        with self.connect() as ws:
            sid=self.start(ws);ws.send_bytes(packet(0));self.until(ws,'ack')
            ws.send_bytes(packet(0));self.assertTrue(self.until(ws,'ack')[-1]['duplicate'])
            ws.send_json({'type':'stop','last_sequence':0});events=self.until(ws,'completed')
        self.assertEqual(len([m for m in events if m['type']=='result']),1)
        detail=self.client.get('/sessions/'+sid).json()
        self.assertEqual(detail['source'],'microphone_live');self.assertEqual(detail['status'],'completed')
        unit=detail['classification_units'][0]
        self.assertEqual(self.client.post('/segments/'+unit['segment_id']+'/feedback',json={'label':'CANH_BAO','note':'Technical test'}).status_code,201)
        self.client.__exit__(None,None,None)
        self.client=TestClient(create_app(lambda:self.service,self.url));self.client.__enter__()
        self.assertEqual(len(self.client.get('/sessions/'+sid).json()['classification_units'][0]['feedback']),1)
        self.service.analyze_live.assert_called_once()
        self.assertEqual(events[-1]['backlog_seconds'],0)
        self.assertEqual(events[-1]['received_seconds'],events[-1]['silence_seconds']+events[-1]['processed_audio_seconds'])
        self.assertTrue(all(m['session_id']==sid for m in events))
        self.assertEqual(len(detail['live_windows']),1)

    def test_silence_only_completed_no_inference_no_backlog(self):
        with self.connect() as ws:
            sid=self.start(ws)
            for i in range(6):ws.send_bytes(packet(i,value=0));self.until(ws,'ack')
            ws.send_json({'type':'stop','last_sequence':5});final=self.until(ws,'completed')[-1]
        self.service.analyze_live.assert_not_called()
        self.assertEqual(final['silence_seconds'],3)
        self.assertEqual(final['backlog_seconds'],0)
        self.assertFalse(final['processing'])
        self.assertEqual(self.client.get('/sessions/'+sid).json()['transcript'],'')

    def test_browser_tab_source_tail_history_export_and_restart(self):
        with self.connect() as ws:
            sid=self.start(ws,'browser_tab_live')
            ws.send_bytes(packet(0,.32));self.until(ws,'ack')
            ws.send_json({'type':'stop','last_sequence':0});events=self.until(ws,'completed')
        detail=self.client.get('/sessions/'+sid).json()
        self.assertEqual(detail['source'],'browser_tab_live')
        self.assertEqual(detail['filename'],'Tab livestream')
        self.assertEqual(detail['transcript'],'Xin chào.')
        self.assertEqual(len(detail['live_windows']),1)
        self.assertEqual(detail['live_windows'],detail['asr_windows'])
        self.assertEqual(detail['status'],'completed')
        self.assertEqual(events[-1]['received_seconds'],.32)
        self.assertEqual(events[-1]['backlog_seconds'],0)
        self.assertEqual(self.client.get('/sessions/'+sid+'/export').json(),detail)
        self.client.__exit__(None,None,None)
        self.client=TestClient(create_app(lambda:self.service,self.url));self.client.__enter__()
        self.assertEqual(self.client.get('/sessions/'+sid).json(),detail)

    def test_source_validation_rejects_before_creating_history(self):
        for source in ('upload','recording','unknown',None,[],{},1):
            with self.subTest(source=source),self.connect() as ws:
                ws.send_json(dict(type='start',format='pcm_s16le',sample_rate=16000,
                                  channels=1,source=source))
                self.assertIn('Nguồn giám sát',ws.receive_json()['message'])
        self.assertEqual(self.client.get('/sessions').json()['total'],0)

    def test_tab_and_microphone_use_one_gate_and_isolated_sessions(self):
        self.service.analyze_live.side_effect=[result('Từ tab.'),result('Từ microphone.')]
        with self.connect() as ws:
            tab_id=self.start(ws,'browser_tab_live')
            with self.connect() as other:
                self.assertIn('bận',other.receive_json()['message'])
            self.assertEqual(self.client.post('/analyze-audio',files={'file':('x.wav',b'audio')}).status_code,409)
            ws.send_bytes(packet(0));self.until(ws,'ack')
            ws.send_json({'type':'stop','last_sequence':0});self.until(ws,'completed')
        with self.connect() as ws:
            mic_id=self.start(ws,'microphone_live')
            ws.send_bytes(packet(0));self.until(ws,'ack')
            ws.send_json({'type':'stop','last_sequence':0});self.until(ws,'completed')
        self.assertNotEqual(tab_id,mic_id)
        tab=self.client.get('/sessions/'+tab_id).json()
        mic=self.client.get('/sessions/'+mic_id).json()
        self.assertEqual((tab['source'],tab['transcript']),('browser_tab_live','Từ tab.'))
        self.assertEqual((mic['source'],mic['transcript']),('microphone_live','Từ microphone.'))
        self.assertTrue(all(call.args[3] is None for call in self.service.analyze_live.call_args_list))

    def test_accelerated_thirty_minute_tab_pcm_accounting_and_tail(self):
        # Transport/accounting stress only: synthetic PCM and mocked VAD/ASR.
        # 30 minutes are submitted faster than wall time, not a latency benchmark.
        def answer(path,start,boundary,carry,**kwargs):
            output=result(f"Đoạn {self.service.analyze_live.call_count}.")
            output['units'][0].update(start=start,end=start+.5)
            return output
        self.service.analyze_live.side_effect=answer
        max_backlog=0
        with self.connect() as ws:
            sid=self.start(ws,'browser_tab_live')
            for index in range(3600):
                voice=index % 40 < 2 or index == 3599
                ws.send_bytes(packet(index,value=1000 if voice else 0))
                messages=self.until(ws,'ack')
                max_backlog=max(max_backlog,*(m.get('backlog_seconds',0) for m in messages))
            ws.send_json({'type':'stop','last_sequence':3599})
            completed=self.until(ws,'completed')[-1]
        detail=self.client.get('/sessions/'+sid).json()
        units=detail['classification_units']
        self.assertEqual(completed['received_seconds'],1800)
        self.assertEqual(completed['backlog_seconds'],0)
        self.assertEqual(completed['pending_windows'],0)
        self.assertEqual(completed['received_seconds'],
                         completed['processed_audio_seconds']+completed['silence_seconds'])
        self.assertEqual(self.service.analyze_live.call_count,91)
        self.assertEqual(len(units),91)
        self.assertEqual(len({unit['unit_id'] for unit in units}),91)
        self.assertEqual(len({unit['text'] for unit in units}),91)
        self.assertEqual(units[-1]['text'],'Đoạn 91.')
        self.assertGreaterEqual(units[-1]['start'],1799)
        self.assertLessEqual(units[-1]['end'],1800)
        self.assertLessEqual(completed['max_pending_windows'],3)
        self.assertLess(max_backlog,12*4)

    def test_two_sessions_no_context_or_output_crossing(self):
        self.service.analyze_live.side_effect=[result('Câu thứ nhất.'), result('Câu thứ hai.')]
        ids=[]
        for _ in range(2):
            with self.connect() as ws:
                ids.append(self.start(ws));ws.send_bytes(packet(0));self.until(ws,'ack')
                ws.send_json({'type':'stop','last_sequence':0});self.until(ws,'completed')
        self.assertNotEqual(*ids)
        self.assertEqual(self.client.get('/sessions/'+ids[0]).json()['transcript'],'Câu thứ nhất.')
        self.assertEqual(self.client.get('/sessions/'+ids[1]).json()['transcript'],'Câu thứ hai.')
        self.assertTrue(all(call.args[3] is None for call in self.service.analyze_live.call_args_list))

    def test_late_inference_after_disconnect_is_discarded(self):
        import time
        entered=threading.Event();release=threading.Event()
        def slow(*args,**kwargs):entered.set();release.wait(3);return result('Kết quả phiên cũ.')
        self.service.analyze_live.side_effect=slow
        with self.connect() as ws:
            sid=self.start(ws)
            for i,value in enumerate([1000,0,0,0]):
                ws.send_bytes(packet(i,value=value));self.until(ws,'ack')
            self.assertTrue(entered.wait(2))
            ws.close()
            time.sleep(.05)
            release.set()
        for _ in range(100):
            detail=self.client.get('/sessions/'+sid).json()
            if detail['status']=='failed':break
            time.sleep(.01)
        self.assertEqual(detail['status'],'failed')
        self.assertEqual(detail['classification_units'],[])
        self.service.analyze_live.side_effect=lambda *args,**kwargs:result('Phiên mới.')
        with self.connect() as ws:
            new=self.start(ws);ws.send_bytes(packet(0));self.until(ws,'ack')
            ws.send_json({'type':'stop','last_sequence':0});self.until(ws,'completed')
        self.assertEqual(self.client.get('/sessions/'+new).json()['transcript'],'Phiên mới.')

    def test_debug_is_opt_in_saves_exact_asr_audio_with_ids(self):
        with tempfile.TemporaryDirectory() as directory,patch('backend.live_audio.ROOT',Path(directory)):
            with self.connect() as ws:
                ws.send_json(dict(type='start',format='pcm_s16le',sample_rate=16000,channels=1,debug_audio=True))
                opened=ws.receive_json();sid=opened['session_id']
                ws.send_bytes(packet(0));self.until(ws,'ack')
                ws.send_json({'type':'stop','last_sequence':0});self.until(ws,'completed')
            import json,wave
            folder=Path(directory)/opened['debug_directory']
            with wave.open(str(folder/'received.wav')) as recording:self.assertEqual(recording.readframes(RATE),packet(0)[4:])
            with wave.open(str(folder/'window_0.wav')) as window:self.assertEqual(window.readframes(RATE),packet(0)[4:])
            trace=json.loads((folder/'windows.jsonl').read_text())
            self.assertEqual(trace['session_id'],sid);self.assertEqual(trace['chunk_start'],0)
            self.assertEqual(trace['sample_rate'],16000)

    def test_busy_other_live_and_upload(self):
        with self.connect() as ws:
            self.start(ws)
            with self.connect() as other:self.assertIn('bận',other.receive_json()['message'])
            self.assertEqual(self.client.post('/analyze-audio',files={'file':('x.wav',b'audio')}).status_code,409)
            ws.send_json({'type':'stop','last_sequence':-1});self.until(ws,'completed')

    def test_invalid_order_and_format(self):
        with self.connect() as ws:
            sid=self.start(ws);ws.send_bytes(packet(2));self.until_error(ws)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['status'],'failed')
        with self.connect() as ws:
            ws.send_json({'type':'start','format':'webm'})
            self.assertEqual(ws.receive_json()['type'],'error')

    def until_error(self,ws):
        while True:
            m=ws.receive_json()
            if m['type']=='error':return m

    def test_disconnect_keeps_completed_units_new_session_allowed(self):
        with self.connect() as ws:
            sid=self.start(ws)
            for i,value in enumerate([1000,0,0,0]):
                ws.send_bytes(packet(i,value=value));self.until(ws,'ack')
            self.until(ws,'result')
        # A normal HTTP read reaches the loop after the disconnect handler finishes.
        import time
        for _ in range(100):
            detail=self.client.get('/sessions/'+sid).json()
            if detail['status']=='failed':break
            time.sleep(.01)
        self.assertEqual(detail['status'],'failed');self.assertEqual(len(detail['classification_units']),1)
        with self.connect() as ws:
            self.start(ws);ws.send_json({'type':'stop','last_sequence':-1});self.until(ws,'completed')

    def test_result_retry_idempotent_conflict_rejected(self):
        sid=history.start_session(self.app.state.database,'mic','microphone_live',self.service.health())
        units=result()['units'];units[0]['unit_id']='LIVE_0'
        history.append_live_result(self.app.state.database,sid,units,1,1)
        history.append_live_result(self.app.state.database,sid,units,1,1)
        units[0]['text']='Conflicting result'
        with self.assertRaises(ValueError):history.append_live_result(self.app.state.database,sid,units,1,1)
        self.assertEqual(len(self.client.get('/sessions/'+sid).json()['classification_units']),1)

    def test_bounded_queue_overload_drains_accepted_audio(self):
        release=threading.Event()
        def slow(*args,**kwargs):release.wait(5);return result()
        self.service.analyze_live.side_effect=slow
        with patch('backend.live_audio.QUEUE_WINDOWS',1),patch('backend.live_audio.WINDOW_SECONDS',1):
            with self.connect() as ws:
                sid=self.start(ws)
                messages=[]
                for i in range(8):
                    ws.send_bytes(packet(i))
                    messages.extend(self.until(ws,'ack'))
                    if any(m['max_pending_windows']==1 for m in messages if 'max_pending_windows' in m) and i>=5:break
                release.set()
                final=self.until_error(ws)
        detail=self.client.get('/sessions/'+sid).json()
        self.assertEqual(detail['status'],'failed');self.assertIn('quá tải',detail['error'])
        self.assertLessEqual(final['max_pending_windows'],1)
        self.assertGreater(len(detail['classification_units']),0)

    def test_worker_failure_does_not_deadlock_draining(self):
        self.service.analyze_live.side_effect=RuntimeError('PRIVATE/path')
        with self.assertLogs('backend.live_audio',level='ERROR'):
            with self.connect() as ws:
                sid=self.start(ws);ws.send_bytes(packet(0));self.until(ws,'ack')
                ws.send_json({'type':'stop','last_sequence':0});error=self.until_error(ws)
        self.assertNotIn('PRIVATE',str(error))
        self.assertFalse(error['processing'])
        self.assertEqual(error['pending_windows'],0)
        self.assertEqual(error['unprocessed_audio_seconds'],.5)
        self.assertEqual(error['backlog_seconds'],.5)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['status'],'failed')


class MigrationTests(unittest.TestCase):
    def test_browser_source_migration_preserves_history_and_refuses_lossy_downgrade(self):
        from alembic import command
        from alembic.config import Config
        from backend.database import ROOT
        from backend.database_transfer import snapshot, verify_rows
        with tempfile.TemporaryDirectory() as directory:
            db=Database('sqlite:///'+(Path(directory)/'source.sqlite3').as_posix())
            config=Config(str(ROOT/'alembic.ini'))
            with db.engine.begin() as conn:
                config.attributes['connection']=conn;command.upgrade(config,'0005_training_usage')
            old_id=history.start_session(db,'Existing mic','microphone_live',
                dict(model_version='binary_v04',checkpoint='model/best'))
            with db.engine.connect() as conn:before=snapshot(conn)
            db.migrate()
            with db.engine.connect() as conn:
                self.assertTrue(verify_rows(before,snapshot(conn))['all_source_rows_equal'])
            self.assertEqual(history.session_detail(db,old_id)['source'],'microphone_live')
            tab_id=history.start_session(db,'Tab livestream','browser_tab_live',
                dict(model_version='binary_v04',checkpoint='model/best'))
            with self.assertRaisesRegex(RuntimeError,'Browser-tab history exists'):
                with db.engine.begin() as conn:
                    config.attributes['connection']=conn;command.downgrade(config,'0005_training_usage')
            self.assertEqual(history.session_detail(db,tab_id)['source'],'browser_tab_live')
            db.migrate()
            db.close()

    def test_sqlite_parent_rebuild_preserves_old_segments_and_feedback(self):
        from alembic import command
        from alembic.config import Config
        from backend.database import ROOT, Feedback, new_id, utc_now
        from backend.schemas import AudioResponse
        with tempfile.TemporaryDirectory() as directory:
            db=Database('sqlite:///'+(Path(directory)/'old.sqlite3').as_posix())
            config=Config(str(ROOT/'alembic.ini'))
            with db.engine.begin() as conn:
                config.attributes['connection']=conn;command.upgrade(config,'0001_history')
            sid=history.start_session(db,'old.m4a','upload',dict(model_version='binary_v04',checkpoint='model/best'))
            response=AudioResponse(filename='old.m4a',processing_time_seconds=1,classification_mode='binary',transcript='Xin chào.',
                classification_units=result()['units'],summary=dict(whisper_segment_count=1,classification_unit_count=1,final_label_counts={'KHONG_CANH_BAO':1,'CANH_BAO':0}))
            history.complete_session(db,sid,response)
            # Seed the actual v0001 schema, not the current ORM (which has v0004 fields).
            from sqlalchemy import text
            with db.engine.begin() as conn:
                conn.execute(text('INSERT INTO feedback (id,segment_id,label,note,created_at) VALUES (:id,:segment,:label,:note,:created)'),
                    dict(id=new_id(),segment=response.classification_units[0].segment_id,label='KHONG_CANH_BAO',note='Original',created=utc_now()))
            from backend.database_transfer import snapshot, verify_rows
            with db.engine.connect() as conn:before=snapshot(conn)
            db.migrate()
            with db.engine.connect() as conn:self.assertTrue(verify_rows(before,snapshot(conn))['all_source_rows_equal'])
            self.assertEqual(len(history.session_detail(db,sid)['classification_units'][0]['feedback']),1)
            db.close()


if __name__=='__main__':unittest.main()
