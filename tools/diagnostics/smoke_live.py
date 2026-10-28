"""Opt-in technical smoke: real audio paced at playback speed, no accuracy scoring."""
import argparse
import asyncio
import hashlib
import json
import struct
import subprocess
import time
import sys
from pathlib import Path
import websockets


async def run(audio, leading_silence=0, trailing_silence=0, debug=False, silence_only=0, compare_upload=False):
    if silence_only:
        pcm=bytes(round(silence_only*16000)*2)
    else:
        conversion = subprocess.run(['ffmpeg','-v','error','-i',str(audio),'-f','s16le',
            '-acodec','pcm_s16le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
        pcm=bytes(round(leading_silence*16000)*2)+conversion.stdout+bytes(round(trailing_silence*16000)*2)
    duration=len(pcm)/32000
    async with websockets.connect('ws://127.0.0.1:8000/ws/monitor',origin='http://127.0.0.1:5173',max_size=2**20) as ws:
        await ws.send(json.dumps(dict(type='start',format='pcm_s16le',sample_rate=16000,channels=1,debug_audio=debug)))
        opened=json.loads(await ws.recv())
        if opened['type']!='started':raise RuntimeError(opened)
        started=time.perf_counter();events=[];first_final=None;sent_all=None
        async def receive():
            nonlocal first_final
            async for raw in ws:
                event=json.loads(raw)
                event['client_elapsed_seconds']=time.perf_counter()-started
                events.append(event)
                if event['type']=='result' and event['classification_units'] and first_final is None:
                    first_final=event['client_elapsed_seconds']
                    print('First finalized result at',round(first_final,2),'seconds',flush=True)
                if event['type'] in {'completed','error'}:return event
        receiving=asyncio.create_task(receive())
        sequence=-1
        for sequence,cursor in enumerate(range(0,len(pcm),16000)):
            # Send a frame only after its real-time capture duration has elapsed.
            deadline=started+min((cursor+16000)/32000,duration)
            await asyncio.sleep(max(0,deadline-time.perf_counter()))
            if receiving.done():raise RuntimeError('Session ended before entire recording was sent')
            await ws.send(struct.pack('<I',sequence)+pcm[cursor:cursor+16000])
        sent_all=time.perf_counter()-started
        await ws.send(json.dumps({'type':'stop','last_sequence':sequence}))
        end=await asyncio.wait_for(receiving,timeout=180)
        if end['type']!='completed':raise RuntimeError(end)
        results=[m for m in events if m['type']=='result']
        report=dict(audio=audio.name if audio else 'generated_silence',audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest() if audio else None,
            physical_microphone=False, leading_silence=leading_silence,trailing_silence=trailing_silence,
            pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            session_id=opened['session_id'],model_version=opened['model_version'],checkpoint=opened['checkpoint'],
            audio_seconds=duration,first_final_seconds=first_final,all_audio_sent_seconds=sent_all,
            result_before_audio_finished=first_final is not None and first_final<sent_all,
            max_pending_windows=max(m.get('max_pending_windows',0) for m in events),
            max_backlog_seconds=max(m.get('backlog_seconds',0) for m in events),
            metrics_timeline=[{k:m[k] for k in ('type','client_elapsed_seconds','received_seconds',
                'backlog_seconds','pending_windows','assembling_seconds','queued_seconds',
                'processing_audio_seconds','processed_audio_seconds','silence_seconds',
                'unprocessed_audio_seconds') if k in m} for m in events],
            results=[{k:m[k] for k in ('client_elapsed_seconds','processing_time_seconds','speech_end_seconds','latency_seconds','pending_windows','context_limit','final_unit_latencies_seconds','latency_timestamp_is_approximate') if k in m} for m in results],
            unit_count=end['unit_count'],status=end['type'], final_metrics=end,
            debug_directory=opened.get('debug_directory'))
        import httpx
        async with httpx.AsyncClient(timeout=180) as client:
            detail=await client.get('http://127.0.0.1:8000/sessions/'+opened['session_id'])
            detail.raise_for_status();report['live_detail']=detail.json()
            if compare_upload:
                import io,wave
                contents=io.BytesIO()
                with wave.open(contents,'wb') as output:
                    output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000);output.writeframes(pcm)
                upload=await client.post('http://127.0.0.1:8000/analyze-audio',files={'file':('same_received_audio.wav',contents.getvalue(),'audio/wav')})
                upload.raise_for_status();report['same_audio_upload']=upload.json()
        assert end['backlog_seconds']==0 and not end['processing'] and end['pending_windows']==0
        assert end['received_seconds']==end['silence_seconds']+end['processed_audio_seconds']
        assert all(m.get('session_id')==opened['session_id'] for m in events)
        if silence_only:
            assert not report['live_detail']['transcript'] and not report['live_detail']['live_windows']
        return report


if __name__=='__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('audio',type=Path,nargs='?');parser.add_argument('--report',type=Path)
    parser.add_argument('--leading-silence',type=float,default=0);parser.add_argument('--trailing-silence',type=float,default=0)
    parser.add_argument('--debug',action='store_true');parser.add_argument('--silence-only',type=float,default=0)
    parser.add_argument('--compare-upload',action='store_true')
    args=parser.parse_args()
    if not args.audio and not args.silence_only:parser.error('Provide audio or --silence-only seconds')
    report=asyncio.run(run(args.audio,args.leading_silence,args.trailing_silence,args.debug,args.silence_only,args.compare_upload))
    if args.report:args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
