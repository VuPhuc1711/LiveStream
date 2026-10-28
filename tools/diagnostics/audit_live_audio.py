"""Technical audio/trace audit, never a classifier benchmark or training input."""
import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
from importlib.metadata import distribution
import numpy as np
import torch
import webrtcvad
from backend.neural_vad import NeuralVad
from backend.speech_buffer import UtteranceBuffer


def read_pcm(path):
    with wave.open(str(path)) as audio:
        spec=dict(sample_rate=audio.getframerate(), channels=audio.getnchannels(),
                  sample_width=audio.getsampwidth(), samples=audio.getnframes())
        pcm=audio.readframes(audio.getnframes())
    assert spec['sample_rate']==16000 and spec['channels']==1 and spec['sample_width']==2
    return pcm,spec


def signal_stats(pcm):
    x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    fft=np.abs(np.fft.rfft(x-x.mean()))**2
    freq=np.fft.rfftfreq(len(x),1/16000)
    total=max(float(fft.sum()),1e-20)
    return dict(rms=float(np.sqrt(np.mean(x*x))), peak=float(np.abs(x).max()),
        dc=float(x.mean()), clipped_samples=int(np.sum(np.abs(x)>=.999)),
        power_below_300hz=float(fft[freq<300].sum()/total),
        power_300_4000hz=float(fft[(freq>=300)&(freq<=4000)].sum()/total))


def neural_scores(pcm):
    model_path=Path(distribution('silero-vad').locate_file('silero_vad/data/silero_vad.jit'))
    model=torch.jit.load(str(model_path),map_location='cpu').eval()
    model.reset_states()
    values=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    scores=[]
    with torch.inference_mode():
        for i in range(0,len(values),512):
            scores.append(float(model(torch.from_numpy(np.pad(values[i:i+512],(0,max(0,512-len(values[i:i+512]))))),16000)))
    return scores,dict(version=distribution('silero-vad').version,
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), frame_samples=512)


def audit(folder):
    pcm,spec=read_pcm(folder/'received.wav')
    chunks=[json.loads(line) for line in (folder/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    cursor=0
    for index,chunk in enumerate(chunks):
        part=pcm[cursor*2:(cursor+chunk['samples'])*2]
        assert chunk['chunk_id']==index and hashlib.sha256(part).hexdigest()==chunk['pcm_sha256']
        cursor+=chunk['samples']
    assert cursor==spec['samples']
    windows=[]
    traces=[json.loads(line) for line in (folder/'windows.jsonl').read_text(encoding='utf-8').splitlines()]
    for trace in traces:
        if trace['samples']:
            part,_=read_pcm(folder/trace['debug_audio_file'])
            start=round(trace['start']*16000)
            assert part==pcm[start*2:(start+trace['samples'])*2]
            assert hashlib.sha256(part).hexdigest()==trace['pcm_sha256']
            windows.append(dict(window_id=trace['window_id'],duration=trace['samples']/16000,
                webrtc_voiced_seconds=trace['voiced_seconds'],stats=signal_stats(part),
                raw_asr=trace['whisper_raw'],unit_ids=trace['unit_ids']))
    conversion=subprocess.run(['ffmpeg','-v','error','-i',str(folder/'received.wav'),'-f','f32le',
        '-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
    expected=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    decoded=np.frombuffer(conversion.stdout,dtype='<f4')
    assert len(decoded)==len(expected)
    scores,model=neural_scores(pcm)
    vad=webrtcvad.Vad(2)
    webrtc_seconds=sum(.02 for i in range(0,len(pcm)-639,640)
        if np.std(np.frombuffer(pcm[i:i+640],dtype='<i2'))/32768>=.001 and vad.is_speech(pcm[i:i+640],16000))
    detector=NeuralVad()
    buffer=UtteranceBuffer(vad=detector)
    admitted=buffer.feed(pcm)+buffer.finish()
    assert sum(len(w.pcm)//2 for w in admitted)+buffer.silence_samples==spec['samples']
    return dict(session_id=folder.name,user_confirmed_no_speech=folder.name=='b892bf34-35af-4d2b-89f0-846f6e571ec3',audio_spec=spec,
        audio_sha256=hashlib.sha256((folder/'received.wav').read_bytes()).hexdigest(),
        chunk_hashes_and_window_slices_match=True,ffmpeg_pcm_max_difference=float(np.max(np.abs(expected-decoded))),
        stats=signal_stats(pcm), old_window_seconds=sum(t['samples']/16000 for t in traces),
        old_window_voiced_seconds=sum(t['voiced_seconds'] for t in traces),
        webrtc_frame_voiced_seconds=webrtc_seconds,neural_vad=model,
        neural_probability_max=max(scores),neural_probability_percentiles=np.percentile(scores,[50,90,99]).tolist(),
        neural_frames_ge_05=sum(p>=.5 for p in scores), neural_frames_ge_035=sum(p>=.35 for p in scores),
        new_admission=dict(config=detector.configuration(),windows=len(admitted),
            asr_audio_seconds=sum(len(w.pcm)/32000 for w in admitted),
            silence_seconds=buffer.silence_samples/16000,all_samples_accounted=True),
        windows=windows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report=audit(args.folder)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='windows'},ensure_ascii=False,indent=2))
