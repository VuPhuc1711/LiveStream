import { useEffect, useRef, useState } from 'react';
import { validateFile } from './api';

export function recordingExtension(mime: string) {
  const type = mime.split(';')[0].trim().toLowerCase();
  return ({ 'audio/webm': 'webm', 'video/webm': 'webm', 'audio/mp4': 'm4a', 'video/mp4': 'mp4', 'audio/wav': 'wav', 'audio/x-wav': 'wav' } as Record<string,string>)[type];
}
export function microphoneError(error: unknown) {
  const name = typeof error === 'object' && error !== null && 'name' in error ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'Quyền microphone bị từ chối. Cho phép microphone trong trình duyệt rồi thử lại.';
  if (name === 'NotFoundError') return 'Không tìm thấy microphone. Hãy kết nối thiết bị rồi thử lại.';
  return 'Ghi âm bị gián đoạn hoặc microphone đang bận. Kiểm tra thiết bị rồi thu lại.';
}

export function useRecorder(onFile: (file: File | null) => void, onError: (error: string) => void) {
  const [phase, setPhase] = useState<'idle'|'requesting'|'recording'|'stopping'>('idle');
  const [seconds, setSeconds] = useState(0);
  const [url, setUrl] = useState('');
  const active = useRef(false), mounted = useRef(false);
  const stream = useRef<MediaStream | null>(null), recorder = useRef<MediaRecorder | null>(null);
  const objectUrl = useRef(''), timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const fileCallback = useRef(onFile), errorCallback = useRef(onError);
  fileCallback.current = onFile; errorCallback.current = onError;
  function releaseTracks() { stream.current?.getTracks().forEach(track => { track.onended = null; track.stop(); }); stream.current = null; clearInterval(timer.current); }
  function clear() { if (objectUrl.current) URL.revokeObjectURL(objectUrl.current); objectUrl.current = ''; setUrl(''); }
  useEffect(() => {
    mounted.current = true;
    const dispose = () => {
      mounted.current = false;
      const current = recorder.current;
      if (current) { current.ondataavailable = null; current.onstop = null; current.onerror = null; if (current.state !== 'inactive') current.stop(); }
      releaseTracks();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = ''; active.current = false;
    };
    window.addEventListener('pagehide', dispose);
    return () => { window.removeEventListener('pagehide', dispose); dispose(); };
  }, []);
  async function start() {
    if (active.current) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      errorCallback.current('Trình duyệt không hỗ trợ ghi âm. Hãy dùng Chrome/Edge trên localhost hoặc HTTPS.'); return;
    }
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/wav'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) { errorCallback.current('Trình duyệt không hỗ trợ định dạng ghi âm phù hợp. Hãy tải file lên.'); return; }
    active.current = true; setPhase('requesting'); errorCallback.current('');
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mounted.current) { media.getTracks().forEach(track => track.stop()); return; }
      stream.current = media;
      const current = new MediaRecorder(media, { mimeType: mime }); recorder.current = current;
      const actualMime = current.mimeType, extension = recordingExtension(actualMime);
      if (!extension) throw new Error('Unsupported recorder MIME');
      let chunks: Blob[] = [], size = 0, failure = '', requestedStop = false, stoppedAt = 0;
      const started = Date.now();
      const fail = (message: string) => { failure = message; if (current.state !== 'inactive') current.stop(); releaseTracks(); };
      current.ondataavailable = event => {
        if (!mounted.current || failure || !event.data.size) return;
        size += event.data.size;
        if (size > 50 * 1024 * 1024) { chunks = []; fail('Bản thu vượt giới hạn 50 MiB. Hãy thu một đoạn ngắn hơn.'); return; }
        chunks.push(event.data);
      };
      current.onerror = () => fail('Ghi âm bị gián đoạn. Hãy kiểm tra microphone và thu lại.');
      media.getAudioTracks().forEach(track => { track.onended = () => fail('Microphone đã ngắt kết nối. Hãy kết nối và thu lại.'); });
      current.onstop = () => {
        releaseTracks(); active.current = false; recorder.current = null;
        if (!mounted.current) return;
        setPhase('idle');
        if (!requestedStop && !failure) failure = 'Ghi âm bị gián đoạn. Vui lòng thu lại.';
        if (failure) { errorCallback.current(failure); return; }
        const blob = new Blob(chunks, { type: actualMime }); chunks = [];
        if (stoppedAt - started < 1000 || !blob.size) { errorCallback.current('Bản thu quá ngắn hoặc rỗng. Hãy thu ít nhất 1 giây và nói rõ vào microphone.'); return; }
        const file = new File([blob], `ban-thu-${Date.now()}.${extension}`, { type: actualMime });
        const problem = validateFile(file); if (problem) { errorCallback.current(problem); return; }
        clear(); objectUrl.current = URL.createObjectURL(blob); setUrl(objectUrl.current); fileCallback.current(file);
      };
      // Mark only explicit user stops as successful, never unexpected recorder stops.
      stopAction.current = () => { if (requestedStop) return; requestedStop = true; stoppedAt = Date.now(); setPhase('stopping'); if (current.state !== 'inactive') current.stop(); releaseTracks(); };
      current.start(250);
      clear(); fileCallback.current(null); setSeconds(0); setPhase('recording');
      timer.current = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250);
    } catch (error) {
      releaseTracks(); active.current = false;
      if (mounted.current) { setPhase('idle'); errorCallback.current(microphoneError(error)); }
    }
  }
  const stopAction = useRef<() => void>(() => {});
  return { phase, seconds, url, active, start, stop: () => stopAction.current(), clear };
}
