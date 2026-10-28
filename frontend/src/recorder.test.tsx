// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import App from './App';
import { recordingExtension } from './useRecorder';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
class Recorder {
  static isTypeSupported = (type: string) => type.startsWith('audio/webm');
  static current: Recorder;
  mimeType = 'audio/webm;codecs=opus'; state = 'inactive';
  ondataavailable: ((event: {data: Blob})=>void)|null = null;
  onstop: (()=>void)|null = null; onerror: (()=>void)|null = null;
  constructor() { Recorder.current = this; }
  start() { this.state = 'recording'; }
  stop() { this.state = 'inactive'; this.ondataavailable?.({data:new Blob(['actual bytes'])}); this.onstop?.(); }
}
let host: HTMLDivElement, root: Root;
let track: {stop: ReturnType<typeof vi.fn>; onended: (()=>void)|null};
let permission: ReturnType<typeof vi.fn>, fetcher: ReturnType<typeof vi.fn>;
const response = {filename:'recording.webm',processing_time_seconds:1,transcript:'Xin chào',classification_units:[],summary:{classification_unit_count:0,whisper_segment_count:0,final_label_counts:{}}};
beforeEach(async()=>{
  vi.useFakeTimers(); host=document.createElement('div'); document.body.append(host); root=createRoot(host);
  track={stop:vi.fn(),onended:null}; permission=vi.fn().mockResolvedValue({getTracks:()=>[track],getAudioTracks:()=>[track]});
  vi.stubGlobal('navigator',{mediaDevices:{getUserMedia:permission}}); vi.stubGlobal('MediaRecorder',Recorder);
  vi.stubGlobal('URL',Object.assign(URL,{createObjectURL:vi.fn(()=> 'blob:recording'),revokeObjectURL:vi.fn()}));
  fetcher=vi.fn().mockImplementation((url:string)=>Promise.resolve({ok:true,json:async()=>url.endsWith('/health')?{ready:true}:response}));
  vi.stubGlobal('fetch',fetcher); await act(async()=>root.render(<App/>));
});
afterEach(async()=>{ await act(async()=>root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.useRealTimers(); });
const button=(text:string)=>Array.from(host.querySelectorAll('button')).find(b=>b.textContent===text)!;
const click=async(text:string)=>act(async()=>button(text).click());
it('start → stop → playback → analyze → results, cleans URL and tracks',async()=>{
  expect(permission).not.toHaveBeenCalled(); await click('Bắt đầu thu âm');
  expect(permission).toHaveBeenCalledWith({audio:true}); expect(host.textContent).toContain('Đang thu âm');
  expect((host.querySelector('input[type=file]') as HTMLInputElement).disabled).toBe(true);
  await act(async()=>vi.advanceTimersByTime(2000)); await click('Dừng thu âm');
  expect(track.stop).toHaveBeenCalled(); expect(host.querySelector('audio')?.getAttribute('src')).toBe('blob:recording');
  expect(host.textContent).toContain('audio/webm;codecs=opus'); await click('Phân tích bản thu');
  const post=fetcher.mock.calls.find(call=>call[0].endsWith('/analyze-audio'))!;
  const file=post[1].body.get('file') as File; expect(file.name).toMatch(/\.webm$/); expect(file.type).toBe('audio/webm;codecs=opus');
  expect(host.textContent).toContain('Kết quả phân tích'); await click('Xóa bản thu'); expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:recording'); expect(host.querySelector('audio')).toBeNull();
});
it('rejects early stop and interruption',async()=>{
  await click('Bắt đầu thu âm'); await click('Dừng thu âm'); expect(host.textContent).toContain('quá ngắn'); expect(host.querySelector('audio')).toBeNull();
  await click('Bắt đầu thu âm'); await act(async()=>track.onended?.()); expect(host.textContent).toContain('ngắt kết nối'); expect(track.stop).toHaveBeenCalled();
});
it.each([['NotAllowedError','bị từ chối'],['NotFoundError','Không tìm thấy'],['NotReadableError','bị gián đoạn']])('reports %s',async(name,message)=>{
  permission.mockRejectedValueOnce(new DOMException('',name)); await click('Bắt đầu thu âm'); expect(host.textContent).toContain(message);
});
it('reports unsupported recording and maps actual MIME without disguising container',async()=>{
  vi.stubGlobal('MediaRecorder',undefined); await click('Bắt đầu thu âm'); expect(host.textContent).toContain('không hỗ trợ ghi âm');
  expect(recordingExtension('audio/mp4;codecs=mp4a.40.2')).toBe('m4a'); expect(recordingExtension('audio/ogg')).toBeUndefined();
});
it('stops tracks on unmount including permission resolving after unmount',async()=>{
  let resolve!: (stream:unknown)=>void; permission.mockReturnValueOnce(new Promise(r=>{resolve=r;}));
  await click('Bắt đầu thu âm'); await act(async()=>root.unmount());
  await act(async()=>resolve({getTracks:()=>[track]})); expect(track.stop).toHaveBeenCalled();
});
it('discards a recording beyond 50 MiB and releases microphone',async()=>{
  await click('Bắt đầu thu âm'); await act(async()=>Recorder.current.ondataavailable?.({data:{size:50*1024*1024+1} as Blob}));
  expect(host.textContent).toContain('vượt giới hạn 50 MiB'); expect(track.stop).toHaveBeenCalled(); expect(host.querySelector('audio')).toBeNull();
});
it('revokes previous playback on re-record and stops active tracks on page exit',async()=>{
  await click('Bắt đầu thu âm'); await act(async()=>vi.advanceTimersByTime(1500)); await click('Dừng thu âm'); await click('Thu lại');
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:recording'); await act(async()=>window.dispatchEvent(new Event('pagehide')));
  expect(track.stop).toHaveBeenCalled(); expect(Recorder.current.state).toBe('inactive');
});
