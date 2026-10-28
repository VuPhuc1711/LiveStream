// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach,afterEach,it,expect,vi } from 'vitest';
import App from './App';
import { timestamp } from './api';
// @ts-expect-error The browser loads this exact JS module in AudioWorklet.
import { PcmResampler } from '../public/pcm-worklet.js';
(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
class Socket {
  static OPEN=1; static current:Socket; readyState=1; bufferedAmount=0;
  onopen:(()=>void)|null=null; onmessage:((e:{data:string})=>void)|null=null;
  onclose:(()=>void)|null=null; onerror:(()=>void)|null=null;
  sent:unknown[]=[]; constructor(public url:string){Socket.current=this;}
  send(data:unknown){this.sent.push(data);} close(){this.readyState=3;this.onclose?.();}
  receive(value:unknown){this.onmessage?.({data:JSON.stringify({session_id:'live-session',...(value as object)})});}
}
class Context {
  static current:Context; state='running';sampleRate=48000;
  audioWorklet={addModule:vi.fn().mockResolvedValue(undefined)};
  resume=vi.fn().mockResolvedValue(undefined);close=vi.fn(async()=>{this.state='closed';});
  destination={};createMediaStreamSource(){return {connect:vi.fn(),disconnect:vi.fn()};}
  createGain(){return {gain:{value:1},connect:vi.fn(),disconnect:vi.fn()};}
  constructor(){Context.current=this;}
}
class Worklet {
  static current:Worklet;
  port={onmessage:null as ((e:{data:unknown})=>void)|null,postMessage:vi.fn((value:string)=>{
    if(value==='stop'){this.port.onmessage?.({data:new ArrayBuffer(32)});this.port.onmessage?.({data:'flushed'});}
  })};connect=vi.fn();disconnect=vi.fn();constructor(){Worklet.current=this;}
}
let root:Root,host:HTMLDivElement,permission:ReturnType<typeof vi.fn>,track:{stop:ReturnType<typeof vi.fn>;onended:(()=>void)|null};
const buttons=()=>Array.from(host.querySelectorAll('button'));
const click=async(text:string)=>act(async()=>buttons().find(b=>b.textContent===text)!.click());
beforeEach(async()=>{
  vi.useFakeTimers();host=document.createElement('div');document.body.append(host);root=createRoot(host);
  track={stop:vi.fn(),onended:null};permission=vi.fn().mockResolvedValue({getTracks:()=>[track],getAudioTracks:()=>[track]});
  vi.stubGlobal('navigator',{mediaDevices:{getUserMedia:permission}});
  vi.stubGlobal('AudioContext',Context);vi.stubGlobal('AudioWorkletNode',Worklet);vi.stubGlobal('WebSocket',Socket);
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({ready:true,classification_mode:'binary'})}));
  await act(async()=>root.render(<App/>));
});
afterEach(async()=>{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();vi.useRealTimers();});
const started=async()=>{
  await click('Bắt đầu giám sát');await act(async()=>{Socket.current.onopen?.();Socket.current.receive({type:'started',session_id:'live-session'});});
};
it('start → incremental official/temporary results → flush last frame → stop → feedback, preserves upload',async()=>{
  expect(permission).not.toHaveBeenCalled();await started();
  expect(permission).toHaveBeenCalledWith({audio:true});expect(host.textContent).toContain('Microphone đang bật');
  expect((host.querySelector('input[type=file]') as HTMLInputElement).disabled).toBe(true);
  expect(Context.current.audioWorklet.addModule).toHaveBeenCalledWith('/pcm-worklet.js');
  await act(async()=>Worklet.current.port.onmessage?.({data:new ArrayBuffer(16000)}));
  expect(new DataView(Socket.current.sent[1] as ArrayBuffer).getUint32(0,true)).toBe(0);
  const unit={unit_id:'LIVE_0',segment_id:'seg1',source_segment_id:0,text:'Rẻ nhất toàn sàn.',start:0,end:2,timestamp_is_approximate:true,final_label:'CANH_BAO',final_confidence:.9,phobert_result:{label:'CANH_BAO',confidence:.9,probabilities:{CANH_BAO:.9,KHONG_CANH_BAO:.1}},explanation:'PhoBERT.'};
  await act(async()=>Socket.current.receive({type:'result',classification_units:[unit],temporary_transcript:'Câu đang nói',pending_windows:1,latency_seconds:2,processing_time_seconds:1}));
  expect(host.textContent).toContain('Rẻ nhất toàn sàn.');expect(host.textContent).toContain('chưa phân loại/chưa lưu');
  expect(buttons().some(b=>b.textContent==='Dự đoán đúng')).toBe(false);
  // Duplicate delivery must not double the final units in UI.
  await act(async()=>Socket.current.receive({type:'result',classification_units:[unit]}));
  expect(host.textContent).toContain('1 cảnh báo / 1 câu');
  await click('Kết thúc');expect(track.stop).toHaveBeenCalled();expect(host.textContent).toContain('Đang xử lý phần còn lại');
  expect(JSON.parse(Socket.current.sent.at(-1) as string)).toEqual({type:'stop',last_sequence:1});
  await act(async()=>Socket.current.receive({type:'completed',session_id:'live-session'}));
  expect(host.textContent).toContain('Phiên đã hoàn tất');expect(buttons().some(b=>b.textContent==='Sửa nhãn')).toBe(true);
  expect((host.querySelector('input[type=file]') as HTMLInputElement).disabled).toBe(false);
});
it('ignores wrong-session and late terminal results; completed is not waiting',async()=>{
  await started();const old=Socket.current,late=old.onmessage;
  await act(async()=>old.receive({type:'result',session_id:'other-session',temporary_transcript:'Không thuộc phiên này'}));
  expect(host.textContent).not.toContain('Không thuộc phiên này');
  await act(async()=>old.receive({type:'completed',backlog_seconds:0,pending_windows:0,processing:false,silence_seconds:3,processed_audio_seconds:1,received_seconds:4}));
  expect(host.textContent).toContain('Đã xử lý hết audio nhận được');
  expect(host.textContent).not.toContain('Chờ audio / kết quả');
  expect(host.textContent).toContain('Tồn đọng: 0.0');
  await started();
  await act(async()=>late?.({data:JSON.stringify({type:'result',session_id:'live-session',temporary_transcript:'Kết quả cũ trả muộn'})}));
  expect(host.textContent).not.toContain('Kết quả cũ trả muộn');
  expect(host.textContent).toContain('Microphone đang bật');
});
it('keeps unresolved ASR separate from classification, persists it after completion and clears it for a new session',async()=>{
  await started();
  expect(host.querySelector('.asr-quality')).toBeNull();
  const warning={window_id:'window-3',start:8.2,end:11.4,reason:'unresolved_decode',retried:true,pending_text:'Hãy subscribe cho kênh không được coi là lời nói'};
  await act(async()=>Socket.current.receive({type:'result',classification_units:[],asr_quality_warning:warning,asr_unresolved_windows:1,asr_recovered_windows:2}));
  expect(host.querySelector('.asr-quality')?.textContent).toContain('1 đoạn chưa nhận dạng đáng tin cậy');
  expect(host.querySelector('.asr-quality')?.textContent).toContain('00:08.200 – 00:11.400');
  expect(host.querySelector('.asr-quality')?.textContent).toContain('không được đưa vào phân loại');
  expect(host.querySelector('.asr-quality')?.textContent).toContain('nhận dạng lại 2 đoạn');
  expect(host.textContent).not.toContain(warning.pending_text);
  expect(host.querySelectorAll('article.unit')).toHaveLength(0);
  await act(async()=>Socket.current.receive({type:'result',asr_quality_warning:warning}));
  expect(host.querySelectorAll('.asr-quality li')).toHaveLength(1);
  await act(async()=>Socket.current.receive({type:'completed',backlog_seconds:0}));
  expect(host.textContent).toContain('Phiên đã hoàn tất');
  expect(host.querySelector('.asr-quality')?.textContent).toContain('không có nghĩa là không cảnh báo');
  await started();expect(host.querySelector('.asr-quality')).toBeNull();
});
it('debug audio starts disabled and is explicit per session',async()=>{
  await started();expect(JSON.parse(Socket.current.sent[0] as string).debug_audio).toBe(false);
  await act(async()=>Socket.current.receive({type:'completed'}));
  const debug=Array.from(host.querySelectorAll('label')).find(l=>l.textContent?.includes('Lưu audio debug'))!.querySelector('input')!;
  await act(async()=>debug.click());await started();
  expect(JSON.parse(Socket.current.sent[0] as string).debug_audio).toBe(true);
  await act(async()=>Socket.current.receive({type:'completed'}));
  expect((debug as HTMLInputElement).checked).toBe(false);
});
it.each([['NotAllowedError','bị từ chối'],['NotFoundError','Không tìm thấy']])('permission %s does not connect',async(name,text)=>{
  permission.mockRejectedValueOnce(new DOMException('',name));await click('Bắt đầu giám sát');expect(host.textContent).toContain(text);
  expect((host.querySelector('input[type=file]') as HTMLInputElement).disabled).toBe(false);
});
it('disconnect / overload / unmount clean microphone without reconnect',async()=>{
  await started();await act(async()=>Socket.current.receive({type:'draining',message:'Backend quá tải'}));
  expect(track.stop).toHaveBeenCalled();expect(host.textContent).toContain('quá tải');
  await act(async()=>Socket.current.receive({type:'error',message:'Phần còn lại chưa được xử lý',session_id:'live-session'}));
  expect(host.textContent).toContain('Phiên đã dừng do lỗi');await started();
  await act(async()=>Socket.current.close());expect(host.textContent).toContain('Mất kết nối');
  await started();await act(async()=>root.unmount());expect(Context.current.close).toHaveBeenCalled();expect(track.stop).toHaveBeenCalled();
});
it('bounds client queue, does not silently continue sending',async()=>{
  await started();Socket.current.bufferedAmount=256001;
  await act(async()=>Worklet.current.port.onmessage?.({data:new ArrayBuffer(16000)}));
  expect(host.textContent).toContain('Kết nối quá chậm');expect(track.stop).toHaveBeenCalled();expect(Socket.current.sent.length).toBe(1);
});
it('separates microphone/tab controls and shares the one-job lock through final drain',async()=>{
  const video={stop:vi.fn(),onended:null as (()=>void)|null,getSettings:()=>({displaySurface:'browser'})};
  const display=vi.fn().mockResolvedValue({getTracks:()=>[track,video],getAudioTracks:()=>[track],getVideoTracks:()=>[video]});
  Object.assign(navigator.mediaDevices,{getDisplayMedia:display});
  expect(host.querySelectorAll('[id^=live-heading-]').length).toBe(2);
  expect(host.textContent).toContain('Giám sát microphone');
  expect(host.textContent).toContain('Giám sát tab livestream');
  await click('Chọn tab livestream');
  expect(permission).not.toHaveBeenCalled();expect(display).toHaveBeenCalledTimes(1);
  expect(buttons().find(b=>b.textContent==='Bắt đầu giám sát')!.disabled).toBe(true);
  expect(buttons().find(b=>b.textContent==='Bắt đầu thu âm')!.disabled).toBe(true);
  await act(async()=>{Socket.current.onopen?.();Socket.current.receive({type:'started'});});
  expect(JSON.parse(Socket.current.sent[0] as string).source).toBe('browser_tab_live');
  expect(host.textContent).toContain('Đang nhận âm thanh tab');
  await act(async()=>video.onended?.());
  expect(JSON.parse(Socket.current.sent.at(-1) as string).type).toBe('stop');
  expect(host.textContent).toContain('Thu tab đã dừng');
  expect(buttons().find(b=>b.textContent==='Bắt đầu giám sát')!.disabled).toBe(true);
  await act(async()=>Socket.current.receive({type:'completed',received_seconds:.001,backlog_seconds:0}));
  expect(buttons().find(b=>b.textContent==='Bắt đầu giám sát')!.disabled).toBe(false);
  await started();expect(permission).toHaveBeenCalledTimes(1);
  expect(buttons().find(b=>b.textContent==='Chọn tab livestream')!.disabled).toBe(true);
});
it('can cancel a pending tab selection and releases a stream that resolves afterwards',async()=>{
  let resolve!:(stream:unknown)=>void;
  const display=vi.fn().mockImplementation(()=>new Promise(done=>{resolve=done;}));
  Object.assign(navigator.mediaDevices,{getDisplayMedia:display});
  await click('Chọn tab livestream');
  expect(buttons().find(b=>b.textContent==='Bắt đầu giám sát')!.disabled).toBe(true);
  await click('Hủy chọn tab');
  expect(buttons().find(b=>b.textContent==='Bắt đầu giám sát')!.disabled).toBe(false);
  await act(async()=>resolve({getTracks:()=>[track]}));expect(track.stop).toHaveBeenCalled();
  expect(permission).not.toHaveBeenCalled();
});
it.each([48000,44100])('continuous resampling from %i produces 16000 samples/sec across worklet frames',rate=>{
  const source=new Float32Array(rate).fill(.5),r=new PcmResampler(rate),out:number[]=[];
  for(let i=0;i<source.length;i+=128)r.push(source.slice(i,i+128),(sample:number)=>out.push(sample));
  expect(out.length).toBe(16000);expect(out.every(n=>Math.abs(n-.5)<1e-6)).toBe(true);
});
it('shows sub-second timestamps instead of a false 00:00–00:00 interval',()=>{
  expect(`${timestamp(0)} – ${timestamp(.78)}`).toBe('00:00 – 00:00.780');
  expect(timestamp(59.9999)).toBe('01:00');
});
