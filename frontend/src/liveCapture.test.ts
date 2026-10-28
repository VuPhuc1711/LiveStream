// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { LiveCapture, type LiveMessage } from './liveCapture';

class Socket {
  static OPEN=1; static instances:Socket[]=[];
  readyState=1;bufferedAmount=0;sent:unknown[]=[];
  onopen:(()=>void)|null=null;onmessage:((e:{data:string})=>void)|null=null;
  onclose:(()=>void)|null=null;onerror:(()=>void)|null=null;
  constructor(public url:string){Socket.instances.push(this);}
  send(value:unknown){this.sent.push(value);}close(){this.readyState=3;this.onclose?.();}
  receive(value:object){this.onmessage?.({data:JSON.stringify({session_id:'tab-session',...value})});}
}
class Context {
  static instances:Context[]=[];static resumeValue:Promise<void>|undefined;state='running';sampleRate=48000;destination={};
  audioWorklet={addModule:vi.fn().mockResolvedValue(undefined)};
  resume=vi.fn(()=>Context.resumeValue??Promise.resolve());close=vi.fn(async()=>{this.state='closed';});
  createMediaStreamSource=vi.fn(()=>({connect:vi.fn(),disconnect:vi.fn()}));
  createGain=vi.fn(()=>({gain:{value:1},connect:vi.fn(),disconnect:vi.fn()}));
  constructor(){Context.instances.push(this);}
}
class Worklet {
  static instances:Worklet[]=[];connect=vi.fn();disconnect=vi.fn();
  port={onmessage:null as ((e:{data:unknown})=>void)|null,postMessage:vi.fn()};
  constructor(){Worklet.instances.push(this);}
  frame(value:ArrayBuffer){this.port.onmessage?.({data:value});}
  flushed(){this.port.onmessage?.({data:'flushed'});}
}
function track(kind:'audio'|'video',surface='browser') {
  return {kind,stop:vi.fn(),onended:null as (()=>void)|null,getSettings:()=>kind==='video'?{displaySurface:surface}:{channelCount:2}};
}
let audio:ReturnType<typeof track>,video:ReturnType<typeof track>,stream:MediaStream;
let display:ReturnType<typeof vi.fn>,microphone:ReturnType<typeof vi.fn>;
let event:ReturnType<typeof vi.fn<(m:LiveMessage)=>void>>,error:ReturnType<typeof vi.fn<(m:string)=>void>>;
let capture:LiveCapture;
const socket=()=>Socket.instances.at(-1)!;
const worklet=()=>Worklet.instances.at(-1)!;
beforeEach(()=>{
  vi.useFakeTimers();Socket.instances=[];Context.instances=[];Context.resumeValue=undefined;Worklet.instances=[];
  audio=track('audio');video=track('video');
  stream={getTracks:()=>[audio,video],getAudioTracks:()=>[audio],getVideoTracks:()=>[video]} as unknown as MediaStream;
  display=vi.fn().mockResolvedValue(stream);microphone=vi.fn().mockResolvedValue(stream);event=vi.fn();error=vi.fn();
  vi.stubGlobal('navigator',{mediaDevices:{getDisplayMedia:display,getUserMedia:microphone}});
  vi.stubGlobal('AudioContext',Context);vi.stubGlobal('AudioWorkletNode',Worklet);vi.stubGlobal('WebSocket',Socket);
  capture=new LiveCapture(event,error,false,'browser_tab_live');
});
afterEach(()=>{capture.dispose();vi.useRealTimers();vi.unstubAllGlobals();});
async function started(){await capture.start();socket().onopen?.();socket().receive({type:'started'});}

it('requests explicit tab sharing, sends only PCM audio with a distinct source and never opens microphone',async()=>{
  await started();
  expect(display).toHaveBeenCalledWith({audio:true,video:{displaySurface:'browser'}});expect(microphone).not.toHaveBeenCalled();
  expect(JSON.parse(socket().sent[0] as string)).toMatchObject({source:'browser_tab_live',format:'pcm_s16le',sample_rate:16000,channels:1});
  expect(Context.instances[0].createMediaStreamSource).toHaveBeenCalledWith(stream);
  const samples=new Int16Array([10,-20,30]);worklet().frame(samples.buffer);
  expect(socket().sent).toHaveLength(2);
  const packet=socket().sent[1] as ArrayBuffer;
  expect(new DataView(packet).getUint32(0,true)).toBe(0);
  expect(new Int16Array(packet.slice(4))).toEqual(samples);
  expect(Context.instances[0].createGain.mock.results[0].value.gain.value).toBe(0);
});

it('browser Stop sharing flushes the last partial PCM frame before one stop and waits for server completion',async()=>{
  await started();const node=worklet();video.onended?.();audio.onended?.();capture.stop();
  expect(node.port.postMessage).toHaveBeenCalledExactlyOnceWith('stop');
  expect(audio.stop).not.toHaveBeenCalled();expect(socket().sent).toHaveLength(1);
  node.frame(new Int16Array([123,456]).buffer);node.flushed();
  expect(new DataView(socket().sent[1] as ArrayBuffer).getUint32(0,true)).toBe(0);
  expect(JSON.parse(socket().sent[2] as string)).toEqual({type:'stop',last_sequence:0});
  expect(audio.stop).toHaveBeenCalledOnce();expect(video.stop).toHaveBeenCalledOnce();
  expect(event).not.toHaveBeenCalledWith(expect.objectContaining({type:'completed'}));
  socket().receive({type:'completed',backlog_seconds:0});
  expect(event).toHaveBeenCalledWith(expect.objectContaining({type:'completed'}));expect(error).not.toHaveBeenCalled();
});

it('ending sharing before the handshake closes an empty server session without starting a worklet',async()=>{
  await capture.start();video.onended?.();socket().onopen?.();socket().receive({type:'started'});
  expect(Worklet.instances).toHaveLength(0);
  expect(JSON.parse(socket().sent[1] as string)).toEqual({type:'stop',last_sequence:-1});
  expect(audio.stop).toHaveBeenCalledOnce();socket().receive({type:'completed'});expect(error).not.toHaveBeenCalled();
});

it.each(['window','monitor'])('rejects %s capture and releases both tracks before connecting',async surface=>{
  video=track('video',surface);await capture.start();
  expect(error).toHaveBeenCalledWith(expect.stringContaining('không chọn cửa sổ hoặc toàn màn hình'));
  expect(audio.stop).toHaveBeenCalledOnce();expect(video.stop).toHaveBeenCalledOnce();expect(Socket.instances).toHaveLength(0);
});

it('rejects a tab without shared audio instead of silently capturing microphone',async()=>{
  display.mockResolvedValueOnce({getTracks:()=>[video],getAudioTracks:()=>[],getVideoTracks:()=>[video]});await capture.start();
  expect(error).toHaveBeenCalledWith(expect.stringContaining('Tab chưa chia sẻ âm thanh'));
  expect(video.stop).toHaveBeenCalledOnce();expect(microphone).not.toHaveBeenCalled();expect(Socket.instances).toHaveLength(0);
});

it('cancelled selection gives a tab-specific message and creates no connection',async()=>{
  display.mockRejectedValueOnce(new DOMException('','NotAllowedError'));await capture.start();
  expect(error).toHaveBeenCalledWith(expect.stringContaining('bạn đã hủy lựa chọn'));expect(Socket.instances).toHaveLength(0);
});

it('cancel during a pending permission request releases the eventually granted stream without connecting',async()=>{
  let resolve!:(stream:MediaStream)=>void;display.mockReturnValueOnce(new Promise<MediaStream>(r=>{resolve=r;}));
  const starting=capture.start();capture.stop();resolve(stream);await starting;
  expect(audio.stop).toHaveBeenCalledOnce();expect(video.stop).toHaveBeenCalledOnce();expect(Socket.instances).toHaveLength(0);
  expect(event).toHaveBeenCalledWith({type:'failed'});
});

it.each(['stop','dispose'] as const)('%s while AudioContext.resume is pending promptly releases capture and never opens a socket',async action=>{
  let resolve!:()=>void;Context.resumeValue=new Promise<void>(r=>{resolve=r;});
  const starting=capture.start();await Promise.resolve();expect(Context.instances).toHaveLength(1);
  capture[action]();expect(audio.stop).toHaveBeenCalledOnce();expect(video.stop).toHaveBeenCalledOnce();
  expect(Context.instances[0].close).toHaveBeenCalledOnce();resolve();await starting;
  expect(Socket.instances).toHaveLength(0);expect(Worklet.instances).toHaveLength(0);
  if(action==='dispose')expect(error).not.toHaveBeenCalled();
});

it('late socket/worklet results after disposal cannot send audio or appear in the next session',async()=>{
  await started();const oldSocket=socket(),late=oldSocket.onmessage,latePcm=worklet().port.onmessage;
  capture.dispose();event.mockClear();
  late?.({data:JSON.stringify({type:'result',session_id:'tab-session',temporary_transcript:'late'})});
  latePcm?.({data:new ArrayBuffer(32)});latePcm?.({data:'flushed'});
  expect(event).not.toHaveBeenCalled();expect(oldSocket.sent).toHaveLength(1);
  capture=new LiveCapture(event,error,false,'browser_tab_live');await started();
  socket().receive({type:'result',session_id:'old-session',temporary_transcript:'other'});
  expect(event).not.toHaveBeenCalledWith(expect.objectContaining({temporary_transcript:'other'}));
});

it('preserves the default microphone input and cleans up on capture failure',async()=>{
  capture=new LiveCapture(event,error);await capture.start();socket().onopen?.();socket().receive({type:'started'});
  expect(microphone).toHaveBeenCalledWith({audio:true});expect(display).not.toHaveBeenCalled();
  expect(JSON.parse(socket().sent[0] as string).source).toBe('microphone_live');
  audio.onended?.();expect(error).toHaveBeenCalledWith(expect.stringContaining('Microphone bị ngắt'));expect(video.stop).toHaveBeenCalledOnce();
});
