import { liveSocketUrl, type Unit, type AsrQualityWarning } from './api';
import { microphoneError } from './useRecorder';
export type LiveSource = 'microphone_live' | 'browser_tab_live';
export interface LiveMessage {
  type: string; processing?: boolean; message?: string; session_id?: string; sequence?: number;
  pending_windows?: number; backlog_seconds?: number; warning_count?: number;
  received_seconds?: number;
  assembling_seconds?: number; queued_seconds?: number; processing_audio_seconds?: number;
  processed_audio_seconds?: number; silence_seconds?: number; debug_directory?: string|null;
  unprocessed_audio_seconds?: number;
  final_unit_latencies_seconds?: number[];
  temporary_transcript?: string; latency_seconds?: number; processing_time_seconds?: number;
  classification_units?: Unit[];
  asr_quality_warning?: AsrQualityWarning;
  asr_unresolved_windows?: number; asr_recovered_windows?: number;
}
export class LiveCapture {
  private stream?: MediaStream; private context?: AudioContext; private node?: AudioWorkletNode;
  private source?: MediaStreamAudioSourceNode; private gain?: GainNode; private socket?: WebSocket;
  private disposed=false; private stopping=false; private terminal=false;
  private flushRequested=false; private stopSent=false;
  private sequence=0; private ack=-1; private ackAt=Date.now();
  private sessionId?: string;
  private timer?: ReturnType<typeof setInterval>; private handshake?: ReturnType<typeof setTimeout>; private flush?: ReturnType<typeof setTimeout>;
  constructor(private event:(m:LiveMessage)=>void, private error:(m:string)=>void, private debugAudio=false,
    private inputSource:LiveSource='microphone_live') {}
  async start() {
    const tab=this.inputSource==='browser_tab_live';
    if (!(tab?navigator.mediaDevices?.getDisplayMedia:navigator.mediaDevices?.getUserMedia) || !window.AudioContext || !window.AudioWorkletNode) {
      this.fail('Trình duyệt không hỗ trợ giám sát PCM. Dùng Chrome/Edge trên localhost hoặc HTTPS.'); return;
    }
    try {
      // Display capture requires video permission, but only the audio track enters the PCM worklet/socket.
      const stream=tab?await navigator.mediaDevices.getDisplayMedia({audio:true,video:{displaySurface:'browser'}})
        :await navigator.mediaDevices.getUserMedia({audio:true});
      if (this.disposed||this.terminal) { stream.getTracks().forEach(t=>t.stop()); return; }
      this.stream=stream;
      if(tab) {
        const surface=stream.getVideoTracks()[0]?.getSettings?.().displaySurface;
        if(surface&&surface!=='browser') {
          this.fail('Nguồn vừa chọn là cửa sổ hoặc toàn màn hình, chưa phải tab livestream. Hãy mở http://127.0.0.1:5173/ trong cùng Chrome/Edge đang phát livestream (không mở trong trình duyệt của Codex), chọn Tab Chrome/Edge và bật Chia sẻ âm thanh tab; không chọn cửa sổ hoặc toàn màn hình.');return;
        }
        if(!stream.getAudioTracks().length) {
          this.fail('Tab chưa chia sẻ âm thanh. Chọn lại Tab Chrome/Edge và bật Chia sẻ âm thanh tab.');return;
        }
        // Browser Stop sharing ends the video track too. Drain the same final PCM frame as the End button.
        stream.getTracks().forEach(t=>{t.onended=()=>this.stop();});
      } else stream.getAudioTracks().forEach(t=>{t.onended=()=>this.fail('Microphone bị ngắt kết nối. Hãy mở phiên mới.');});
      const context=new AudioContext(); this.context=context;
      await context.resume(); await context.audioWorklet.addModule('/pcm-worklet.js');
      if (this.disposed||this.terminal) return;
      const socket=new WebSocket(liveSocketUrl()); this.socket=socket;
      this.handshake=setTimeout(()=>this.fail('Backend không phản hồi khi mở giám sát.'),15000);
      socket.onopen=()=>{
        if(this.disposed||this.terminal)return;
        const settings=stream.getAudioTracks()[0]?.getSettings?.();
        socket.send(JSON.stringify({type:'start',source:this.inputSource,format:'pcm_s16le',sample_rate:16000,channels:1,debug_audio:this.debugAudio,
          capture_info:{source_sample_rate:context.sampleRate,channel_count:settings?.channelCount,
            auto_gain_control:settings?.autoGainControl,noise_suppression:settings?.noiseSuppression,
            echo_cancellation:settings?.echoCancellation}}));
      };
      socket.onmessage=e=>{
        if(this.disposed||this.terminal)return;
        try {
          const m=JSON.parse(e.data) as LiveMessage;
          if (m.type==='started') {
            if(this.sessionId || !m.session_id)return;
            this.sessionId=m.session_id;clearTimeout(this.handshake);
            if(this.stopping) { this.sendStop();this.releaseAudio();return; }
            this.attach(context,stream);
          } else if(this.sessionId && m.session_id!==this.sessionId)return;
          if (m.type==='ack') { this.ack=m.sequence!; this.ackAt=Date.now(); }
          if (m.type==='draining') { this.stopping=true; this.releaseAudio(); }
          if (m.type==='completed'||m.type==='error') {
            this.terminal=true; this.releaseAudio(); if(m.type==='error')this.error(m.message||'Giám sát thất bại.');
          }
          this.event(m);
        } catch { this.fail('Backend trả dữ liệu giám sát không hợp lệ.'); }
      };
      socket.onerror=()=>this.fail('Không kết nối được backend để giám sát. Kiểm tra server rồi mở phiên mới.');
      socket.onclose=()=>{ if(!this.disposed&&!this.terminal)this.fail('Mất kết nối giám sát. Kết quả đã chốt nằm trong lịch sử; không tự nối lại.'); this.releaseAudio();clearTimeout(this.handshake); };
    } catch(e) {
      if(!this.disposed)this.fail(tab?this.tabError(e):microphoneError(e));
    }
  }
  private tabError(error:unknown) {
    const name=typeof error==='object'&&error!==null&&'name' in error?error.name:'';
    if(name==='NotAllowedError'||name==='SecurityError')return 'Chưa được phép chia sẻ tab hoặc bạn đã hủy lựa chọn. Chọn lại Tab Chrome/Edge và bật Chia sẻ âm thanh tab.';
    if(name==='NotFoundError')return 'Không tìm thấy tab có thể chia sẻ. Mở livestream trong Chrome/Edge rồi thử lại.';
    return 'Không thu được âm thanh tab. Kiểm tra tab đang phát tiếng, chọn Tab Chrome/Edge và bật Chia sẻ âm thanh tab.';
  }
  private attach(context:AudioContext,stream:MediaStream) {
    this.node=new AudioWorkletNode(context,'pcm-capture');
    this.node.onprocessorerror=()=>this.fail('Bộ thu âm bị gián đoạn. Phần chưa được xác nhận chưa được xử lý; hãy mở phiên mới.');
    this.node.port.onmessage=e=>{
      if(this.disposed||this.terminal||!this.node)return;
      if(e.data==='flushed') {
        if(!this.flushRequested)return;
        clearTimeout(this.flush);
        this.sendStop();
        this.releaseAudio();return;
      }
      if(this.socket?.readyState!==WebSocket.OPEN)return;
      if(this.socket.bufferedAmount>256000||this.sequence-this.ack>16) {
        this.fail('Kết nối quá chậm, đã dừng giám sát. Audio chưa được xác nhận chưa được xử lý; xem phần đã chốt trong lịch sử.');return;
      }
      const pcm=e.data as ArrayBuffer, packet=new Uint8Array(4+pcm.byteLength);
      new DataView(packet.buffer).setUint32(0,this.sequence++,true);packet.set(new Uint8Array(pcm),4);this.socket.send(packet.buffer);
    };
    this.source=context.createMediaStreamSource(stream);this.gain=context.createGain();this.gain.gain.value=0;
    this.source.connect(this.node);this.node.connect(this.gain);this.gain.connect(context.destination);
    this.timer=setInterval(()=>{ if(!this.stopping&&this.sequence-1>this.ack&&Date.now()-this.ackAt>10000)this.fail('Backend không xác nhận audio trong 10 giây. Đã dừng giám sát.'); },1000);
  }
  stop() {
    if(this.stopping||this.disposed||this.terminal)return;
    if(!this.socket) { this.fail('Đã hủy yêu cầu thu trước khi phiên sẵn sàng. Chưa gửi audio vào backend.');return; }
    this.stopping=true;this.event({type:'draining',session_id:this.sessionId});
    if(this.node) {
      this.flushRequested=true;
      this.flush=setTimeout(()=>this.fail('Không hoàn tất được chunk cuối. Phiên bị gián đoạn; không tự gửi lại.'),3000);
      this.node.port.postMessage('stop');
    }
    // If capture ended while opening the socket, finish the handshake then close an empty session.
    else if(this.sessionId) { this.sendStop();this.releaseAudio(); }
    else this.releaseAudio();
  }
  private sendStop() {
    if(this.stopSent||this.socket?.readyState!==WebSocket.OPEN)return;
    this.stopSent=true;this.socket.send(JSON.stringify({type:'stop',last_sequence:this.sequence-1}));
  }
  private releaseAudio() {
    clearInterval(this.timer);clearTimeout(this.flush);this.flushRequested=false;this.stream?.getTracks().forEach(t=>{t.onended=null;t.stop();});this.stream=undefined;
    this.source?.disconnect();this.node?.disconnect();this.gain?.disconnect();if(this.node)this.node.port.onmessage=null;this.node=undefined;
    if(this.context&&this.context.state!=='closed')void this.context.close();this.context=undefined;
  }
  private fail(message:string) { if(this.terminal||this.disposed)return;this.terminal=true;this.error(message);this.event({type:'failed'});this.dispose(); }
  dispose() {
    this.disposed=true;clearTimeout(this.handshake);clearTimeout(this.flush);this.releaseAudio();
    if(this.socket) { this.socket.onclose=null;this.socket.onerror=null;this.socket.onmessage=null;this.socket.onopen=null;this.socket.close(); }
  }
}
