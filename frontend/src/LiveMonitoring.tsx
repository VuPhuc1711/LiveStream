import { useEffect,useRef,useState } from 'react';
import { labels,timestamp,visibleUnits,type Unit,type AsrQualityWarning } from './api';
import { LiveCapture,type LiveMessage,type LiveSource } from './liveCapture';
import FeedbackEditor from './FeedbackEditor';
import AsrWarnings from './AsrWarnings';
export default function LiveMonitoring({disabled,onActive,onFinished,source='microphone_live'}:{disabled:boolean;onActive:(v:boolean)=>void;onFinished:(id:string)=>void;source?:LiveSource}) {
  const tab=source==='browser_tab_live',headingId=`live-heading-${source}`;
  const [phase,setPhase]=useState('idle'),[seconds,setSeconds]=useState(0),[error,setError]=useState('');
  const [session,setSession]=useState(''),[units,setUnits]=useState<Unit[]>([]),[temporary,setTemporary]=useState(''),[all,setAll]=useState(false);
  const [asrWarnings,setAsrWarnings]=useState<AsrQualityWarning[]>([]);
  const [stats,setStats]=useState<LiveMessage>({type:'idle'}),capture=useRef<LiveCapture|null>(null),active=useRef(false);
  const [debugAudio,setDebugAudio]=useState(false),generation=useRef(0);
  const callbacks=useRef({onActive,onFinished});callbacks.current={onActive,onFinished};
  const activePhase=['requesting','recording','draining'].includes(phase);
  useEffect(()=>{const exit=()=>{generation.current++;capture.current?.dispose();};window.addEventListener('pagehide',exit);return()=>{window.removeEventListener('pagehide',exit);exit();};},[]);
  useEffect(()=>{if(phase!=='recording')return;const start=Date.now();setSeconds(0);const timer=setInterval(()=>setSeconds(Math.floor((Date.now()-start)/1000)),1000);return()=>clearInterval(timer);},[phase]);
  function start() {
    if(active.current||disabled)return;
    active.current=true;callbacks.current.onActive(true);capture.current?.dispose();setError('');setSession('');setUnits([]);setTemporary('');setAll(false);setAsrWarnings([]);setStats({type:'idle'});setSeconds(0);setPhase('requesting');
    const current=++generation.current;
    capture.current=new LiveCapture(m=>{
      if(current!==generation.current)return;
      setStats(old=>({...old,...m}));if(m.session_id)setSession(m.session_id);
      const asrWarning=m.asr_quality_warning;
      if(asrWarning)setAsrWarnings(old=>old.some(w=>w.window_id===asrWarning.window_id)?old:[...old,asrWarning]);
      if(m.type==='started')setPhase('recording');
      if(m.type==='draining'){setPhase('draining');if(m.message&&m.message!=='Đang xử lý phần còn lại')setError(m.message);}
      if(m.type==='result'){setTemporary(m.temporary_transcript||'');setUnits(old=>[...old,...(m.classification_units||[]).filter(u=>!old.some(o=>o.unit_id===u.unit_id))]);}
      if(['completed','error','failed'].includes(m.type)){setPhase(m.type==='completed'?'completed':'failed');setDebugAudio(false);if(m.type==='completed')setError('');setTemporary('');active.current=false;callbacks.current.onActive(false);if(m.session_id)callbacks.current.onFinished(m.session_id);}
    },message=>{if(current===generation.current)setError(message);},debugAudio,source);void capture.current.start();
  }
  const warnings=units.filter(u=>u.final_label==='CANH_BAO').length;
  return <section id={`monitor-${source}`} className="live-panel" aria-labelledby={headingId}><p className="eyebrow">{tab?'TAB LIVESTREAM':'MICROPHONE'} · GẦN THỜI GIAN THỰC</p><h2 id={headingId}>{tab?'Giám sát tab livestream':'Giám sát microphone'}</h2><p>{tab?'Lấy âm thanh từ một tab Chrome/Edge đang phát livestream. Không thu microphone và không gửi hình ảnh lên backend.':'Nói vào microphone; kết quả xuất hiện dần.'} Chỉ một phiên phân tích hoạt động. Không lưu âm thanh lâu dài.</p>
    {tab&&<><p><strong>Mở ứng dụng này trong Chrome/Edge đang phát livestream.</strong> Nếu đang mở trong trình duyệt của Codex, sao chép <code>http://127.0.0.1:5173/</code> sang một tab mới của Chrome/Edge đó.</p><p>Trong hộp chia sẻ, chọn <strong>Tab Chrome / Microsoft Edge</strong>, chọn tab đang phát và bật <strong>Chia sẻ âm thanh của tab</strong>. Không chọn Cửa sổ hoặc Toàn màn hình. Giữ tab phát tiếng và trang này mở. Nút dừng chia sẻ của trình duyệt cũng sẽ kết thúc phiên sau khi xử lý phần còn lại.</p></>}
    <div className="mic-actions"><button className="primary" disabled={disabled||activePhase} onClick={start}>{tab?'Chọn tab livestream':'Bắt đầu giám sát'}</button><button disabled={phase!=='recording'&&phase!=='requesting'} onClick={()=>capture.current?.stop()}>{phase==='requesting'?(tab?'Hủy chọn tab':'Hủy bật microphone'):tab?'Kết thúc tab':'Kết thúc'}</button></div>
    <label><input type="checkbox" checked={debugAudio} disabled={disabled||activePhase} onChange={e=>setDebugAudio(e.target.checked)}/> Lưu audio debug cho phiên thử (mặc định tắt)</label>
    {debugAudio&&<p>Chỉ phiên bật tùy chọn này lưu PCM nhận được và các WAV đưa vào Whisper trên máy backend. Xóa sau khi đối chiếu; không dùng cho train.</p>}
    {stats.debug_directory&&<small>Audio debug: {stats.debug_directory} · received.wav là toàn bộ audio nhận; window_*.wav là từng đoạn ASR.</small>}
    <p role="status">{phase==='requesting'?(tab?'Đang chờ chọn tab, chia sẻ âm thanh và mở kết nối…':'Đang xin quyền microphone và mở kết nối…'):phase==='recording'?`● ${tab?'Đang nhận âm thanh tab':'Microphone đang bật'} · Kết nối đang mở · ${seconds} giây`:phase==='draining'?`Đang xử lý phần còn lại · ${tab?'Thu tab':'Microphone'} đã dừng`:phase==='completed'?'Phiên đã hoàn tất và lưu vào lịch sử':phase==='failed'?'Phiên đã dừng do lỗi':tab?'Chưa chia sẻ tab':'Microphone đang tắt'}</p>
    {session&&<small>Phiên: {session} · Backend đã xác nhận nhận {(stats.received_seconds??0).toFixed(1)} giây audio</small>}
    {session&&<div className="live-stats"><span>{phase==='completed'?'Đã xử lý hết audio nhận được':phase==='failed'?'Đã dừng · xem phần chưa xử lý bên dưới':stats.processing?'Đang nhận dạng / phân loại':phase==='draining'?'Đang chốt phần còn lại':'Chờ tiếng nói'}</span><span>Chờ xử lý: {stats.pending_windows??0} đoạn</span><span>Tồn đọng: {(stats.backlog_seconds??0).toFixed(1)} giây</span><span>Đang gom: {(stats.assembling_seconds??0).toFixed(1)}s · Trong hàng chờ: {(stats.queued_seconds??0).toFixed(1)}s · Đang xử lý: {(stats.processing_audio_seconds??0).toFixed(1)}s</span><span>Đã qua ASR: {(stats.processed_audio_seconds??0).toFixed(1)}s · Không có tiếng nói: {(stats.silence_seconds??0).toFixed(1)}s</span><span>{warnings} cảnh báo / {units.length} câu</span>{stats.latency_seconds!==undefined&&<span>Độ trễ câu đã chốt (ước lượng): {(stats.final_unit_latencies_seconds?.length ? Math.max(...stats.final_unit_latencies_seconds) : stats.latency_seconds).toFixed(1)} giây · Xử lý {stats.processing_time_seconds?.toFixed(1)} giây</span>}</div>}
    {error&&<p className="error" role="alert">{error}</p>}
    {phase==='failed'&&<p>{session?`Audio chưa xử lý: ${(stats.unprocessed_audio_seconds??stats.backlog_seconds??0).toFixed(1)} giây. Phiên không được báo hoàn tất.`:'Chưa tạo phiên phân tích; chưa gửi audio lên backend.'}</p>}
    <AsrWarnings warnings={asrWarnings} unresolvedCount={stats.asr_unresolved_windows} recoveredCount={stats.asr_recovered_windows}/>
    <small>VAD phát hiện tiếng nói, giữ đệm đầu/cuối 0,3s; chờ lặng 1,5s; cửa sổ tối đa 12s. Giữ câu cuối qua ranh giới cưỡng bức (tối đa 24s ngữ cảnh). ASR vẫn có thể nhận sai; audio debug giúp nghe đối chiếu.</small>
    {temporary&&<div className="live-temporary"><strong>Transcript tạm · chưa phân loại/chưa lưu</strong><p>{temporary}</p></div>}
    {session&&<><details className="transcript"><summary>Transcript đã chốt</summary><p>{units.map(u=>u.text).join(' ')||'Chưa có đoạn đã chốt.'}</p></details><label><input type="checkbox" checked={all} onChange={e=>setAll(e.target.checked)}/> Hiện tất cả câu giám sát</label>
      {visibleUnits(units,all).map(u=><article className={'unit '+u.final_label} key={u.unit_id}><div className="unit-top"><span className="badge">{labels[u.final_label]}</span><span className="time">{timestamp(u.start)} – {timestamp(u.end)}<small>Thời gian ước lượng</small></span></div><h4>{u.text}</h4><p>{u.explanation}</p><details><summary>Xác suất mô hình</summary><ul>{Object.entries(u.phobert_result.probabilities).map(([l,p])=><li key={l}>{labels[l as keyof typeof labels]}: {(p*100).toFixed(1)}%</li>)}</ul><p>Không phải độ chính xác hệ thống.</p></details>{!activePhase&&<FeedbackEditor unit={u}/>}</article>)}
      {!warnings&&<p>Chưa có cảnh báo trong các câu đã chốt.</p>}</>}
  </section>;
}
