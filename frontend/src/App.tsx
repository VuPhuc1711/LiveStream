import { useRecorder } from './useRecorder';
import SessionHistory from './SessionHistory';
import FeedbackEditor from './FeedbackEditor';
import SupplementaryData from './SupplementaryData';
import LiveMonitoring from './LiveMonitoring';
import AsrWarnings from './AsrWarnings';
import type { LiveSource } from './liveCapture';
import { useEffect, useRef, useState } from 'react';
import { analyze, getHealth, labels, overviewExplanation, timestamp, validateFile, visibleUnits, type Health, type Result } from './api';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [checking, setChecking] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [liveSource, setLiveSource] = useState<LiveSource | null>(null);
  const liveActive = liveSource !== null;
  const [liveSession, setLiveSession] = useState<string>();
  const sending = useRef(false);
  const [elapsed, setElapsed] = useState(0);
  const [all, setAll] = useState(false);
  const [dragging, setDragging] = useState(false);
  const mic = useRecorder(next => { setFile(next); setResult(null); setAll(false); setError(''); }, setError);
  const locked = busy || liveActive || mic.phase !== 'idle';
  async function check() { setChecking(true); try { setHealth(await getHealth()); } catch { setHealth(null); } finally { setChecking(false); } }
  useEffect(() => { void check(); }, []);
  useEffect(() => { if (!busy) return; const start = Date.now(); setElapsed(0); const timer = setInterval(() => setElapsed(Math.floor((Date.now()-start)/1000)), 1000); return () => clearInterval(timer); }, [busy]);
  function choose(files: FileList | null) {
    if (sending.current || mic.active.current || liveActive) return;
    mic.clear(); setResult(null); setAll(false); setFile(null); setError('');
    if (!files?.length) return;
    if (files.length !== 1) { setError('Vui lòng chọn một file mỗi lần.'); return; }
    const next = files[0], problem = validateFile(next);
    if (problem) setError(problem); else setFile(next);
  }
  async function submit() {
    if (!file || sending.current || mic.active.current || liveActive) return;
    sending.current = true; setBusy(true); setError(''); setResult(null);
    try { setResult(await analyze(file, mic.url ? 'recording' : 'upload')); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không thể xử lý file.'); void check(); }
    finally { sending.current = false; setBusy(false); }
  }
  const binary = (result?.classification_mode || health?.classification_mode) === 'binary';
  const units = result ? visibleUnits(result.classification_units, all) : [];
  return <div className="shell">
    <header><a className="brand" href="#"><span className="brand-icon">◉</span> LIVEGUARD <span className="tag">THỬ NGHIỆM</span></a><span className="header-note">Hỗ trợ kiểm duyệt nội dung</span></header>
    <main>
      <div className="intro"><p className="eyebrow">PHÂN TÍCH ÂM THANH</p><h1>Giám sát nội dung livestream</h1><p>Chọn giám sát microphone, tab livestream hoặc phân tích bản ghi.</p><nav className="monitor-links" aria-label="Chọn nguồn giám sát"><a href="#monitor-microphone_live">Microphone</a><a href="#monitor-browser_tab_live">Tab livestream</a><a href="#upload-heading">Bản ghi</a></nav></div>
      <div className="connection" role="status"><span className={'dot ' + (health?.ready ? 'online' : '')}/><span>{checking ? 'Đang kiểm tra kết nối…' : health?.ready ? 'Backend đã sẵn sàng' : health ? 'Model chưa sẵn sàng' : 'Mất kết nối backend'}</span><button className="link-button" onClick={check} disabled={checking}>Thử lại</button></div>
      <div className="model-status" role="status">
        <strong>Bản thử nghiệm hỗ trợ kiểm duyệt</strong>
        <span>{health?.ready && health.model_version ? `Model backend đang nạp: PhoBERT ${health.model_version}` : 'Chưa xác nhận được phiên bản model từ backend.'}</span>
        {health?.ready && health.classifier === 'phobert_only' && <small>Chế độ phân loại: chỉ PhoBERT</small>}
        {health?.ready && health.checkpoint && <small>Checkpoint: {health.checkpoint}</small>}
        <small>{binary ? 'Hai nhãn: không cảnh báo không có nghĩa đã xác thực; cảnh báo là đề nghị kiểm duyệt.' : 'Demo cục bộ; kết quả cần người kiểm duyệt xem lại.'}</small>
      </div>
      <section className="upload-panel" aria-labelledby="upload-heading"><div className="section-heading"><span className="step">01</span><div><h2 id="upload-heading">Chọn bản ghi</h2><p>Một file âm thanh hoặc video, tối đa 50 MiB.</p></div></div>
        <div className={'dropzone ' + (dragging ? 'dragging' : '')} onDragOver={e => { e.preventDefault(); if (!locked) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files); }}>
          <span className="upload-icon" aria-hidden="true">↑</span><strong>Kéo thả file vào đây</strong><span>hoặc chọn file trên thiết bị</span><label className={'file-button ' + (locked ? 'disabled' : '')}>Chọn file<input aria-label="Chọn file âm thanh hoặc video" type="file" accept=".mp3,.m4a,.wav,.mp4,.webm" disabled={locked} onChange={e => { choose(e.target.files); e.target.value = ''; }}/></label><small>MP3 · M4A · WAV · MP4 · WebM</small>
        </div>
        <div className="microphone-panel">
          <h3>Hoặc thu âm trực tiếp</h3>
          <p>Chỉ bật microphone khi bạn bắt đầu. Bản thu không được lưu vào dataset.</p>
          <div className="mic-actions">
            <button className="file-button" disabled={locked} onClick={() => void mic.start()}>{mic.url ? 'Thu lại' : 'Bắt đầu thu âm'}</button>
            <button className="file-button" disabled={mic.phase !== 'recording'} onClick={mic.stop}>Dừng thu âm</button>
            {mic.url && <button className="link-button" disabled={locked} onClick={() => { mic.clear(); setFile(null); setResult(null); setError(''); }}>Xóa bản thu</button>}
          </div>
          <div role="status">{mic.phase === 'requesting' ? 'Đang chờ quyền microphone…' : mic.phase === 'recording' ? `● Đang thu âm · ${mic.seconds} giây` : mic.phase === 'stopping' ? 'Đang hoàn tất bản thu…' : ''}</div>
          {mic.url && <div className="recording-preview"><label htmlFor="recording-playback">Nghe lại bản thu</label><audio id="recording-playback" aria-label="Nghe lại bản thu" controls src={mic.url}/><small>Định dạng: {file?.type} · {mic.seconds} giây</small></div>}
        </div>
        {file && <div className="file-info"><span aria-hidden="true">♫</span><div><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toLocaleString('vi-VN', { maximumFractionDigits: 2 })} MiB</small></div><span className="tag">ĐÃ CHỌN</span></div>}
        <div className="actions"><p>File chỉ được dùng để xử lý và sẽ được dọn sau khi hoàn tất.</p><button className="primary" disabled={!file || locked || !health?.ready} onClick={submit}>{busy ? 'Đang phân tích…' : mic.url ? 'Phân tích bản thu' : 'Phân tích →'}</button></div>
        {busy && <div className="waiting" role="status"><span className="spinner"/>Đang nhận dạng và phân loại nội dung · Đã chờ {elapsed} giây<p>Thời gian phụ thuộc độ dài bản ghi. Vui lòng giữ trang này mở.</p></div>}
        {error && <div className="error" role="alert">{error}</div>}
      </section>
      {result ? <section className="results" aria-labelledby="result-heading"><div className="section-heading"><span className="step">02</span><div><h2 id="result-heading">Kết quả phân tích</h2><p>{result.filename}</p></div><span className="completed">Đã hoàn tất</span></div>
        <div className="metrics"><div><span>Thời gian xử lý</span><strong>{result.processing_time_seconds.toFixed(1)}<small> giây</small></strong></div><div><span>Đơn vị phân loại</span><strong>{result.summary.classification_unit_count}</strong></div><div className="red"><span>Cảnh báo</span><strong>{result.summary.final_label_counts.CANH_BAO || 0}</strong></div>{!binary && <div className="yellow"><span>Cần xác minh</span><strong>{result.summary.final_label_counts.CAN_XAC_MINH || 0}</strong></div>}</div>
        <details className="transcript"><summary>Transcript đầy đủ</summary><p>{result.transcript || 'Không nhận dạng được lời nói.'}</p></details>
        <AsrWarnings warnings={result.asr_quality_warnings} unresolvedCount={result.asr_unresolved_windows} recoveredCount={result.asr_recovered_windows}/>
        <div className="list-heading"><h3>Nội dung cần chú ý <span>{units.length}</span></h3><label><input type="checkbox" checked={all} onChange={e => setAll(e.target.checked)}/> Hiện tất cả câu</label></div>
        {units.length === 0 && <div className="empty"><strong>{result.classification_units.length ? (binary ? (result.asr_quality_warnings?.length || result.asr_unresolved_windows ? 'Không có cảnh báo trong các câu đã phân loại.' : 'Không có cảnh báo.') : 'Không có cảnh báo hoặc nội dung cần xác minh.') : 'Không nhận dạng được câu nào.'}</strong><p>{result.classification_units.length ? 'Bật “Hiện tất cả câu” để xem các câu không cảnh báo.' : result.asr_quality_warnings?.length || result.asr_unresolved_windows ? 'Xem các đoạn chưa nhận dạng đáng tin cậy ở trên. Chưa có kết quả phân loại cho các đoạn đó.' : 'Kiểm tra bản ghi có lời nói rồi chọn lại file.'}</p></div>}
        <div className="unit-list">{units.map(unit => <article className={'unit ' + unit.final_label} key={unit.unit_id}><div className="unit-top"><span className="badge">{labels[unit.final_label]}</span><span className="time">{timestamp(unit.start)} – {timestamp(unit.end)}{unit.timestamp_is_approximate && <small>Thời gian ước lượng</small>}</span></div><h4>{unit.text}</h4><p className="explanation">{overviewExplanation(unit.explanation)}</p><details><summary>Chi tiết PhoBERT</summary><p>{unit.explanation}</p><p>PhoBERT: {labels[unit.phobert_result.label]} · Xác suất mô hình {(unit.phobert_result.confidence*100).toFixed(1)}%</p><p>Xác suất PhoBERT của nhãn dự đoán: {(unit.final_confidence*100).toFixed(1)}%. Đây không phải độ chính xác của hệ thống.</p><ul>{Object.entries(unit.phobert_result.probabilities).map(([label, probability]) => <li key={label}>{labels[label as keyof typeof labels]}: {(probability*100).toFixed(1)}%</li>)}</ul></details><FeedbackEditor unit={unit}/></article>)}</div>
      </section> : !busy && <div className="start-note"><span>02</span><p>Kết quả sẽ xuất hiện ở đây sau khi phân tích.<small>{binary ? 'Các cảnh báo được sắp theo thời gian.' : 'Cảnh báo và nội dung cần xác minh được sắp theo thời gian.'}</small></p></div>}
      <LiveMonitoring disabled={busy || mic.phase !== 'idle' || !health?.ready || liveSource==='browser_tab_live'} onActive={v=>setLiveSource(old=>v?'microphone_live':old==='microphone_live'?null:old)} onFinished={setLiveSession}/>
      <LiveMonitoring source="browser_tab_live" disabled={busy || mic.phase !== 'idle' || !health?.ready || liveSource==='microphone_live'} onActive={v=>setLiveSource(old=>v?'browser_tab_live':old==='browser_tab_live'?null:old)} onFinished={setLiveSession}/>
      <SessionHistory recentSessionId={liveSession || result?.session_id}/>
      <SupplementaryData/>
    </main><footer>LIVEGUARD <span>Thông tin hỗ trợ kiểm duyệt · Quyết định cuối thuộc người kiểm duyệt</span></footer>
  </div>;
}
