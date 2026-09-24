import { useEffect, useRef, useState } from 'react';
import { analyze, getHealth, labels, overviewExplanation, timestamp, validateFile, visibleUnits, type Health, type Result } from './api';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [checking, setChecking] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const sending = useRef(false);
  const [elapsed, setElapsed] = useState(0);
  const [all, setAll] = useState(false);
  const [dragging, setDragging] = useState(false);
  async function check() { setChecking(true); try { setHealth(await getHealth()); } catch { setHealth(null); } finally { setChecking(false); } }
  useEffect(() => { void check(); }, []);
  useEffect(() => { if (!busy) return; const start = Date.now(); setElapsed(0); const timer = setInterval(() => setElapsed(Math.floor((Date.now()-start)/1000)), 1000); return () => clearInterval(timer); }, [busy]);
  function choose(files: FileList | null) {
    if (sending.current) return;
    setResult(null); setAll(false); setFile(null); setError('');
    if (!files?.length) return;
    if (files.length !== 1) { setError('Vui lòng chọn một file mỗi lần.'); return; }
    const next = files[0], problem = validateFile(next);
    if (problem) setError(problem); else setFile(next);
  }
  async function submit() {
    if (!file || sending.current) return;
    sending.current = true; setBusy(true); setError(''); setResult(null);
    try { setResult(await analyze(file)); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không thể xử lý file.'); void check(); }
    finally { sending.current = false; setBusy(false); }
  }
  const units = result ? visibleUnits(result.classification_units, all) : [];
  return <div className="shell">
    <header><a className="brand" href="#"><span className="brand-icon">◉</span> LIVEGUARD <span className="tag">THỬ NGHIỆM</span></a><span className="header-note">Hỗ trợ kiểm duyệt nội dung</span></header>
    <main>
      <div className="intro"><p className="eyebrow">PHÂN TÍCH ÂM THANH</p><h1>Giám sát nội dung livestream</h1><p>Từ lời nói đến nội dung cần chú ý. Tải bản ghi để bắt đầu kiểm tra.</p></div>
      <div className="connection" role="status"><span className={'dot ' + (health?.ready ? 'online' : '')}/><span>{checking ? 'Đang kiểm tra kết nối…' : health?.ready ? 'Backend đã sẵn sàng' : health ? 'Model chưa sẵn sàng' : 'Mất kết nối backend'}</span><button className="link-button" onClick={check} disabled={checking}>Thử lại</button></div>
      <section className="upload-panel" aria-labelledby="upload-heading"><div className="section-heading"><span className="step">01</span><div><h2 id="upload-heading">Chọn bản ghi</h2><p>Một file âm thanh hoặc video, tối đa 50 MiB.</p></div></div>
        <div className={'dropzone ' + (dragging ? 'dragging' : '')} onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files); }}>
          <span className="upload-icon" aria-hidden="true">↑</span><strong>Kéo thả file vào đây</strong><span>hoặc chọn file trên thiết bị</span><label className={'file-button ' + (busy ? 'disabled' : '')}>Chọn file<input aria-label="Chọn file âm thanh hoặc video" type="file" accept=".mp3,.m4a,.wav,.mp4" disabled={busy} onChange={e => { choose(e.target.files); e.target.value = ''; }}/></label><small>MP3 · M4A · WAV · MP4</small>
        </div>
        {file && <div className="file-info"><span aria-hidden="true">♫</span><div><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toLocaleString('vi-VN', { maximumFractionDigits: 2 })} MiB</small></div><span className="tag">ĐÃ CHỌN</span></div>}
        <div className="actions"><p>File chỉ được dùng để xử lý và sẽ được dọn sau khi hoàn tất.</p><button className="primary" disabled={!file || busy || !health?.ready} onClick={submit}>{busy ? 'Đang phân tích…' : 'Phân tích →'}</button></div>
        {busy && <div className="waiting" role="status"><span className="spinner"/>Đang nhận dạng và phân loại nội dung · Đã chờ {elapsed} giây<p>Thời gian phụ thuộc độ dài bản ghi. Vui lòng giữ trang này mở.</p></div>}
        {error && <div className="error" role="alert">{error}</div>}
      </section>
      {result ? <section className="results" aria-labelledby="result-heading"><div className="section-heading"><span className="step">02</span><div><h2 id="result-heading">Kết quả phân tích</h2><p>{result.filename}</p></div><span className="completed">Đã hoàn tất</span></div>
        <div className="metrics"><div><span>Thời gian xử lý</span><strong>{result.processing_time_seconds.toFixed(1)}<small> giây</small></strong></div><div><span>Đơn vị phân loại</span><strong>{result.summary.classification_unit_count}</strong></div><div className="red"><span>Cảnh báo</span><strong>{result.summary.final_label_counts.CANH_BAO || 0}</strong></div><div className="yellow"><span>Cần xác minh</span><strong>{result.summary.final_label_counts.CAN_XAC_MINH || 0}</strong></div></div>
        <details className="transcript"><summary>Transcript đầy đủ</summary><p>{result.transcript || 'Không nhận dạng được lời nói.'}</p></details>
        <div className="list-heading"><h3>Nội dung cần chú ý <span>{units.length}</span></h3><label><input type="checkbox" checked={all} onChange={e => setAll(e.target.checked)}/> Hiện tất cả câu</label></div>
        {units.length === 0 && <div className="empty"><strong>{result.classification_units.length ? 'Không có cảnh báo hoặc nội dung cần xác minh.' : 'Không nhận dạng được câu nào.'}</strong><p>{result.classification_units.length ? 'Bật “Hiện tất cả câu” để xem các câu không cảnh báo.' : 'Kiểm tra bản ghi có lời nói rõ ràng rồi chọn lại file.'}</p></div>}
        <div className="unit-list">{units.map(unit => <article className={'unit ' + unit.final_label} key={unit.unit_id}><div className="unit-top"><span className="badge">{labels[unit.final_label]}</span><span className="time">{timestamp(unit.start)} – {timestamp(unit.end)}{unit.timestamp_is_approximate && <small>Thời gian ước lượng</small>}</span></div><h4>{unit.text}</h4><p className="explanation">{overviewExplanation(unit.explanation)}</p><details><summary>Chi tiết PhoBERT & Rules</summary><p>{unit.explanation}</p><p>PhoBERT: {labels[unit.phobert_result.label]} · Confidence {(unit.phobert_result.confidence*100).toFixed(1)}%</p><p>Xác suất PhoBERT của nhãn cuối: {(unit.final_confidence*100).toFixed(1)}%. Đây không phải độ chính xác của hệ thống.</p><ul>{Object.entries(unit.phobert_result.probabilities).map(([label, probability]) => <li key={label}>{labels[label as keyof typeof labels]}: {(probability*100).toFixed(1)}%</li>)}</ul><strong>Rules</strong><pre>{JSON.stringify(unit.rule_result, null, 2)}</pre></details></article>)}</div>
      </section> : !busy && <div className="start-note"><span>02</span><p>Kết quả sẽ xuất hiện ở đây sau khi phân tích.<small>Cảnh báo và nội dung cần xác minh được sắp theo thời gian.</small></p></div>}
    </main><footer>LIVEGUARD <span>Thông tin hỗ trợ kiểm duyệt · Quyết định cuối thuộc người kiểm duyệt</span></footer>
  </div>;
}

