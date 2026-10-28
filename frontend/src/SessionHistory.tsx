import { useEffect, useRef, useState } from 'react';
import { downloadSession, getSession, getSessions, labels, timestamp, type SessionDetail, type SessionPage } from './api';
import FeedbackEditor from './FeedbackEditor';
import AsrWarnings from './AsrWarnings';

const statuses = { processing: 'Đang xử lý', completed: 'Hoàn tất', failed: 'Thất bại' };
const hashSession = () => /^#session=([a-f0-9-]{36})$/.exec(window.location.hash)?.[1] || '';
export default function SessionHistory({ recentSessionId }: { recentSessionId?: string }) {
  const [open, setOpen] = useState(!!hashSession());
  const [selected, setSelected] = useState(hashSession());
  const [page, setPage] = useState(1);
  const [warningsOnly, setWarningsOnly] = useState(false);
  const [status, setStatus] = useState('');
  const [data, setData] = useState<SessionPage | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');
  const [exporting, setExporting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const exportLock = useRef(false);
  useEffect(() => {
    if (!open) return;
    let active = true; setLoading(true); setError(''); setData(null);
    getSessions(page, warningsOnly, status).then(value => { if (active) setData(value); }).catch(e => { if (active) setError(e.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [open, page, warningsOnly, status, refresh, recentSessionId]);
  useEffect(() => {
    if (!open || !selected) return;
    let active = true; setDetailLoading(true); setDetail(null); setDetailError('');
    getSession(selected).then(value => { if (active) setDetail(value); }).catch(e => { if (active) setDetailError(e.message); }).finally(() => { if (active) setDetailLoading(false); });
    return () => { active = false; };
  }, [open, selected, refresh]);
  function select(id: string) { setSelected(id); window.history.replaceState(null, '', '#session=' + id); }
  async function exportReport() {
    if (!detail || exportLock.current) return;
    exportLock.current = true; setExporting(true); setDetailError('');
    try { await downloadSession(detail.id); } catch (e) { setDetailError(e instanceof Error ? e.message : 'Không tải được báo cáo.'); }
    finally { exportLock.current = false; setExporting(false); }
  }
  return <section className="history-panel" aria-labelledby="history-heading"><h2 id="history-heading">Lịch sử phân tích</h2><p>Chỉ lưu văn bản, kết quả và phản hồi; không lưu âm thanh để nghe lại. Phản hồi không tự được đưa vào train.</p>
    <div className="history-actions"><button aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Thu gọn lịch sử' : 'Mở lịch sử'}</button>{recentSessionId && <button onClick={() => { setOpen(true); select(recentSessionId); }}>Mở phiên vừa phân tích</button>}</div>
    {open && <><div className="history-filters"><label>Nội dung<select value={warningsOnly ? 'warnings' : 'all'} onChange={e => { setWarningsOnly(e.target.value === 'warnings'); setPage(1); }}><option value="all">Tất cả phiên</option><option value="warnings">Phiên có cảnh báo gốc</option></select></label><label>Trạng thái<select value={status} onChange={e => { setStatus(e.target.value); setPage(1); }}><option value="">Tất cả trạng thái</option>{Object.entries(statuses).map(([key, name]) => <option key={key} value={key}>{name}</option>)}</select></label><button disabled={loading} onClick={() => setRefresh(v => v + 1)}>Tải lại lịch sử</button></div>
    {loading && <p role="status">Đang tải lịch sử…</p>}{error && <p className="error" role="alert">{error}</p>}
    {data && <><p>{data.total} phiên · Trang {page}</p>{data.items.length === 0 && <p className="empty">Chưa có phiên phù hợp.</p>}<ul className="session-list">{data.items.map(s => <li key={s.id}><button className="session-card" aria-pressed={selected === s.id} onClick={() => select(s.id)}><strong>{s.filename}</strong><span>{new Date(s.created_at).toLocaleString('vi-VN')} · {s.source === 'browser_tab_live' ? 'Giám sát tab livestream' : s.source === 'microphone_live' ? 'Giám sát microphone' : s.source === 'recording' ? 'Thu âm' : 'Upload'}</span><span>{statuses[s.status]} · {s.warning_count} cảnh báo / {s.segment_count} câu · {s.model_version}</span></button></li>)}</ul><div className="history-actions"><button disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>Trang trước</button><button disabled={page * data.page_size >= data.total || loading} onClick={() => setPage(p => p + 1)}>Trang sau</button></div></>}
    {detailLoading && <p role="status">Đang tải chi tiết phiên…</p>}{detailError && <p className="error" role="alert">{detailError}</p>}
    {detail && <div className="session-detail"><h3>Chi tiết phiên · {detail.filename}</h3><p>{statuses[detail.status]} · {new Date(detail.started_at).toLocaleString('vi-VN')}{detail.ended_at && ` → ${new Date(detail.ended_at).toLocaleString('vi-VN')}`} · {detail.processing_time_seconds?.toFixed(1) ?? '—'} giây</p><p>Model đã dùng: {detail.model_version}</p><small className="checkpoint">{detail.checkpoint}</small><button disabled={exporting} onClick={() => void exportReport()}>{exporting ? 'Đang tải…' : 'Tải báo cáo JSON'}</button>{detail.error && <p className="error">{detail.error}</p>}<details className="transcript"><summary>Transcript phiên đã lưu</summary><p>{detail.transcript || 'Chưa có transcript.'}</p></details>
      <AsrWarnings warnings={detail.asr_quality_warnings} unresolvedCount={detail.asr_unresolved_windows} recoveredCount={detail.asr_recovered_windows}/>
      {detail.classification_units.map(unit => <article className={'unit ' + unit.final_label} key={unit.segment_id}><div className="unit-top"><span className="badge">Dự đoán gốc: {labels[unit.final_label]}</span><span className="time">{timestamp(unit.start)} – {timestamp(unit.end)}{unit.timestamp_is_approximate && <small>Thời gian ước lượng</small>}</span></div><h4>{unit.text}</h4><p>{unit.explanation}</p><details><summary>Xác suất mô hình</summary><ul>{Object.entries(unit.phobert_result.probabilities).map(([label, probability]) => <li key={label}>{labels[label as keyof typeof labels]}: {(probability * 100).toFixed(1)}%</li>)}</ul><p>Đây không phải độ chính xác của hệ thống.</p></details><FeedbackEditor unit={unit}/></article>)}
    </div>}</>}
  </section>;
}
