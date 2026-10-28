import { useEffect, useRef, useState } from 'react';
import { labels, saveFeedback, sampleFromFeedback, type ReviewSample, type BinaryLabel, type Feedback, type Unit } from './api';
import ReviewEditor from './ReviewEditor';

export default function FeedbackEditor({ unit }: { unit: Unit }) {
  const [entries, setEntries] = useState<Feedback[]>(unit.feedback || []);
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState<BinaryLabel>(unit.final_label as BinaryLabel);
  const [note, setNote] = useState('');
  const [text, setText] = useState(unit.text);
  const [situation, setSituation] = useState('');
  const [sample, setSample] = useState<ReviewSample | null>(null);
  const [busy, setBusy] = useState(false);
  const sending = useRef(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  useEffect(() => { setEntries(unit.feedback || []); }, [unit.feedback]);
  async function save(value: BinaryLabel) {
    if (!unit.segment_id || sending.current) return;
    sending.current = true; setBusy(true); setError(''); setMessage('');
    try { const entry = await saveFeedback(unit.segment_id, value, note, text, situation); setEntries(old => old.some(f=>f.id===entry.id)?old:[...old, entry]); setEditing(false); setSample(null); setMessage('Đã lưu phản hồi. Dự đoán gốc được giữ nguyên. Chưa đưa vào dữ liệu đã duyệt.'); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không lưu được phản hồi.'); }
    finally { sending.current = false; setBusy(false); }
  }
  async function prepare(feedback: Feedback) {
    if(sending.current)return;
    sending.current=true;setBusy(true);setError('');
    try{setSample(await sampleFromFeedback(feedback.id));}
    catch(e){setError(e instanceof Error?e.message:'Không mở được mẫu bổ sung.');}
    finally{sending.current=false;setBusy(false);}
  }
  if (!unit.segment_id) return null;
  return <div className="feedback-panel">
    <p><strong>Dự đoán gốc:</strong> {labels[unit.final_label]}</p>
    {entries.length > 0 && <div className="feedback-log"><strong>Phản hồi người dùng</strong><ul>{entries.map(f => <li key={f.id}>{labels[f.label]} · {new Date(f.created_at).toLocaleString('vi-VN')}{f.corrected_text&&<p>Văn bản xác nhận: {f.corrected_text}</p>}{f.note && <p>{f.note}</p>}<button disabled={busy} onClick={()=>void prepare(f)}>Chuẩn bị duyệt đưa vào dữ liệu</button></li>)}</ul></div>}
    <div className="history-actions"><button disabled={busy} onClick={() => void save(unit.final_label as BinaryLabel)}>Dự đoán đúng</button><button disabled={busy} onClick={() => { setEditing(!editing); setMessage(''); }}>Sửa nhãn</button></div>
    {editing && <form onSubmit={e => { e.preventDefault(); void save(label); }}><label>Văn bản xác nhận / sửa<textarea required value={text} onChange={e=>setText(e.target.value)} maxLength={4000} disabled={busy}/></label><label>Nhãn xác nhận<select value={label} onChange={e => setLabel(e.target.value as BinaryLabel)} disabled={busy}><option value="KHONG_CANH_BAO">Không cảnh báo</option><option value="CANH_BAO">Cảnh báo</option></select></label><label>Tình huống<input value={situation} onChange={e=>setSituation(e.target.value)} maxLength={200} disabled={busy}/></label><label>Ghi chú / lý do sửa<textarea value={note} onChange={e => setNote(e.target.value)} maxLength={2000} disabled={busy}/></label><button type="submit" disabled={busy}>{busy ? 'Đang lưu…' : 'Lưu phản hồi'}</button></form>}
    {!editing && <small>Muốn thêm ghi chú, chọn “Sửa nhãn”; có thể giữ nhãn gốc.</small>}
    {busy && <p role="status">Đang lưu phản hồi…</p>}{message && <p role="status">{message}</p>}{error && <p className="error" role="alert">{error}</p>}
    {sample&&<ReviewEditor sample={sample} onChange={setSample}/>}
  </div>;
}
