import { useEffect, useRef, useState } from 'react';
import { approveReviewSample, editReviewSample, labels, type BinaryLabel, type ReviewContent, type ReviewSample } from './api';

export function ReviewFields({content,setContent,busy}: {content:ReviewContent;setContent:(next:ReviewContent)=>void;busy:boolean}) {
  const change=(key:keyof ReviewContent,value:string)=>setContent({...content,[key]:value});
  return <>
    <label>Văn bản dùng cho dữ liệu<textarea required maxLength={4000} value={content.text} disabled={busy} onChange={e=>change('text',e.target.value)}/></label>
    <label>Nhãn đề xuất / xác nhận<select value={content.label} disabled={busy} onChange={e=>change('label',e.target.value as BinaryLabel)}><option value="KHONG_CANH_BAO">Không cảnh báo</option><option value="CANH_BAO">Cảnh báo</option></select></label>
    <label>Tình huống / nhóm nội dung<input required maxLength={200} value={content.situation} disabled={busy} onChange={e=>change('situation',e.target.value)}/></label>
    <label>Lý do sửa / gán nhãn<textarea required maxLength={2000} value={content.reason} disabled={busy} onChange={e=>change('reason',e.target.value)}/></label>
    <label>Family (các biến thể cùng ý / buổi thu)<input required maxLength={120} value={content.family_id} disabled={busy} onChange={e=>change('family_id',e.target.value)}/></label>
  </>;
}
const contentOf=(s:ReviewSample):ReviewContent=>({text:s.text,label:s.label,family_id:s.family_id,situation:s.situation,reason:s.reason});
export default function ReviewEditor({sample,onChange}: {sample:ReviewSample;onChange:(s:ReviewSample)=>void}) {
  const [content,setContent]=useState(contentOf(sample));
  const [reviewer,setReviewer]=useState('');
  const [confirmed,setConfirmed]=useState(false);
  const [busy,setBusy]=useState(false);
  const [message,setMessage]=useState('');
  const [error,setError]=useState('');
  const lock=useRef(false);
  useEffect(()=>{setContent(contentOf(sample));setConfirmed(false);},[sample.id,sample.revision]);
  const dirty=JSON.stringify(content)!==JSON.stringify(contentOf(sample));
  async function act(approval:boolean) {
    if(lock.current)return;
    lock.current=true;setBusy(true);setError('');setMessage('');
    try {
      const next=approval?await approveReviewSample(sample,reviewer):await editReviewSample(sample,content,reviewer);
      onChange(next);window.dispatchEvent(new Event('review-samples-changed'));
      setMessage(approval?'Đã duyệt phiên bản này; mẫu có thể xuất CSV.':'Đã lưu sửa đổi. Phiên bản mới cần duyệt lại.');
    } catch(e){setError(e instanceof Error?e.message:'Không lưu được mẫu.');}
    finally{lock.current=false;setBusy(false);}
  }
  return <div className="review-editor">
    <p><strong>{sample.status==='DA_DUYET'?'Đã duyệt':'Chờ duyệt'}</strong> · Phiên bản {sample.revision} · Nguồn: {sample.source_type}</p>
    {sample.reviewer&&<p>Người duyệt: {sample.reviewer} · {new Date(sample.reviewed_at!).toLocaleString('vi-VN')}</p>}
    {sample.original_text!==null&&<details><summary>Bằng chứng gốc (giữ nguyên)</summary><p>{sample.original_text}</p><p>Dự đoán gốc: {sample.predicted_label?labels[sample.predicted_label]:'—'} · {sample.model_version}</p><ul>{Object.entries(sample.probabilities||{}).map(([key,value])=><li key={key}>{labels[key as BinaryLabel]}: {(value*100).toFixed(2)}% xác suất mô hình</li>)}</ul><small>Phiên: {sample.session_id} · Feedback: {sample.feedback_id}. Không xác nhận đã nghe audio chỉ từ transcript.</small></details>}
    <form onSubmit={e=>{e.preventDefault();void act(false);}}>
      <ReviewFields content={content} setContent={setContent} busy={busy}/>
      <label>Tên người đang sửa / duyệt<input value={reviewer} maxLength={120} disabled={busy} onChange={e=>setReviewer(e.target.value)} placeholder="Nhập tên của bạn khi duyệt"/></label>
      <button disabled={busy||!dirty} type="submit">Lưu sửa đổi (cần duyệt lại)</button>
    </form>
    <label className="check"><input type="checkbox" checked={confirmed} disabled={busy||dirty} onChange={e=>setConfirmed(e.target.checked)}/>Tôi đã kiểm tra văn bản và nhãn của phiên bản này, đồng ý đưa vào dữ liệu bổ sung.</label>
    <button disabled={busy||dirty||!confirmed||!reviewer.trim()||sample.status==='DA_DUYET'} onClick={()=>void act(true)}>Duyệt đưa vào dữ liệu</button>
    {dirty&&<p>Lưu sửa đổi trước khi duyệt; nhãn và văn bản được duyệt cùng một phiên bản.</p>}
    {sample.history&&<details><summary>Lịch sử thay đổi ({sample.history.length})</summary>{sample.history.map(h=><div className="review-revision" key={h.revision}><strong>v{h.revision} · {h.action}</strong><p>{new Date(h.created_at).toLocaleString('vi-VN')} · {h.actor||'Chưa có người duyệt'}</p><p>{h.snapshot.text}</p><p>{labels[h.snapshot.label]} · {h.snapshot.status} · {h.snapshot.reason}</p></div>)}</details>}
    {busy&&<p role="status">Đang lưu…</p>}{message&&<p role="status">{message}</p>}{error&&<p className="error" role="alert">{error}</p>}
  </div>;
}
