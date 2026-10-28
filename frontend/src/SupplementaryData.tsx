import { useEffect, useRef, useState } from 'react';
import { createReviewSample, downloadReviewExport, getReviewExports, getReviewSample, getReviewSamples, labels, makeReviewExport, type ReviewContent, type ReviewExport, type ReviewPage, type ReviewSample } from './api';
import ReviewEditor, { ReviewFields } from './ReviewEditor';

const empty:ReviewContent={text:'',label:'KHONG_CANH_BAO',family_id:'',situation:'',reason:''};
export default function SupplementaryData() {
  const [open,setOpen]=useState(false),[page,setPage]=useState(1),[status,setStatus]=useState(''),[label,setLabel]=useState(''),[situation,setSituation]=useState('');
  const [data,setData]=useState<ReviewPage|null>(null),[exports,setExports]=useState<ReviewExport[]>([]),[selected,setSelected]=useState<ReviewSample|null>(null);
  const [refresh,setRefresh]=useState(0),[loading,setLoading]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
  const [createOpen,setCreateOpen]=useState(false),[content,setContent]=useState(empty),[source,setSource]=useState<'USER_ENTERED'|'AI_GENERATED'>('USER_ENTERED'),[parent,setParent]=useState('');
  const lock=useRef(false),selection=useRef(0);
  useEffect(()=>{const listener=()=>setRefresh(v=>v+1);window.addEventListener('review-samples-changed',listener);return()=>window.removeEventListener('review-samples-changed',listener);},[]);
  useEffect(()=>{
    if(!open)return;
    let active=true;setLoading(true);setError('');
    Promise.all([getReviewSamples(page,status,label,situation),getReviewExports()]).then(([rows,files])=>{if(active){setData(rows);setExports(files);}}).catch(e=>{if(active)setError(e.message);}).finally(()=>{if(active)setLoading(false);});
    return()=>{active=false;};
  },[open,page,status,label,situation,refresh]);
  async function select(id:string){const token=++selection.current;setError('');try{const row=await getReviewSample(id);if(token===selection.current)setSelected(row);}catch(e){if(token===selection.current)setError(e instanceof Error?e.message:'Không đọc được mẫu.');}}
  async function task(action:()=>Promise<void>){if(lock.current)return;lock.current=true;setBusy(true);setError('');setMessage('');try{await action();setRefresh(v=>v+1);}catch(e){setError(e instanceof Error?e.message:'Không xử lý được dữ liệu.');}finally{lock.current=false;setBusy(false);}}
  return <section className="history-panel" aria-labelledby="supplement-heading"><h2 id="supplement-heading">Dữ liệu bổ sung</h2><p>Chỉ xuất phiên bản đã được duyệt. Chưa chia split, chưa ghép train/validation hoặc train model. Câu AI luôn giữ nguồn AI_GENERATED.</p><button aria-expanded={open} onClick={()=>setOpen(!open)}>{open?'Thu gọn dữ liệu':'Mở dữ liệu bổ sung'}</button>
    {open&&<>
      {data&&<p>{data.counts.DA_DUYET} đã duyệt · {data.counts.CHO_DUYET} chờ duyệt</p>}
      {data?.accumulation&&<div className="review-editor" aria-label="Thống kê tích lũy dữ liệu">
        <h3>Đã duyệt, chưa sử dụng để train: {data.accumulation.unused_training_count} mẫu</h3>
        <p>{data.accumulation.family_count} family · {data.accumulation.exported_current_revision_count} phiên bản hiện tại đã xuất CSV · {data.accumulation.trained_current_revision_count} phiên bản hiện tại đã dùng train · {data.accumulation.validation_only_count} mẫu chỉ dùng validation</p>
        <p>Xuất CSV không đánh dấu đã train. Thống kê theo phiên bản đã duyệt hiện tại, trên toàn bộ dữ liệu bổ sung. Không tự train khi đủ số lượng.</p>
        {([
          ['Nhãn',data.accumulation.labels],['Nguồn',data.accumulation.sources],
          ['Tình huống',data.accumulation.situations],['Family',data.accumulation.families]
        ] as [string,Record<string,number>][]).map(([title,counts])=><details key={title}><summary>{title} — mẫu chưa dùng train</summary><ul>{Object.entries(counts).map(([key,count])=><li key={key}>{labels[key as keyof typeof labels] || ({REAL_CONFIRMED:'Lời nói thực tế đã xác nhận',AI_GENERATED:'AI soạn',OTHER_NOT_CONFIRMED_REAL:'Nguồn khác, chưa xác nhận là lời nói thực tế'} as Record<string,string>)[key] || key}: {count}</li>)}</ul></details>)}
      </div>}
      <div className="history-filters"><label>Trạng thái duyệt<select value={status} onChange={e=>{setStatus(e.target.value);setPage(1);}}><option value="">Tất cả</option><option value="CHO_DUYET">Chờ duyệt</option><option value="DA_DUYET">Đã duyệt</option></select></label><label>Nhãn dữ liệu<select value={label} onChange={e=>{setLabel(e.target.value);setPage(1);}}><option value="">Hai nhãn</option><option value="KHONG_CANH_BAO">Không cảnh báo</option><option value="CANH_BAO">Cảnh báo</option></select></label><label>Tình huống lọc<select value={situation} onChange={e=>{setSituation(e.target.value);setPage(1);}}><option value="">Tất cả tình huống</option>{data?.situations.map(s=><option key={s}>{s}</option>)}</select></label><button disabled={loading||busy} onClick={()=>setRefresh(v=>v+1)}>Tải lại dữ liệu</button></div>
      <div className="history-actions"><button disabled={busy} onClick={()=>setCreateOpen(!createOpen)}>Nhập câu đề xuất</button><button disabled={busy||!data?.counts.DA_DUYET} onClick={()=>void task(async()=>{const file=await makeReviewExport();await downloadReviewExport(file);setMessage(`Đã xuất ${file.sample_count} mẫu đã duyệt: ${file.filename}`);})}>Xuất CSV các mẫu đã duyệt</button></div>
      {createOpen&&<form className="review-editor" onSubmit={e=>{e.preventDefault();void task(async()=>{const row=await createReviewSample(content,source,parent||undefined);setSelected(row);setCreateOpen(false);setContent(empty);setMessage('Đã tạo câu đề xuất chờ duyệt.');});}}><ReviewFields content={content} setContent={setContent} busy={busy}/><label>Nguồn câu<select value={source} disabled={busy} onChange={e=>setSource(e.target.value as typeof source)}><option value="USER_ENTERED">Người dùng nhập (chưa xác nhận là lời nói thực tế)</option><option value="AI_GENERATED">AI soạn</option></select></label><label>ID mẫu nguồn (nếu là biến thể)<input value={parent} onChange={e=>setParent(e.target.value)} disabled={busy}/></label><button disabled={busy}>Lưu đề xuất chờ duyệt</button></form>}
      {loading&&<p role="status">Đang tải dữ liệu…</p>}{busy&&<p role="status">Đang lưu / xuất…</p>}{error&&<p className="error" role="alert">{error}</p>}{message&&<p role="status">{message}</p>}
      {data&&<><p>{data.total} mẫu phù hợp · Trang {page}</p>{!data.items.length&&<p>Chưa có mẫu phù hợp.</p>}<ul className="session-list">{data.items.map(s=><li key={s.id}><button className="session-card" aria-pressed={selected?.id===s.id} onClick={()=>void select(s.id)}><strong>{s.text}</strong><span>{labels[s.label]} · {s.status==='DA_DUYET'?'Đã duyệt':'Chờ duyệt'} · {s.source_type} · {s.situation}</span><small>{s.id}</small></button></li>)}</ul><div className="history-actions"><button disabled={page<=1||loading} onClick={()=>setPage(p=>p-1)}>Mẫu trang trước</button><button disabled={page*data.page_size>=data.total||loading} onClick={()=>setPage(p=>p+1)}>Mẫu trang sau</button></div></>}
      {selected&&<div className="session-detail"><h3>Rà soát mẫu {selected.id}</h3><button onClick={()=>void select(selected.id)}>Tải lại mẫu và lịch sử</button><ReviewEditor key={selected.id} sample={selected} onChange={s=>{setSelected(s);void select(s.id);setRefresh(v=>v+1);}}/></div>}
      {exports.length>0&&<details><summary>Các bản CSV đã xuất ({exports.length})</summary>{exports.map(file=><div className="review-revision" key={file.id}><p>{file.filename} · {file.sample_count} mẫu · {new Date(file.created_at).toLocaleString('vi-VN')}</p><small className="checkpoint">SHA-256: {file.sha256}<br/>{file.local_path}</small><button disabled={busy} onClick={()=>void task(async()=>{await downloadReviewExport(file);})}>Tải lại CSV</button></div>)}</details>}
    </>}
  </section>;
}
