// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import SessionHistory from './SessionHistory';
import FeedbackEditor from './FeedbackEditor';
import { analyze, downloadSession, type Unit } from './api';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const id = '00000000-0000-0000-0000-000000000001';
const unit: Unit = {unit_id:'U001',segment_id:'segment1',source_segment_id:0,text:'Câu đã lưu.',start:1,end:3,timestamp_is_approximate:true,final_label:'CANH_BAO',final_confidence:.9,phobert_result:{label:'CANH_BAO',confidence:.9,probabilities:{KHONG_CANH_BAO:.1,CANH_BAO:.9}},explanation:'Kết quả gốc.',feedback:[]};
const item = {id,source:'recording',filename:'recording.webm',created_at:'2026-10-05T00:00:00+00:00',started_at:'2026-10-05T00:00:00+00:00',ended_at:'2026-10-05T00:00:02+00:00',status:'completed',processing_time_seconds:2,error:null,model_version:'binary_v04',checkpoint:'models/binary_v04/best',warning_count:1,segment_count:1};
const detail = {...item,transcript:'Câu đã lưu.',classification_units:[unit],summary:{classification_unit_count:1,whisper_segment_count:1,final_label_counts:{KHONG_CANH_BAO:0,CANH_BAO:1}}};
let host: HTMLDivElement, root: Root;
beforeEach(() => {window.history.replaceState(null,'','#');host=document.createElement('div');document.body.append(host);root=createRoot(host);});
afterEach(async()=>{await act(async()=>root.unmount());host.remove();window.history.replaceState(null,'','#');vi.unstubAllGlobals();vi.restoreAllMocks();vi.useRealTimers();});
const click = async (text: string) => act(async()=>Array.from(host.querySelectorAll('button')).find(b=>b.textContent===text)!.click());
it('reads saved history, details, filtering and pagination without changing original predictions',async()=>{
  const fetcher=vi.fn().mockImplementation((url:string)=>Promise.resolve({ok:true,json:async()=>url.includes('/sessions?')?{items:[item],page:1,page_size:10,total:11}:detail}));
  vi.stubGlobal('fetch',fetcher);await act(async()=>root.render(<SessionHistory/>));expect(fetcher).not.toHaveBeenCalled();
  await click('Mở lịch sử');expect(host.textContent).toContain('recording.webm');expect(host.textContent).toContain('Thu âm');
  await act(async()=>(host.querySelector('.session-card') as HTMLButtonElement).click());expect(host.textContent).toContain('Dự đoán gốc: Cảnh báo');expect(host.textContent).toContain('Thời gian ước lượng');expect(window.location.hash).toBe('#session='+id);expect(host.querySelector('audio')).toBeNull();
  await click('Trang sau');expect(fetcher.mock.calls.at(-1)![0]).toContain('page=2');
  const select=host.querySelector('.history-filters select') as HTMLSelectElement;
  await act(async()=>{select.value='warnings';select.dispatchEvent(new Event('change',{bubbles:true}));});
  expect(fetcher.mock.calls.at(-1)![0]).toContain('warnings_only=true');expect(fetcher.mock.calls.at(-1)![0]).toContain('page=1');
});
it('reopens the bookmarked session on reload',async()=>{
  window.history.replaceState(null,'','#session='+id);
  const fetcher=vi.fn().mockImplementation((url:string)=>Promise.resolve({ok:true,json:async()=>url.includes('?')?{items:[],total:0,page:1,page_size:10}:detail}));vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<SessionHistory/>));expect(host.textContent).toContain('Chi tiết phiên');expect(host.textContent).toContain('binary_v04');
});
it('shows saved unresolved ASR notices on reload without displaying rejected raw text or relabeling it',async()=>{
  window.history.replaceState(null,'','#session='+id);
  const saved={...detail,asr_quality_warnings:[{window_id:'window-1',start:3,end:5.2,reason:'decode_quality',retried:true,pending_text:'Nội dung chưa xác nhận'}],asr_unresolved_windows:1};
  vi.stubGlobal('fetch',vi.fn().mockImplementation((url:string)=>Promise.resolve({ok:true,json:async()=>url.includes('?')?{items:[],total:0,page:1,page_size:10}:saved})));
  await act(async()=>root.render(<SessionHistory/>));
  expect(host.querySelector('.asr-quality')?.textContent).toContain('1 đoạn chưa nhận dạng đáng tin cậy');
  expect(host.querySelector('.asr-quality')?.textContent).toContain('00:03 – 00:05.200');
  expect(host.textContent).not.toContain('Nội dung chưa xác nhận');
  expect(host.querySelectorAll('article.unit')).toHaveLength(1);
  expect(host.querySelector('article.unit')?.textContent).toContain('Câu đã lưu.');
});
it('labels browser-tab sessions in history rather than upload',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({items:[{...item,source:'browser_tab_live',filename:'Tab livestream'}],page:1,page_size:10,total:1})}));
  await act(async()=>root.render(<SessionHistory/>));await click('Mở lịch sử');
  expect(host.querySelector('.session-card')?.textContent).toContain('Giám sát tab livestream');
});
it('shows waiting, empty history and network failures with retry',async()=>{
  let finish!:(value:unknown)=>void;
  const fetcher=vi.fn().mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<SessionHistory/>));await click('Mở lịch sử');expect(host.textContent).toContain('Đang tải lịch sử');
  await act(async()=>finish({ok:true,json:async()=>({items:[],total:0,page:1,page_size:10})}));expect(host.textContent).toContain('Chưa có phiên phù hợp');
  fetcher.mockRejectedValueOnce(new Error('offline'));await click('Tải lại lịch sử');expect(host.textContent).toContain('Không kết nối được backend');
});
it('saves corrections separately, locks duplicate sends and exposes failures',async()=>{
  let finish!:(value:unknown)=>void;
  const fetcher=vi.fn().mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<FeedbackEditor unit={unit}/>));await click('Sửa nhãn');
  await act(async()=>{const select=host.querySelector('select')!;select.value='KHONG_CANH_BAO';select.dispatchEvent(new Event('change',{bubbles:true}));});
  await act(async()=>{const form=host.querySelector('form')!;form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));});
  expect(fetcher).toHaveBeenCalledTimes(1);expect(JSON.parse(fetcher.mock.calls[0][1].body).label).toBe('KHONG_CANH_BAO');
  await act(async()=>finish({ok:true,json:async()=>({id:'f1',segment_id:'segment1',label:'KHONG_CANH_BAO',note:'',created_at:item.created_at})}));expect(host.textContent).toContain('Đã lưu phản hồi');expect(host.textContent).toContain('Dự đoán gốc: Cảnh báo');expect(host.querySelector('.feedback-log')?.textContent).toContain('Không cảnh báo');
  fetcher.mockResolvedValueOnce({ok:false,status:500});await click('Dự đoán đúng');expect(host.textContent).toContain('Backend xử lý thất bại');
});
it('tags microphone requests as recording and downloads JSON via the export API',async()=>{
  const fetcher=vi.fn().mockResolvedValue({ok:true,json:async()=>({})});vi.stubGlobal('fetch',fetcher);
  await analyze(new File(['x'],'recording.webm'),'recording');expect(fetcher.mock.calls[0][1].body.get('source')).toBe('recording');
  const create=vi.spyOn(URL,'createObjectURL').mockReturnValue('blob:export');const revoke=vi.spyOn(URL,'revokeObjectURL').mockImplementation(()=>{});const anchor=vi.spyOn(HTMLAnchorElement.prototype,'click').mockImplementation(()=>{});
  vi.useFakeTimers();await downloadSession(id);expect(fetcher.mock.calls.at(-1)![0]).toContain('/export');expect(create).toHaveBeenCalled();expect(anchor).toHaveBeenCalledOnce();vi.advanceTimersByTime(1000);expect(revoke).toHaveBeenCalledWith('blob:export');
});
