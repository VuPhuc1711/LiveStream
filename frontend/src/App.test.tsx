// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from './App';
import { analyze, overviewExplanation, validateFile, visibleUnits, type Unit } from './api';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
const health = { ready: true, status: 'ok' };
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.useRealTimers(); });
it('validates extension, empty file and exact size boundary', () => {
  expect(validateFile(new File(['x'], 'a.exe'))).toContain('MP3');
  expect(validateFile(new File([], 'a.wav'))).toContain('rỗng');
  expect(validateFile({ name:'a.m4a', size:50*1024*1024 } as File)).toBeNull();
  expect(validateFile({ name:'a.m4a', size:50*1024*1024+1 } as File)).toContain('50 MiB');
});
it('filters safe sentences without relabeling and sorts chronologically', () => {
  const units = [{start:3,end:4,final_label:'CANH_BAO'}, {start:0,end:1,final_label:'KHONG_CANH_BAO'}, {start:1,end:2,final_label:'CAN_XAC_MINH'}] as Unit[];
  expect(visibleUnits(units,false).map(u=>u.final_label)).toEqual(['CAN_XAC_MINH','CANH_BAO']);
  expect(visibleUnits(units,true)).toHaveLength(3);
  expect(units[1].final_label).toBe('KHONG_CANH_BAO');
});
it('shows the model-only explanation without Rules', () => {
  const text = 'PhoBERT phân loại câu này dựa trên nội dung đã nhận dạng; người kiểm duyệt cần xem lại khi cần.';
  expect(overviewExplanation(text)).toBe(text);
});
it('sends multipart file without manually setting Content-Type', async () => {
  const fetcher = vi.fn().mockResolvedValue({ok:true,json:async()=>({})}); vi.stubGlobal('fetch',fetcher);
  const file = new File(['audio'], 'sample.m4a'); await analyze(file);
  const [url,init] = fetcher.mock.calls[0]; expect(url).toContain('/analyze-audio');
  expect(init.body.get('file')).toBe(file); expect(init.headers).toBeUndefined();
});
it.each([400,413,500])('shows a Vietnamese API error for %s without retry', async status => {
  const fetcher = vi.fn().mockResolvedValue({ok:false,status}); vi.stubGlobal('fetch',fetcher);
  await expect(analyze(new File(['x'],'a.wav'))).rejects.toThrow(status===413?'50 MiB':status===400?'không hợp lệ':'thất bại');
  expect(fetcher).toHaveBeenCalledTimes(1);
});
it('shows offline state with retry', async () => {
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new TypeError('network')));
  await act(async()=>root.render(<App/>)); expect(host.textContent).toContain('Mất kết nối backend');
  expect(host.querySelector('button')?.textContent).toBe('Thử lại');
  expect(host.textContent).toContain('Chưa xác nhận được phiên bản model');
});
it.each(['v03','v07'])('shows the model actually reported by backend: %s', async version => {
  const checkpoint = `models/phobert_${version}_example/best`;
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({...health,model_version:version,checkpoint,classifier:'phobert_only'})}));
  await act(async()=>root.render(<App/>));
  expect(host.textContent).toContain('Bản thử nghiệm hỗ trợ kiểm duyệt');
  expect(host.textContent).toContain(`Model backend đang nạp: PhoBERT ${version}`);
  expect(host.textContent).toContain('Chế độ phân loại: chỉ PhoBERT');
  expect(host.textContent).toContain(checkpoint);
});
it('renders a model-only result without Rules details', async () => {
  const prediction = {label:'CAN_XAC_MINH',confidence:0.72,probabilities:{KHONG_CANH_BAO:0.1,CAN_XAC_MINH:0.72,CANH_BAO:0.18}};
  const result = {filename:'sample.m4a',processing_time_seconds:1,transcript:'Bảo hành hai năm.',
    summary:{classification_unit_count:1,whisper_segment_count:1,final_label_counts:{CAN_XAC_MINH:1}},
    classification_units:[{unit_id:'U0001',source_segment_id:0,text:'Bảo hành hai năm.',start:0,end:2,
      timestamp_is_approximate:false,final_label:'CAN_XAC_MINH',final_confidence:0.72,
      phobert_result:prediction,explanation:'PhoBERT phân loại câu này dựa trên nội dung đã nhận dạng; người kiểm duyệt cần xem lại khi cần.'}]};
  const fetcher = vi.fn().mockResolvedValueOnce({ok:true,json:async()=>({...health,classifier:'phobert_only'})})
    .mockResolvedValueOnce({ok:true,json:async()=>result});
  vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<App/>));
  const input = host.querySelector('input[type=file]') as HTMLInputElement;
  Object.defineProperty(input,'files',{value:[new File(['x'],'sample.m4a')],configurable:true});
  await act(async()=>input.dispatchEvent(new Event('change',{bubbles:true})));
  await act(async()=>(host.querySelector('.primary') as HTMLButtonElement).click());
  expect(host.textContent).toContain('Chi tiết PhoBERT');
  expect(host.textContent).not.toContain('Rules');
  expect(host.textContent).toContain('Cần xác minh');
});
it('locks submission while pending, counts waiting time, renders results and clears on new selection', async () => {
  let finish!: (response: unknown)=>void;
  const fetcher = vi.fn().mockResolvedValueOnce({ok:true,json:async()=>health}).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
  vi.stubGlobal('fetch',fetcher); await act(async()=>root.render(<App/>));
  const input = host.querySelector('input[type=file]') as HTMLInputElement;
  const select = async (name:string) => { Object.defineProperty(input,'files',{value:[new File(['x'],name)],configurable:true}); await act(async()=>input.dispatchEvent(new Event('change',{bubbles:true}))); };
  await select('first.m4a'); vi.useFakeTimers();
  const button = host.querySelector('.primary') as HTMLButtonElement;
  await act(async()=>{button.click(); button.click();}); expect(button.disabled).toBe(true); expect(input.disabled).toBe(true);
  await act(async()=>vi.advanceTimersByTime(2000)); expect(host.textContent).toContain('Đã chờ 2 giây'); expect(fetcher).toHaveBeenCalledTimes(2);
  await act(async()=>finish({ok:true,json:async()=>({filename:'first.m4a',processing_time_seconds:2,transcript:'Xin chào',classification_units:[],summary:{classification_unit_count:0,whisper_segment_count:0,final_label_counts:{}}})}));
  expect(host.textContent).toContain('Kết quả phân tích'); expect(host.textContent).toContain('Không nhận dạng được câu nào');
  await select('next.wav'); expect(host.textContent).not.toContain('Kết quả phân tích'); expect(host.textContent).toContain('next.wav');
});
it('shows upload ASR gaps without calling them safe or showing rejected text as transcript',async()=>{
  const result={filename:'sample.wav',processing_time_seconds:1,transcript:'',classification_units:[],summary:{classification_unit_count:0,whisper_segment_count:0,final_label_counts:{}},asr_quality_warnings:[{window_id:'window-0',start:0,end:2,reason:'decode_quality',retried:true,pending_text:'Phụ đề không được xác nhận'}],asr_unresolved_windows:1};
  vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce({ok:true,json:async()=>health}).mockResolvedValueOnce({ok:true,json:async()=>result}));
  await act(async()=>root.render(<App/>));
  const input=host.querySelector('input[type=file]') as HTMLInputElement;
  Object.defineProperty(input,'files',{value:[new File(['x'],'sample.wav')],configurable:true});
  await act(async()=>input.dispatchEvent(new Event('change',{bubbles:true})));
  await act(async()=>(host.querySelector('.primary') as HTMLButtonElement).click());
  expect(host.querySelector('.asr-quality')?.textContent).toContain('không được đưa vào phân loại');
  expect(host.querySelector('.results .empty')?.textContent).toContain('Chưa có kết quả phân loại cho các đoạn đó.');
  expect(host.textContent).not.toContain('Phụ đề không được xác nhận');
  expect(host.querySelectorAll('article.unit')).toHaveLength(0);
});
