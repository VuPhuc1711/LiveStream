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
it('keeps confidence and raw rule evidence in details rather than the overview', () => {
  const text = 'PhoBERT dự đoán CANH_BAO với confidence 70.73%. Rules không khớp cụm từ nào. Đạt ngưỡng cảnh báo 55%; chọn CANH_BAO.';
  expect(overviewExplanation(text)).toBe('Đạt ngưỡng cảnh báo 55%; chọn CANH_BAO.');
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
