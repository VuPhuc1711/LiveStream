// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import App from './App';
import { validateResult, visibleUnits, type Result } from './api';

const result: Result = {classification_mode:'binary',filename:'demo.m4a',processing_time_seconds:1,
  transcript:'Xin chào.',summary:{classification_unit_count:1,whisper_segment_count:1,
  final_label_counts:{KHONG_CANH_BAO:1,CANH_BAO:0}},classification_units:[{
  unit_id:'u1',source_segment_id:0,text:'Xin chào.',start:0,end:1,timestamp_is_approximate:false,
  final_label:'KHONG_CANH_BAO',final_confidence:0.8,explanation:'Mô tả.',
  phobert_result:{label:'KHONG_CANH_BAO',confidence:0.8,probabilities:{KHONG_CANH_BAO:0.8,CANH_BAO:0.2}}}]};
it('accepts genuine binary probabilities, filters safe units and rejects renamed three-class output', () => {
  expect(validateResult(result)).toBe(result);
  expect(visibleUnits(result.classification_units,false)).toHaveLength(0);
  expect(visibleUnits(result.classification_units,true)).toHaveLength(1);
  const bad=structuredClone(result);bad.classification_units[0].phobert_result.probabilities.CAN_XAC_MINH=0;
  expect(()=>validateResult(bad)).toThrow('hai nhãn');
});
it('binary UI hides verification statistics and preserves file upload', async () => {
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce({ok:true,json:async()=>({ready:true,
    classification_mode:'binary',model_version:'binary_v01',classifier:'phobert_only'})})
    .mockResolvedValueOnce({ok:true,json:async()=>result}));
  try {
    await act(async()=>root.render(<App/>));
    const input=host.querySelector('input[type=file]') as HTMLInputElement;
    Object.defineProperty(input,'files',{value:[new File(['x'],'demo.m4a')]});
    await act(async()=>input.dispatchEvent(new Event('change',{bubbles:true})));
    await act(async()=>(host.querySelector('.primary') as HTMLButtonElement).click());
    expect(host.textContent).toContain('Không có cảnh báo.');
    expect(host.textContent).not.toContain('Cần xác minh');
    expect(host.querySelector('button')?.textContent).toBe('Thử lại');
  } finally {await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});
