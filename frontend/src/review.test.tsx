// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import FeedbackEditor from './FeedbackEditor';
import ReviewEditor from './ReviewEditor';
import SupplementaryData from './SupplementaryData';
import { type ReviewSample, type Unit } from './api';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT:boolean }).IS_REACT_ACT_ENVIRONMENT=true;
const sample:ReviewSample={id:'s1',feedback_id:'f1',segment_id:'seg1',session_id:'sess1',source_sample_id:null,original_text:'Nếu rõ hết hàng.',predicted_label:'KHONG_CANH_BAO',probabilities:{KHONG_CANH_BAO:.97,CANH_BAO:.03},model_version:'binary_v04',text:'Nếu giỏ hết hàng, sang Shopee đặt nhé.',label:'CANH_BAO',family_id:'SESSION_sess1',situation:'DIEU_HUONG',reason:'Mời đặt ngoài TikTok.',source_type:'USER_FEEDBACK',status:'CHO_DUYET',reviewer:null,reviewed_at:null,revision:1};
const unit:Unit={unit_id:'U1',segment_id:'seg1',source_segment_id:0,text:sample.original_text!,start:0,end:2,timestamp_is_approximate:true,final_label:'KHONG_CANH_BAO',final_confidence:.97,phobert_result:{label:'KHONG_CANH_BAO',confidence:.97,probabilities:sample.probabilities!},explanation:'Gốc.',feedback:[]};
let host:HTMLDivElement,root:Root;
beforeEach(()=>{host=document.createElement('div');document.body.append(host);root=createRoot(host);});
afterEach(async()=>{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();vi.restoreAllMocks();});
const click=async(text:string)=>act(async()=>Array.from(host.querySelectorAll('button')).find(b=>b.textContent===text)!.click());
async function value(element:HTMLInputElement|HTMLTextAreaElement,text:string){await act(async()=>{const prototype=element instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(prototype,'value')!.set!.call(element,text);element.dispatchEvent(new Event('input',{bubbles:true}));});}
const ok=(value:unknown)=>({ok:true,json:async()=>value});
it('distinguishes exported approved revisions from training usage and shows provenance',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockImplementation((url:string)=>Promise.resolve(ok(
    url.includes('/exports')?[]:{items:[],total:0,page:1,page_size:20,situations:[],
      counts:{DA_DUYET:10,CHO_DUYET:0},accumulation:{approved_count:10,
      unused_training_count:10,trained_current_revision_count:0,exported_current_revision_count:10,
      validation_only_count:0,family_count:1,families:{SESSION_TEST:10},
      situations:{DIEU_HUONG:10},labels:{CANH_BAO:6,KHONG_CANH_BAO:4},
      sources:{REAL_CONFIRMED:1,AI_GENERATED:9}}}))));
  await act(async()=>root.render(<SupplementaryData/>));await click('Mở dữ liệu bổ sung');
  expect(host.textContent).toContain('chưa sử dụng để train: 10 mẫu');
  expect(host.textContent).toContain('10 phiên bản hiện tại đã xuất CSV');
  expect(host.textContent).toContain('0 phiên bản hiện tại đã dùng train');
  expect(host.textContent).toContain('Lời nói thực tế đã xác nhận: 1');
  expect(host.textContent).toContain('AI soạn: 9');
  expect(host.textContent).toContain('SESSION_TEST: 10');
});
it('keeps text corrections separate and requires explicit approval before export eligibility',async()=>{
  const f={id:'f1',segment_id:'seg1',label:'CANH_BAO',note:'Điều hướng.',corrected_text:sample.text,situation:'DIEU_HUONG',created_at:'2026-10-06T00:00:00Z'};
  const fetcher=vi.fn().mockResolvedValueOnce(ok(f)).mockResolvedValueOnce(ok(sample));vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<FeedbackEditor unit={unit}/>));await click('Sửa nhãn');
  await value(host.querySelector('textarea')!,sample.text);
  await act(async()=>{const s=host.querySelector('select')!;s.value='CANH_BAO';s.dispatchEvent(new Event('change',{bubbles:true}));});
  await click('Lưu phản hồi');expect(fetcher).toHaveBeenCalledTimes(1);
  const sent=JSON.parse(fetcher.mock.calls[0][1].body);expect(sent.corrected_text).toBe(sample.text);expect(sent.label).toBe('CANH_BAO');
  expect(host.textContent).toContain('Dự đoán gốc: Không cảnh báo');await click('Chuẩn bị duyệt đưa vào dữ liệu');
  const approve=Array.from(host.querySelectorAll('button')).find(b=>b.textContent==='Duyệt đưa vào dữ liệu')!;expect(approve.disabled).toBe(true);
  expect(fetcher.mock.calls.every(([url])=>!String(url).includes('/approve'))).toBe(true);
});
it('locks duplicate approval and sends reviewer plus exact revision',async()=>{
  let finish!:(value:unknown)=>void;const fetcher=vi.fn().mockImplementation(()=>new Promise(resolve=>finish=resolve));vi.stubGlobal('fetch',fetcher);
  const changed=vi.fn();await act(async()=>root.render(<ReviewEditor sample={sample} onChange={changed}/>));
  await value(host.querySelector('input[placeholder]')!,'Người duyệt thử');await act(async()=>(host.querySelector('input[type=checkbox]') as HTMLInputElement).click());
  await click('Duyệt đưa vào dữ liệu');await click('Duyệt đưa vào dữ liệu');expect(fetcher).toHaveBeenCalledOnce();
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({expected_revision:1,reviewer:'Người duyệt thử',confirm_text:true});
  await act(async()=>finish(ok({...sample,status:'DA_DUYET',revision:2,reviewer:'Người duyệt thử'})));
  expect(changed).toHaveBeenCalledOnce();expect(host.textContent).toContain('Đã duyệt phiên bản này');
});
it('edits approved data into pending, shows errors and original evidence',async()=>{
  const approved={...sample,status:'DA_DUYET' as const,revision:2,reviewer:'Người duyệt thử',reviewed_at:'2026-10-06T00:00:00Z'};
  const fetcher=vi.fn().mockResolvedValueOnce(ok({...sample,revision:3,text:'Câu sửa mới.'})).mockResolvedValueOnce({ok:false,status:409,json:async()=>({detail:'Mẫu đã thay đổi; tải lại.'})});vi.stubGlobal('fetch',fetcher);
  const changed=vi.fn();await act(async()=>root.render(<ReviewEditor sample={approved} onChange={changed}/>));
  expect(host.textContent).toContain('Nếu rõ hết hàng.');await value(host.querySelector('textarea')!,'Câu sửa mới.');
  expect((host.querySelector('input[type=checkbox]') as HTMLInputElement).disabled).toBe(true);
  await act(async()=>host.querySelector('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  expect(changed.mock.calls[0][0].status).toBe('CHO_DUYET');expect(host.textContent).toContain('Phiên bản mới cần duyệt lại');
  await act(async()=>host.querySelector('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  expect(host.textContent).toContain('Mẫu đã thay đổi; tải lại');
});
it('lists status/label/situation filters and imports AI only as pending',async()=>{
  const rows={items:[sample],total:1,page:1,page_size:10,situations:['DIEU_HUONG'],counts:{DA_DUYET:0,CHO_DUYET:1}};
  const fetcher=vi.fn().mockImplementation((url:string,options?:RequestInit)=>Promise.resolve(ok(options?.method==='POST'?sample:url.endsWith('/exports')?[]:rows)));vi.stubGlobal('fetch',fetcher);
  await act(async()=>root.render(<SupplementaryData/>));expect(fetcher).not.toHaveBeenCalled();await click('Mở dữ liệu bổ sung');
  expect(host.textContent).toContain('0 đã duyệt · 1 chờ duyệt');expect(Array.from(host.querySelectorAll('button')).find(b=>b.textContent==='Xuất CSV các mẫu đã duyệt')!.disabled).toBe(true);
  const selects=host.querySelectorAll('.history-filters select');await act(async()=>{const s=selects[0] as HTMLSelectElement;s.value='CHO_DUYET';s.dispatchEvent(new Event('change',{bubbles:true}));});
  expect(fetcher.mock.calls.some(([url])=>String(url).includes('status=CHO_DUYET'))).toBe(true);
  await click('Nhập câu đề xuất');const form=host.querySelector('form')!;
  await value(form.querySelector('textarea')!,'Câu đề xuất mới.');const select=form.querySelectorAll('select')[1];
  await act(async()=>{select.value='AI_GENERATED';select.dispatchEvent(new Event('change',{bubbles:true}));form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));});
  const post=fetcher.mock.calls.find(([,options])=>options?.method==='POST')!;
  expect(JSON.parse(post[1].body).source_type).toBe('AI_GENERATED');expect(JSON.parse(post[1].body).reviewer).toBeUndefined();
});
