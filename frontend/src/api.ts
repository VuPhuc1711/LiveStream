export type Label = 'KHONG_CANH_BAO' | 'CAN_XAC_MINH' | 'CANH_BAO';
export type BinaryLabel = Exclude<Label, 'CAN_XAC_MINH'>;
export type ClassificationMode = 'binary' | 'three_class';
export interface Feedback { id: string; segment_id: string; label: BinaryLabel; note: string; created_at: string; corrected_text?: string | null; situation?: string }
export interface Unit { segment_id?: string; feedback?: Feedback[] }
export interface AsrQualityWarning { window_id: string; start: number; end: number; reason: string; retried: boolean; pending_text?: string }
export interface AsrQualityInfo { asr_quality_warnings?: AsrQualityWarning[]; asr_unresolved_windows?: number; asr_recovered_windows?: number }
export interface Result extends AsrQualityInfo { session_id?: string }
export interface SessionItem { id: string; source: 'upload' | 'recording' | 'microphone_live' | 'browser_tab_live'; filename: string; created_at: string; started_at: string; ended_at: string | null; status: 'processing' | 'completed' | 'failed'; processing_time_seconds: number | null; error: string | null; model_version: string; checkpoint: string; warning_count: number; segment_count: number }
export interface SessionPage { items: SessionItem[]; total: number; page: number; page_size: number }
export interface SessionDetail extends Omit<SessionItem, 'warning_count' | 'segment_count'>, AsrQualityInfo { transcript: string; summary: Result['summary']; classification_units: Unit[] }
export const labels: Record<Label, string> = { KHONG_CANH_BAO: 'Không cảnh báo', CAN_XAC_MINH: 'Cần xác minh', CANH_BAO: 'Cảnh báo' };
export interface Unit { unit_id: string; source_segment_id: string | number; text: string; start: number; end: number; timestamp_is_approximate: boolean; final_label: Label; final_confidence: number; phobert_result: { label: Label; confidence: number; probabilities: Partial<Record<Label, number>> }; explanation: string }
export interface Result { classification_mode?: ClassificationMode; filename: string; processing_time_seconds: number; transcript: string; summary: { classification_unit_count: number; whisper_segment_count: number; final_label_counts: Partial<Record<Label, number>> }; classification_units: Unit[] }
export interface Health { classification_mode?: ClassificationMode; labels?: Label[]; status: string; ready: boolean; checkpoint: string | null; model_version: string | null; classifier: string; whisper: string; phobert: string }
export const liveSocketUrl = () => base.replace(/^http/, 'ws') + '/ws/monitor';
const base = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
export function validateFile(file: File): string | null {
  if (!/\.(mp3|m4a|wav|mp4|webm)$/i.test(file.name)) return 'Chỉ hỗ trợ file MP3, M4A, WAV, MP4 hoặc WebM.';
  if (file.size === 0) return 'File rỗng. Vui lòng chọn file khác.';
  if (file.size > 50 * 1024 * 1024) return 'File vượt giới hạn 50 MiB.';
  return null;
}
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(base + path, options); }
  catch { throw new Error('Không kết nối được backend. Kiểm tra server và kết nối rồi thử lại.'); }
  if (!response.ok) {
    if (path.startsWith('/supplementary') && [400,409].includes(response.status)) {
      try { const detail = await response.json(); if (typeof detail.detail === 'string') throw new Error(detail.detail); }
      catch (e) { if (e instanceof Error && e.name !== 'SyntaxError') throw e; }
    }
    const messages: Record<number, string> = { 409: 'Tài nguyên đang bận với phiên khác. Hãy thử lại sau.', 400: 'File không hợp lệ, rỗng hoặc không đọc được âm thanh.', 404: 'Không tìm thấy phiên hoặc đoạn phân tích.', 413: 'File vượt giới hạn 50 MiB.', 422: 'Dữ liệu gửi lên không hợp lệ.', 503: 'Model chưa sẵn sàng.' };
    throw new Error(messages[response.status] || 'Backend xử lý thất bại. Vui lòng thử lại sau.');
  }
  try { return await response.json() as T; }
  catch { throw new Error('Backend trả về dữ liệu không hợp lệ.'); }
}
export const getHealth = () => request<Health>('/health', { signal: AbortSignal.timeout(8000) });
export function validateResult(result: Result): Result {
  if (result.classification_mode === 'binary') {
    const expected = ['CANH_BAO', 'KHONG_CANH_BAO'];
    const keysMatch = (value: object) => JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expected);
    if (!keysMatch(result.summary.final_label_counts) || result.classification_units.some(unit =>
      unit.final_label === 'CAN_XAC_MINH' || unit.phobert_result.label === 'CAN_XAC_MINH' ||
      !keysMatch(unit.phobert_result.probabilities))) {
      throw new Error('Backend trả nhãn không khớp chế độ hai nhãn. Không thể hiển thị kết quả.');
    }
  }
  return result;
}
export async function analyze(file: File, source: 'upload' | 'recording' = 'upload') { const form = new FormData(); form.append('file', file); form.append('source', source); return validateResult(await request<Result>('/analyze-audio', { method: 'POST', body: form })); }
export const getSessions = (page: number, warningsOnly: boolean, status: string = '') => request<SessionPage>(`/sessions?page=${page}&page_size=10&warnings_only=${warningsOnly}${status ? '&status=' + encodeURIComponent(status) : ''}`);
export const getSession = (id: string) => request<SessionDetail>(`/sessions/${encodeURIComponent(id)}`);
export const saveFeedback = (id: string, label: BinaryLabel, note: string, corrected_text?: string, situation?: string) => request<Feedback>(`/segments/${encodeURIComponent(id)}/feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label, note, corrected_text, situation }) });

export interface ReviewContent { text: string; label: BinaryLabel; family_id: string; situation: string; reason: string }
export interface ReviewSample extends ReviewContent { id: string; feedback_id: string | null; segment_id: string | null; session_id: string | null; source_sample_id: string | null; original_text: string | null; predicted_label: BinaryLabel | null; probabilities: Record<BinaryLabel,number> | null; model_version: string | null; source_type: 'AI_GENERATED' | 'USER_ENTERED' | 'USER_FEEDBACK' | 'USER_CONFIRMED_SPEECH'; status: 'CHO_DUYET' | 'DA_DUYET'; reviewer: string | null; reviewed_at: string | null; revision: number; history?: { revision: number; action: string; actor: string | null; created_at: string; snapshot: ReviewSample }[] }
export interface ReviewAccumulation {
  approved_count:number; unused_training_count:number; trained_current_revision_count:number;
  exported_current_revision_count:number; validation_only_count:number; family_count:number;
  families:Record<string,number>; situations:Record<string,number>; labels:Record<string,number>; sources:Record<string,number>;
}
export interface ReviewPage { items: ReviewSample[]; total: number; page: number; page_size: number; situations: string[]; counts: Record<'CHO_DUYET' | 'DA_DUYET',number>; accumulation?:ReviewAccumulation }
export interface ReviewExport { id: string; filename: string; sha256: string; created_at: string; sample_count: number; local_path: string }
const jsonPost = (body: object): RequestInit => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
export const sampleFromFeedback = (id: string) => request<ReviewSample>(`/supplementary/from-feedback/${encodeURIComponent(id)}`,{method:'POST'});
export const getReviewSamples = (page=1,status='',label='',situation='') => request<ReviewPage>('/supplementary/samples?'+new URLSearchParams({page:String(page),...(status?{status}:{}),...(label?{label}:{}),...(situation?{situation}:{})}));
export const getReviewSample = (id: string) => request<ReviewSample>(`/supplementary/samples/${encodeURIComponent(id)}`);
export const createReviewSample = (content: ReviewContent, source_type: 'AI_GENERATED' | 'USER_ENTERED',source_sample_id?: string) => request<ReviewSample>('/supplementary/samples',jsonPost({...content,source_type,source_sample_id}));
export const editReviewSample = (sample: ReviewSample, content: ReviewContent, editor: string) => request<ReviewSample>(`/supplementary/samples/${sample.id}`,{...jsonPost({...content,expected_revision:sample.revision,editor}),method:'PUT'});
export const approveReviewSample = (sample: ReviewSample,reviewer: string) => request<ReviewSample>(`/supplementary/samples/${sample.id}/approve`,jsonPost({expected_revision:sample.revision,reviewer,confirm_text:true}));
export const getReviewExports = () => request<ReviewExport[]>('/supplementary/exports');
export const makeReviewExport = () => request<ReviewExport>('/supplementary/exports',{method:'POST'});
export async function downloadReviewExport(record: ReviewExport) {
  let response: Response;
  try { response=await fetch(base+`/supplementary/exports/${record.id}/download`); }
  catch { throw new Error('Không kết nối được backend để tải CSV.'); }
  if (!response.ok) throw new Error('Không tải được bản CSV đã lưu.');
  const url=URL.createObjectURL(await response.blob()),link=document.createElement('a');
  link.href=url;link.download=record.filename;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export async function downloadSession(id: string) {
  const report = await request<SessionDetail>(`/sessions/${encodeURIComponent(id)}/export`);
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = `session_${id}.json`;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function visibleUnits(units: Unit[], showAll: boolean) { return units.filter(u => showAll || u.final_label !== 'KHONG_CANH_BAO').sort((a,b) => a.start - b.start || a.end - b.end); }
export function timestamp(seconds: number) {
  const ms=Math.max(0,Math.round(seconds*1000)),whole=Math.floor(ms/1000),fraction=ms%1000;
  return `${String(Math.floor(whole/60)).padStart(2,'0')}:${String(whole%60).padStart(2,'0')}${fraction ? '.'+String(fraction).padStart(3,'0') : ''}`;
}
export function overviewExplanation(text: string) {
  return text;
}
