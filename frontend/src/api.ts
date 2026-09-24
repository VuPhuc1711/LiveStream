export type Label = 'KHONG_CANH_BAO' | 'CAN_XAC_MINH' | 'CANH_BAO';
export const labels: Record<Label, string> = { KHONG_CANH_BAO: 'Không cảnh báo', CAN_XAC_MINH: 'Cần xác minh', CANH_BAO: 'Cảnh báo' };
export interface Unit { unit_id: string; source_segment_id: string | number; text: string; start: number; end: number; timestamp_is_approximate: boolean; final_label: Label; final_confidence: number; phobert_result: { label: Label; confidence: number; probabilities: Record<Label, number> }; rule_result: Record<string, unknown>; explanation: string }
export interface Result { filename: string; processing_time_seconds: number; transcript: string; summary: { classification_unit_count: number; whisper_segment_count: number; final_label_counts: Record<Label, number> }; classification_units: Unit[] }
export interface Health { status: string; ready: boolean; checkpoint: string; whisper: string; phobert: string }
const base = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
export function validateFile(file: File): string | null {
  if (!/\.(mp3|m4a|wav|mp4)$/i.test(file.name)) return 'Chỉ hỗ trợ file MP3, M4A, WAV hoặc MP4.';
  if (file.size === 0) return 'File rỗng. Vui lòng chọn file khác.';
  if (file.size > 50 * 1024 * 1024) return 'File vượt giới hạn 50 MiB.';
  return null;
}
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(base + path, options); }
  catch { throw new Error('Không kết nối được backend. Kiểm tra server và kết nối rồi thử lại.'); }
  if (!response.ok) {
    const messages: Record<number, string> = { 400: 'File không hợp lệ, rỗng hoặc không đọc được âm thanh.', 413: 'File vượt giới hạn 50 MiB.', 422: 'Dữ liệu gửi lên không hợp lệ.' };
    throw new Error(messages[response.status] || 'Backend xử lý thất bại. Vui lòng thử lại sau.');
  }
  try { return await response.json() as T; }
  catch { throw new Error('Backend trả về dữ liệu không hợp lệ.'); }
}
export const getHealth = () => request<Health>('/health', { signal: AbortSignal.timeout(8000) });
export function analyze(file: File) { const form = new FormData(); form.append('file', file); return request<Result>('/analyze-audio', { method: 'POST', body: form }); }
export function visibleUnits(units: Unit[], showAll: boolean) { return units.filter(u => showAll || u.final_label !== 'KHONG_CANH_BAO').sort((a,b) => a.start - b.start || a.end - b.end); }
export function timestamp(seconds: number) { const whole = Math.floor(seconds); return `${String(Math.floor(whole/60)).padStart(2,'0')}:${String(whole%60).padStart(2,'0')}`; }
// Full backend explanation remains in details; keep numeric confidence out of the overview.
export function overviewExplanation(text: string) {
  return text.replace(/^PhoBERT dự đoán .*?confidence .*?%\.\s*/, '')
    .replace(/^Rules không khớp cụm từ nào\.\s*/, '')
    .replace(/^Rules khớp các cụm từ: .*?\.\s*/, '')
    .replace(/\s*Nhãn cuối được chuyển theo chính sách; final_confidence[\s\S]*$/, '');
}
