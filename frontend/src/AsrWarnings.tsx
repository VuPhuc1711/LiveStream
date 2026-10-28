import { timestamp, type AsrQualityWarning } from './api';

export default function AsrWarnings({ warnings = [], unresolvedCount, recoveredCount = 0 }: {
  warnings?: AsrQualityWarning[]; unresolvedCount?: number; recoveredCount?: number;
}) {
  const unresolved = Math.max(unresolvedCount ?? 0, warnings.length);
  if (!unresolved && !recoveredCount) return null;
  return <aside className="asr-quality" aria-label="Chất lượng nhận dạng" role="status">
    {unresolved > 0 && <><strong>{unresolved} đoạn chưa nhận dạng đáng tin cậy</strong>
      <p>Các đoạn này không được đưa vào phân loại. Chưa xác định được nội dung; không có nghĩa là không cảnh báo.</p>
      {warnings.length > 0 && <ul>{warnings.map(w => <li key={w.window_id}>
        <span>{timestamp(w.start)} – {timestamp(w.end)}</span>
        {w.retried ? ' · Đã thử nhận dạng lại, vẫn cần kiểm tra.' : ' · Cần kiểm tra lại audio.'}
      </li>)}</ul>}
      <p>Nếu cần đối chiếu, thu một phiên mới có bật lưu audio debug rồi nghe lại đoạn tương ứng.</p>
    </>}
    {recoveredCount > 0 && <p>ASR đã nhận dạng lại {recoveredCount} đoạn; có thể vẫn nhận sai từ.</p>}
  </aside>;
}
