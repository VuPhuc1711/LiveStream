import gc
import hashlib
import os
import shutil
from pathlib import Path

import torch
import whisper
from speech.cuda_runtime import configure_cuda_fp16_runtime, cuda_linear_worker_check


def _whisper_without_parameter_storage(dims):
    """Use native encoder/decoder classes without allocating random weights.

    Native Whisper.__init__ cannot run wholly on meta because its sparse
    alignment-head construction has no Meta kernel. Only its module assembly
    is repeated here; all layers remain the upstream Whisper implementations.
    """
    model = whisper.Whisper.__new__(whisper.Whisper)
    torch.nn.Module.__init__(model)
    model.dims = dims
    with torch.device('meta'):
        model.encoder = whisper.model.AudioEncoder(dims.n_mels, dims.n_audio_ctx,
            dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
        model.decoder = whisper.model.TextDecoder(dims.n_vocab, dims.n_text_ctx,
            dims.n_text_state, dims.n_text_head, dims.n_text_layer)
    # This nonpersistent buffer is absent from model_state_dict and therefore
    # must be materialized explicitly. Alignment heads are restored separately.
    mask = torch.empty(dims.n_text_ctx, dims.n_text_ctx, dtype=torch.float32,
        device='cpu').fill_(-float('inf')).triu_(1)
    model.decoder.register_buffer('mask', mask, persistent=False)
    return model


def load_cached_half_precision(model_name, download_root=None):
    """Map verified official weights without a second full FP32 model in RAM.

    Whisper's normal loader constructs FP32 parameters beside the checkpoint.
    On small Windows pagefiles that can fail before CUDA staging starts. Keep
    native Whisper layers, state validation and alignment heads, but allocate
    no temporary parameters and assign the memory-mapped checkpoint directly.
    No download, model substitution or unchecked checkpoint fallback is made.
    """
    if model_name not in whisper._MODELS or model_name not in whisper._ALIGNMENT_HEADS:
        raise ValueError('Bộ nạp FP16 cần tên checkpoint Whisper chính thức.')
    url = whisper._MODELS[model_name]
    if download_root is None:
        default_cache = Path.home() / '.cache'
        download_root = Path(os.environ.get('XDG_CACHE_HOME', default_cache)) / 'whisper'
    path = Path(download_root).expanduser().resolve() / url.rsplit('/', 1)[-1]
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy checkpoint Whisper đã cache: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != url.rsplit('/', 2)[-2]:
        raise RuntimeError('Checkpoint Whisper cache sai SHA256; không nạp hoặc tự đổi model.')
    checkpoint = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
    dims = whisper.ModelDimensions(**checkpoint['dims'])
    model = _whisper_without_parameter_storage(dims)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True, assign=True)
    model.set_alignment_heads(whisper._ALIGNMENT_HEADS[model_name])
    if any(value.is_meta for value in model.parameters()) or any(value.is_meta for value in model.buffers()):
        raise RuntimeError('Whisper còn tensor meta chưa nạp; không khởi động suy luận.')
    return model


def stage_half_precision(model):
    model.half()
    # Whisper LayerNorm.forward deliberately computes in FP32, including its weights.
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()


class WhisperTranscriber:
    def __init__(self, model_name="small", device=None, initial_prompt='', precision=None,
                 language='vi', task='transcribe', download_root=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        half_weights = precision == 'fp16'
        self.precision = precision or ('fp16' if self.device.type == 'cuda' else 'fp32')
        if self.precision not in {'fp16', 'fp32'} or (self.device.type == 'cpu' and self.precision != 'fp32'):
            raise ValueError('Whisper CPU cần fp32; precision phải là fp32 hoặc fp16.')
        if model_name.endswith('.en') or language != 'vi' or task != 'transcribe':
            raise ValueError('Luồng ASR dùng Whisper đa ngôn ngữ, vi/transcribe.')
        # PyTorch reads this process-local flag on its first CUDA addmm call.
        # Select the supported classic CUDA path before loading/warming models.
        self.cuda_runtime_policy = dict(configure_cuda_fp16_runtime(self.device, self.precision))
        self.runtime_self_check = {'status': 'not_run'}
        self.runtime_fault = None
        self.model_name = model_name
        # Explicitly pin existing defaults; only initial_prompt varies in A/B.
        self.decode_options = dict(language=language, task=task, fp16=self.precision == 'fp16',
            temperature=(0., .2, .4, .6, .8, 1.), compression_ratio_threshold=2.4,
            logprob_threshold=-1., no_speech_threshold=.6, condition_on_previous_text=True,
            initial_prompt=initial_prompt or None, carry_initial_prompt=False,
            beam_size=None, best_of=None, word_timestamps=False, verbose=False)
        # Diagnostic comparison only. Real audio trials failed to recover voice;
        # the production pipeline never promotes this aligned retry automatically.
        self.retry_decode_options = dict(self.decode_options, temperature=0., beam_size=5,
            condition_on_previous_text=False, word_timestamps=True)
        # Controlled full-window trials recovered both observed prompt-induced
        # failures with this single change. Keep native timestamp decoding;
        # word alignment/partitioning failed and are diagnostic only.
        self.recovery_decode_options = dict(self.decode_options, initial_prompt=None)
        print(f"Loading Whisper model: {model_name} ({self.device}, {self.precision})", flush=True)
        # Stage FP16 weights on CPU to avoid checkpoint + FP32 model coexisting on 4 GiB CUDA.
        # Preserve legacy CLI weight dtype when precision was not explicitly configured.
        self.weights_precision = 'fp16_except_layernorm' if half_weights else 'fp32'
        low_memory_load = half_weights and self.device.type == 'cuda'
        self.loading_strategy = 'verified_cached_mmap_fp16_assign_v1' if low_memory_load else 'official_whisper_loader'
        if low_memory_load:
            self.model = load_cached_half_precision(model_name, download_root)
        else:
            load_device = 'cpu' if half_weights else self.device
            options = {'download_root': download_root} if download_root is not None else {}
            self.model = whisper.load_model(model_name, device=load_device, **options)
        if half_weights:
            stage_half_precision(self.model)
        self.model.to(self.device)
        self.model.eval()

    def configuration(self):
        return dict(model=self.model_name, device=str(self.device), precision=self.precision,
                    weights_precision=self.weights_precision,
                    loading_strategy=self.loading_strategy,
                    sample_rate=16000, channels=1, decoding=dict(self.decode_options),
                    layer_norm_precision='fp32',
                    context_scope='within_one_transcribe_call_only',
                    cuda_runtime=dict(self.cuda_runtime_policy),
                    runtime_self_check=dict(self.runtime_self_check),
                    recovery=dict(revision='full_window_no_prompt_v1',
                        decoding=dict(self.recovery_decode_options),
                        context_scope='same_audio_without_textual_prior'),
                    diagnostic_retry=dict(revision='bounded_timestamp_retry_trial_v1',
                        enabled_in_pipeline=False,
                        decoding=dict(self.retry_decode_options),
                        context_scope='independent_retry_same_audio_and_model'))

    def transcribe(self, audio_path):
        # Một lần gọi cho toàn bộ file; giữ text, start/end và segment gốc.
        return self._transcribe_with_options(audio_path, self.decode_options)

    def verify_runtime(self):
        """Validate the loaded GPU projection in a worker before reporting ready."""
        if self.device.type == 'cuda' and self.precision == 'fp16':
            self.runtime_self_check = dict(cuda_linear_worker_check(
                self.model, torch, self.cuda_runtime_policy))
        else:
            self.runtime_self_check = {'status': 'not_applicable', 'device': str(self.device)}
        return dict(self.runtime_self_check)

    def transcribe_retry(self, audio_path):
        """Diagnostic only: alignment can fit hallucinated text to real audio.

        No production caller should use valid retry timestamps as evidence that
        the transcript is correct. Preserve this trial method for reproducibility.
        """
        return self._transcribe_with_options(audio_path, self.retry_decode_options)

    def transcribe_without_prompt(self, audio_path):
        """One same-model full-window attempt; no alignment or transcript prompt."""
        return self._transcribe_with_options(audio_path, self.recovery_decode_options)

    def _transcribe_with_options(self, audio_path, decode_options):
        if self.model is None:
            raise RuntimeError("Whisper đã được giải phóng; không thể transcribe tiếp.")
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {audio_path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Không tìm thấy FFmpeg trong PATH. Kiểm tra bằng: ffmpeg -version")
        try:
            with torch.inference_mode():
                result = self.model.transcribe(str(audio_path), **decode_options)
            self.runtime_fault = None
            return result
        except RuntimeError as error:
            if self.device.type == 'cuda' and any(word in str(error).upper() for word in ('CUDA', 'CUBLAS')):
                self.runtime_fault = {'status': 'failed', 'category': 'cuda_execution_error',
                                      'requires_backend_restart': True}
            raise

    def close(self):
        """Giải phóng Whisper trước khi nạp PhoBERT trên GPU ít VRAM."""
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    transcriber = WhisperTranscriber("small")

    result = transcriber.transcribe("audio/voicetiengviet.mp3")

    print("\n===== TRANSCRIPT =====")
    print(result["text"])
