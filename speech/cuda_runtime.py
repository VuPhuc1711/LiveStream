"""Small, explicit CUDA runtime checks without importing torch at module load.

The Windows workaround selects the classic CUDA addmm implementation. It does
not change model weights, numeric precision, decoding options or device.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType


ADDMM_LT_ENV = 'DISABLE_ADDMM_CUDA_LT'
RUNTIME_POLICY = 'windows_cuda_fp16_classic_addmm_v1'
SELF_CHECK_REVISION = 'worker_encoder_query_v1'


def configure_cuda_fp16_runtime(device, precision, *, environ=None, platform=None):
    """Set the process-local default before the first CUDA addmm operation.

    PyTorch caches this variable on its first addmm call. Callers must configure
    it before model warm-up/inference, and restart the process to change it.
    An explicit 0 or 1 is respected; an invalid value is never silently ignored.
    The returned mapping is an immutable startup snapshot, not a live env view.
    """
    environment = os.environ if environ is None else environ
    platform_name = sys.platform if platform is None else platform
    device_type = getattr(device, 'type', str(device).split(':', 1)[0])
    applicable = (str(platform_name).lower().startswith('win')
                  and device_type == 'cuda' and precision == 'fp16')
    explicit = ADDMM_LT_ENV in environment
    flag = environment.get(ADDMM_LT_ENV)
    if flag is not None and flag not in {'0', '1'}:
        raise ValueError(f'{ADDMM_LT_ENV} phải là 0 hoặc 1; cần sửa cấu hình rồi khởi động lại backend.')
    if applicable and not explicit:
        environment[ADDMM_LT_ENV] = '1'
        flag = '1'
    return MappingProxyType(dict(
        policy=RUNTIME_POLICY,
        platform=str(platform_name),
        device_type=str(device_type),
        precision=str(precision),
        applicable=applicable,
        environment_variable=ADDMM_LT_ENV,
        environment_value=flag,
        explicit_override=explicit,
        # Containers may explicitly request the same flag on Linux. Automatic
        # defaulting remains Windows-only; report the explicit CUDA/FP16 request
        # independently of that platform policy.
        classic_addmm_requested=device_type == 'cuda' and precision == 'fp16' and flag == '1',
        requires_process_restart_to_change=True,
    ))


def cuda_linear_worker_check(model, torch_module, runtime_snapshot,
                             *, executor_factory=ThreadPoolExecutor):
    """Run one real encoder query projection in a newly created worker.

    Use exactly the encoder context shape and its loaded parameter device/dtype.
    This is a bounded, single projection, not an audio decode or transcription.
    The executor is fully joined on both success and failure. No timeout is
    advertised: a Python timeout cannot safely abort a native CUDA operation.
    The torch module is injected so importing this module allocates no GPU state.
    """
    flag = runtime_snapshot.get('environment_value')

    def check_in_worker():
        started = time.perf_counter()
        query = model.encoder.blocks[0].attn.query
        weight = query.weight
        if weight.device.type != 'cuda' or weight.dtype != torch_module.float16:
            raise ValueError('Tự kiểm tra này yêu cầu trọng số Whisper thật trên CUDA/FP16.')
        context = model.dims.n_audio_ctx
        state = model.dims.n_audio_state
        if not isinstance(context, int) or not isinstance(state, int) or context <= 0 or state <= 0:
            raise ValueError('Kích thước encoder Whisper không hợp lệ.')
        if context * state > 16_000_000:
            raise ValueError('Kích thước encoder vượt giới hạn tự kiểm tra CUDA.')
        input_shape = (1, context, state)
        with torch_module.cuda.device(weight.device), torch_module.inference_mode():
            inputs = torch_module.zeros(input_shape, device=weight.device, dtype=weight.dtype)
            output = query(inputs)
            if tuple(output.shape) != input_shape:
                raise RuntimeError('Phép chiếu query Whisper trả kích thước không đúng.')
            if not bool(torch_module.isfinite(output).all().item()):
                raise RuntimeError('Phép chiếu query Whisper trả NaN hoặc vô cực.')
            torch_module.cuda.synchronize(weight.device)
        return dict(
            revision=SELF_CHECK_REVISION,
            status='passed',
            operation='encoder.blocks[0].attn.query',
            input_shape=input_shape,
            device=str(weight.device),
            dtype=str(weight.dtype),
            thread_id=threading.get_ident(),
            thread_name=threading.current_thread().name,
            elapsed_seconds=round(time.perf_counter() - started, 6),
            environment_variable=ADDMM_LT_ENV,
            environment_value=flag,
            policy=runtime_snapshot.get('policy'),
            audio_or_decode_used=False,
        )

    try:
        with executor_factory(max_workers=1, thread_name_prefix='asr-cuda-selfcheck') as executor:
            result = executor.submit(check_in_worker).result()
    except Exception as error:
        raise RuntimeError(
            'Whisper CUDA/FP16 không vượt qua tự kiểm tra phép chiếu encoder trong worker '
            f'({ADDMM_LT_ENV}={flag!r}): {type(error).__name__}: {error}. '
            'Backend chưa sẵn sàng; không tự chuyển CPU hoặc model khác.'
        ) from error
    return MappingProxyType(result)
