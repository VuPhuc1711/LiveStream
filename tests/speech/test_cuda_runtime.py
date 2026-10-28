import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from speech.cuda_runtime import (ADDMM_LT_ENV, configure_cuda_fp16_runtime,
                                 cuda_linear_worker_check)


class FakeDevice:
    def __init__(self, device_type='cuda'):
        self.type = device_type

    def __str__(self):
        return f'{self.type}:0'


class CudaRuntimeTests(unittest.TestCase):
    def test_windows_cuda_fp16_sets_process_default_and_freezes_snapshot(self):
        environment = {}
        snapshot = configure_cuda_fp16_runtime(FakeDevice(), 'fp16',
            environ=environment, platform='win32')
        self.assertEqual(environment, {ADDMM_LT_ENV: '1'})
        self.assertTrue(snapshot['classic_addmm_requested'])
        self.assertFalse(snapshot['explicit_override'])
        environment[ADDMM_LT_ENV] = '0'
        self.assertEqual(snapshot['environment_value'], '1')
        with self.assertRaises(TypeError):
            snapshot['environment_value'] = '0'

    def test_explicit_zero_and_one_are_both_respected(self):
        for flag in ('0', '1'):
            with self.subTest(flag=flag):
                environment = {ADDMM_LT_ENV: flag}
                snapshot = configure_cuda_fp16_runtime('cuda:0', 'fp16',
                    environ=environment, platform='win32')
                self.assertEqual(environment[ADDMM_LT_ENV], flag)
                self.assertTrue(snapshot['explicit_override'])
                self.assertEqual(snapshot['classic_addmm_requested'], flag == '1')

    def test_linux_explicit_flag_is_reported_without_changing_default_policy(self):
        for flag in ('0', '1'):
            with self.subTest(flag=flag):
                environment = {ADDMM_LT_ENV: flag}
                snapshot = configure_cuda_fp16_runtime('cuda:0', 'fp16',
                    environ=environment, platform='linux')
                self.assertEqual(environment, {ADDMM_LT_ENV: flag})
                self.assertTrue(snapshot['explicit_override'])
                self.assertFalse(snapshot['applicable'])
                self.assertEqual(snapshot['classic_addmm_requested'], flag == '1')

    def test_invalid_flag_rejected_and_unrelated_devices_untouched(self):
        for flag in ('', 'true', '2'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                configure_cuda_fp16_runtime('cuda', 'fp16',
                    environ={ADDMM_LT_ENV: flag}, platform='win32')
        for platform, device, precision in (('linux', 'cuda', 'fp16'),
                ('win32', 'cpu', 'fp32'), ('win32', 'cuda', 'fp32')):
            with self.subTest(platform=platform, device=device, precision=precision):
                environment = {}
                snapshot = configure_cuda_fp16_runtime(device, precision,
                    environ=environment, platform=platform)
                self.assertEqual(environment, {})
                self.assertFalse(snapshot['applicable'])
                self.assertFalse(snapshot['classic_addmm_requested'])

    def make_model_and_torch(self):
        device = FakeDevice()
        dtype = 'mock_float16'
        query = Mock(return_value=SimpleNamespace(shape=(1, 1500, 1280)))
        query.weight = SimpleNamespace(device=device, dtype=dtype)
        model = SimpleNamespace(dims=SimpleNamespace(n_audio_ctx=1500, n_audio_state=1280),
            encoder=SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(query=query))]),
            transcribe=Mock())
        cuda = SimpleNamespace(device=Mock(side_effect=lambda value: nullcontext()),
                               synchronize=Mock())
        finite = Mock()
        finite.all.return_value.item.return_value = True
        torch = SimpleNamespace(float16=dtype, cuda=cuda,
            inference_mode=Mock(side_effect=nullcontext), zeros=Mock(return_value=object()),
            isfinite=Mock(return_value=finite))
        return model, torch

    def track_executors(self):
        pools = []

        def factory(**kwargs):
            pool = ThreadPoolExecutor(**kwargs)
            pools.append(pool)
            return pool

        return pools, factory

    def test_check_uses_actual_query_shape_dtype_device_in_joined_worker(self):
        model, torch = self.make_model_and_torch()
        policy = configure_cuda_fp16_runtime('cuda', 'fp16', environ={}, platform='win32')
        pools, factory = self.track_executors()
        caller_thread = threading.get_ident()
        snapshot = cuda_linear_worker_check(model, torch, policy, executor_factory=factory)
        query = model.encoder.blocks[0].attn.query
        torch.zeros.assert_called_once_with((1, 1500, 1280),
            device=query.weight.device, dtype=query.weight.dtype)
        query.assert_called_once_with(torch.zeros.return_value)
        torch.cuda.synchronize.assert_called_once_with(query.weight.device)
        model.transcribe.assert_not_called()
        self.assertEqual(snapshot['status'], 'passed')
        self.assertEqual(snapshot['environment_value'], '1')
        self.assertNotEqual(snapshot['thread_id'], caller_thread)
        self.assertTrue(snapshot['thread_name'].startswith('asr-cuda-selfcheck'))
        self.assertTrue(all(not thread.is_alive() for pool in pools for thread in pool._threads))
        with self.assertRaises(TypeError):
            snapshot['status'] = 'failed'

    def test_cuda_failure_is_visible_without_fallback_and_executor_closes(self):
        model, torch = self.make_model_and_torch()
        query = model.encoder.blocks[0].attn.query
        query.side_effect = RuntimeError('CUBLAS_STATUS_NOT_SUPPORTED')
        pools, factory = self.track_executors()
        with self.assertRaisesRegex(RuntimeError, 'CUBLAS_STATUS_NOT_SUPPORTED'):
            cuda_linear_worker_check(model, torch, {'environment_value': '1'},
                                     executor_factory=factory)
        self.assertEqual(query.call_count, 1)
        model.transcribe.assert_not_called()
        self.assertTrue(all(not thread.is_alive() for pool in pools for thread in pool._threads))

    def test_nonfinite_or_wrong_device_never_reports_ready(self):
        model, torch = self.make_model_and_torch()
        torch.isfinite.return_value.all.return_value.item.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'NaN'):
            cuda_linear_worker_check(model, torch, {'environment_value': '1'})
        model, torch = self.make_model_and_torch()
        model.encoder.blocks[0].attn.query.weight.device = FakeDevice('cpu')
        with self.assertRaisesRegex(RuntimeError, 'CUDA/FP16'):
            cuda_linear_worker_check(model, torch, {'environment_value': '1'})
        torch.zeros.assert_not_called()


if __name__ == '__main__':
    unittest.main()
