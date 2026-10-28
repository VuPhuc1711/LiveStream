"""Readiness must include actual worker CUDA validation, not only model loading."""
import unittest
from unittest.mock import Mock, patch
from backend.model_config import ROOT
from backend.pipeline_service import PipelineService

class StartupRuntimeTests(unittest.TestCase):
    def start(self,fail=False):
        service=PipelineService()
        config=dict(version='binary_v04',checkpoint='models/fixture/best',
            classification_mode='binary',classifier='phobert_only',backend_device='cpu')
        detector=Mock();detector.phobert_detector.checkpoint_path=ROOT/config['checkpoint']
        detector.phobert_detector.device='cpu'
        with patch('backend.pipeline_service.check_environment',return_value={'cuda_available':True}), \
             patch('backend.pipeline_service.get_active_model',return_value=config), \
             patch('backend.pipeline_service.verify_checkpoint'), \
             patch('backend.pipeline_service.create_detector',return_value=detector), \
             patch('backend.neural_vad.NeuralVad'), \
             patch('speech.asr_config.load_asr_config',return_value=dict(model='turbo',device='cuda',precision='fp16',
                  initial_prompt='',language='vi',task='transcribe',download_root=None)), \
             patch('speech.transcribe.WhisperTranscriber') as transcriber:
            loaded=transcriber.return_value
            loaded.runtime_fault=None
            loaded.configuration.return_value={'model':'turbo','runtime_self_check':{'status':'passed'}}
            def validate():
                self.assertIs(service.detector,detector)
                self.assertEqual(service.model_version,'binary_v04')
                if fail:raise RuntimeError('CUBLAS_STATUS_NOT_SUPPORTED')
            loaded.verify_runtime.side_effect=validate
            if fail:
                with self.assertRaisesRegex(RuntimeError,'CUBLAS'):service.start()
                loaded.configuration.assert_not_called()
                self.assertFalse(service.health()['ready'])
                self.assertEqual(service.health()['whisper'],'not_ready')
            else:
                service.start();loaded.verify_runtime.assert_called_once()
                self.assertTrue(service.health()['ready'])
                loaded.runtime_fault={'status':'failed','category':'cuda_execution_error'}
                self.assertFalse(service.health()['ready'])
                self.assertEqual(service.health()['whisper'],'not_ready')
                self.assertEqual(service.asr_configuration['runtime_self_check']['status'],'passed')
        return service

    def test_validation_runs_after_both_models_loaded_and_fault_affects_health(self):self.start()
    def test_failed_validation_prevents_ready(self):self.start(fail=True)
