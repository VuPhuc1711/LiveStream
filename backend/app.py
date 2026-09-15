import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline_service import InvalidAudioError, PipelineService
from backend.schemas import AudioResponse

MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {'.mp3', '.m4a', '.wav', '.mp4'}
logger = logging.getLogger(__name__)


def create_app(service_factory=PipelineService):
    @asynccontextmanager
    async def lifespan(app):
        service = service_factory()
        app.state.pipeline = service
        try:
            service.start()
            yield
        finally:
            service.close()

    app = FastAPI(title='Livestream audio analysis', lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
                       allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @app.get('/health')
    def health():
        return app.state.pipeline.health()

    @app.post('/analyze-audio', response_model=AudioResponse)
    def analyze_audio(file: UploadFile | None = File(default=None)):
        if file is None:
            raise HTTPException(400, 'Thiếu file âm thanh.')
        started = time.perf_counter()
        try:
            filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(400, 'Chỉ hỗ trợ .mp3, .m4a, .wav và .mp4.')
            # The client filename is never used as a filesystem path.
            with tempfile.TemporaryDirectory(prefix='livestream_audio_') as directory:
                path = Path(directory) / ('upload' + suffix)
                size = 0
                with path.open('wb') as destination:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise HTTPException(413, 'File vượt giới hạn 50 MiB.')
                        destination.write(chunk)
                if not size:
                    raise HTTPException(400, 'File âm thanh rỗng.')
                result = app.state.pipeline.analyze(path)
                return AudioResponse(filename=filename,
                                     processing_time_seconds=time.perf_counter() - started, **result)
        except InvalidAudioError:
            raise HTTPException(400, 'File không chứa âm thanh hợp lệ hoặc đã bị hỏng.') from None
        except HTTPException:
            raise
        except Exception:
            logger.exception('Audio pipeline failed')
            raise HTTPException(500, 'Không thể xử lý âm thanh. Vui lòng kiểm tra file hoặc thử lại.') from None
        finally:
            file.file.close()

    return app


app = create_app()
