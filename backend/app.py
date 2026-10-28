import logging
import tempfile
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.exc import SQLAlchemyError
from backend.database import Database
from backend import history

from backend.pipeline_service import InvalidAudioError, PipelineService
from backend.schemas import AudioResponse, FeedbackInput, ReviewCreate, ReviewUpdate, ReviewApproval
from backend import review_data
from backend.live_audio import ResourceGate, monitor

MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {'.mp3', '.m4a', '.wav', '.mp4', '.webm'}
logger = logging.getLogger(__name__)


def create_app(service_factory=PipelineService, database_url=None):
    @asynccontextmanager
    async def lifespan(app):
        database = Database(database_url)
        service = service_factory()
        app.state.pipeline = service
        app.state.database = database
        try:
            database.migrate()
            database.interrupt_processing()
            service.start()
            yield
        finally:
            service.close()
            database.close()

    app = FastAPI(title='Livestream audio analysis', lifespan=lifespan)
    app.state.audio_gate = ResourceGate()
    app.add_middleware(CORSMiddleware,
                       allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                       allow_methods=['GET', 'POST', 'PUT'], allow_headers=['Content-Type'])

    @app.get('/health')
    def health():
        return dict(**app.state.pipeline.health(), database=app.state.database.health())

    @app.websocket('/ws/monitor')
    async def live_microphone(websocket: WebSocket):
        await monitor(websocket, app.state.pipeline, app.state.database, app.state.audio_gate)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, error):
        logger.error('Database operation failed', exc_info=(type(error), error, error.__traceback__))
        return JSONResponse(status_code=500, content={'detail': 'Không thể lưu hoặc đọc dữ liệu. Vui lòng thử lại sau.'})

    @app.get('/sessions')
    def sessions(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100),
                 status: Literal['processing','completed','failed'] | None = None, warnings_only: bool = False):
        return history.list_sessions(app.state.database, page, page_size, status, warnings_only)

    @app.get('/sessions/{session_id}')
    def session(session_id: str):
        return history.session_detail(app.state.database, session_id)

    @app.get('/sessions/{session_id}/export')
    def export(session_id: str):
        detail = history.session_detail(app.state.database, session_id)
        return Response(json.dumps(detail, ensure_ascii=False, indent=2), media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="session_{detail["id"]}.json"'})

    @app.post('/segments/{segment_id}/feedback', status_code=201)
    def feedback(segment_id: str, payload: FeedbackInput):
        return history.add_feedback(app.state.database, segment_id, payload)

    @app.get('/supplementary/samples')
    def samples(page: int = Query(1,ge=1), page_size: int = Query(10,ge=1,le=100),
                status: Literal['CHO_DUYET','DA_DUYET'] | None = None,
                label: Literal['KHONG_CANH_BAO','CANH_BAO'] | None = None,
                situation: str | None = Query(None,max_length=200)):
        return review_data.list_samples(app.state.database,page,page_size,status,label,situation)

    @app.post('/supplementary/from-feedback/{feedback_id}')
    def sample_from_feedback(feedback_id: str):
        return review_data.from_feedback(app.state.database,feedback_id)

    @app.post('/supplementary/samples', status_code=201)
    def create_sample(payload: ReviewCreate):
        return review_data.create(app.state.database,payload)

    @app.get('/supplementary/samples/{sample_id}')
    def sample_detail(sample_id: str):
        return review_data.detail(app.state.database,sample_id)

    @app.put('/supplementary/samples/{sample_id}')
    def edit_sample(sample_id: str,payload: ReviewUpdate):
        return review_data.edit(app.state.database,sample_id,payload)

    @app.post('/supplementary/samples/{sample_id}/approve')
    def approve_sample(sample_id: str,payload: ReviewApproval):
        return review_data.approve(app.state.database,sample_id,payload)

    @app.get('/supplementary/exports')
    def exports():
        return review_data.list_exports(app.state.database)

    @app.post('/supplementary/exports')
    def create_export():
        return review_data.make_export(app.state.database)

    @app.get('/supplementary/exports/{export_id}/download')
    def download_export(export_id: str):
        content,filename = review_data.get_export(app.state.database,export_id)
        return Response(content,media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition':f'attachment; filename="{filename}"'})

    @app.post('/analyze-audio', response_model=AudioResponse)
    def analyze_audio(file: UploadFile | None = File(default=None),
                      source: Literal['upload','recording'] = Form(default='upload')):
        if file is None:
            raise HTTPException(400, 'Thiếu file âm thanh.')
        if not app.state.audio_gate.claim():
            file.file.close()
            raise HTTPException(409, 'Tài nguyên đang bận với phiên giám sát hoặc phân tích khác.')
        started = time.perf_counter()
        session_id = None
        failure_message = 'Không thể xử lý âm thanh. Vui lòng kiểm tra file hoặc thử lại.'
        try:
            filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(400, 'Chỉ hỗ trợ .mp3, .m4a, .wav, .mp4 và .webm.')
            health = app.state.pipeline.health()
            if not health['ready']:
                raise HTTPException(503, 'Model chưa sẵn sàng.')
            session_id = history.start_session(app.state.database, filename, source, health)
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
                response = AudioResponse(session_id=session_id, filename=filename,
                                         processing_time_seconds=time.perf_counter() - started, **result)
                if response.classification_mode != 'binary':
                    raise RuntimeError('History requires a binary model')
                history.complete_session(app.state.database, session_id, response)
                return response
        except InvalidAudioError:
            failure_message = 'File không chứa âm thanh hợp lệ hoặc đã bị hỏng.'
            raise HTTPException(400, 'File không chứa âm thanh hợp lệ hoặc đã bị hỏng.') from None
        except HTTPException as error:
            failure_message = str(error.detail)
            raise
        except SQLAlchemyError:
            raise
        except Exception:
            logger.exception('Audio pipeline failed')
            raise HTTPException(500, 'Không thể xử lý âm thanh. Vui lòng kiểm tra file hoặc thử lại.') from None
        finally:
            app.state.audio_gate.release()
            file.file.close()
            if session_id is not None:
                try:
                    from backend.database import AnalysisSession
                    with app.state.database.sessions() as db:
                        pending = db.get(AnalysisSession, session_id).status == 'processing'
                    if pending:
                        history.fail_session(app.state.database, session_id, failure_message,
                                             time.perf_counter() - started)
                except SQLAlchemyError:
                    logger.exception('Could not persist failed session')

    return app


app = create_app()
