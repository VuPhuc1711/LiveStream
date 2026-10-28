# syntax=docker/dockerfile:1
# Native Windows venvs and model weights are never copied into this image.
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    HF_HOME=/app/runtime/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

WORKDIR /app
COPY docker/requirements-backend.lock.txt /opt/requirements-backend.lock.txt
COPY docker/requirements-vad.txt /opt/requirements-vad.txt
# Whisper has an sdist only. Its pure-Python wheel is built in this isolated
# image; the pinned native torch/preprocessing package versions are retained.
RUN python -m pip install --require-hashes -r /opt/requirements-backend.lock.txt \
    && python -m pip check \
    && python -m pip install --no-deps -r /opt/requirements-vad.txt \
    && python -c "from importlib.metadata import distribution; from pathlib import Path; d=distribution('silero-vad'); assert d.version=='6.2.1'; assert Path(d.locate_file('silero_vad/data/silero_vad.jit')).is_file()"

COPY --chown=app:app backend /app/backend
COPY --chown=app:app speech /app/speech
COPY --chown=app:app nlp /app/nlp
COPY --chown=app:app tests /app/tests
COPY --chown=app:app tools /app/tools
COPY --chown=app:app alembic.ini asr_config.json pipeline_model.json /app/
COPY --chown=app:app config/runtime_assets.json /app/config/runtime_assets.json
COPY --chown=app:app config/data/current_dataset.json /app/config/data/current_dataset.json
COPY --chown=app:app data/current /app/data/current
COPY --chown=app:app docker/backend-entrypoint.py /usr/local/bin/livestream-backend.py
RUN mkdir -p /app/runtime /app/models /app/config/data /app/data/current \
    && chown -R app:app /app/runtime /app/models /app/config /app/data

USER app
EXPOSE 8000
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "/usr/local/bin/livestream-backend.py"]
CMD ["serve"]
