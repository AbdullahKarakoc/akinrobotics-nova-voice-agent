FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_VERBOSITY=error \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    MODELS_DIR=/models

WORKDIR /app

# libgomp: OpenMP runtime used by CTranslate2 / ONNX Runtime; curl: health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY web ./web
COPY data ./data
COPY eval ./eval
COPY scripts ./scripts

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=20 \
    CMD curl -fs http://localhost:8000/health || exit 1

# Models (Whisper, Piper voices, embeddings) are downloaded on first start into
# the /models volume, so the image stays small and later starts are instant.
CMD ["sh", "-c", "python -m scripts.download_models && exec python -m app"]
