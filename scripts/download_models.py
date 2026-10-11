"""Download every local model the agent needs into MODELS_DIR (idempotent).

    python -m scripts.download_models

The LLM is served by Ollama and pulled separately (`ollama pull <model>`).
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("download")


def piper_voice(name: str, target_dir: Path) -> None:
    onnx = target_dir / f"{name}.onnx"
    if onnx.exists() and onnx.with_suffix(".onnx.json").exists():
        return
    locale, speaker, quality = name.split("-")
    prefix = f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}"
    staging = target_dir / ".download"
    for suffix in (".onnx", ".onnx.json"):
        path = hf_hub_download("rhasspy/piper-voices", f"{prefix}{suffix}", local_dir=staging)
        shutil.move(path, target_dir / f"{name}{suffix}")
    shutil.rmtree(staging, ignore_errors=True)
    log.info("piper voice ready", extra={"data": {"voice": name}})


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    start = time.perf_counter()
    models = settings.models_dir

    for language in settings.languages:
        piper_voice(settings.voice_for(language), models / "piper")

    from faster_whisper.utils import download_model

    download_model(settings.stt_model, cache_dir=str(models / "whisper"))
    log.info("whisper ready", extra={"data": {"model": settings.stt_model}})

    from fastembed import TextEmbedding

    TextEmbedding(model_name=settings.embedding_model, cache_dir=str(models / "embeddings"))
    log.info("embedding model ready", extra={"data": {"model": settings.embedding_model}})

    log.info("all models ready", extra={"data": {"seconds": round(time.perf_counter() - start, 1)}})


if __name__ == "__main__":
    main()
