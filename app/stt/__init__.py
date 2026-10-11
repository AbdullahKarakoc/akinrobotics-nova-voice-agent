from __future__ import annotations

from app.config import Settings
from app.stt.base import SpeechToText, Transcript


def create_stt(settings: Settings) -> SpeechToText:
    """Factory: add a new engine by adding a branch here (it only has to implement `transcribe`)."""
    if settings.stt_backend == "faster_whisper":
        from app.stt.faster_whisper_stt import FasterWhisperSTT

        return FasterWhisperSTT(
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            cpu_threads=settings.stt_cpu_threads,
            beam_size=settings.stt_beam_size,
            hotwords=settings.stt_hotwords,
            dynamic_window=settings.stt_dynamic_window,
            repetition_penalty=settings.stt_repetition_penalty,
            download_root=settings.models_dir / "whisper",
        )
    raise ValueError(f"Unknown STT backend: {settings.stt_backend}")


__all__ = ["SpeechToText", "Transcript", "create_stt"]
