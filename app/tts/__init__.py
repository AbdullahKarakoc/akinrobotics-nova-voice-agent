from __future__ import annotations

from app.config import Settings
from app.tts.base import SpeechAudio, TextToSpeech


def create_tts(settings: Settings) -> TextToSpeech:
    """Factory: add a new engine by adding a branch here (it only has to implement `synthesize`)."""
    if settings.tts_backend == "piper":
        from app.tts.piper_tts import PiperTTS

        return PiperTTS(
            voices={lang: settings.voice_for(lang) for lang in settings.languages},
            voices_dir=settings.models_dir / "piper",
            length_scale=settings.tts_length_scale,
        )
    raise ValueError(f"Unknown TTS backend: {settings.tts_backend}")


__all__ = ["SpeechAudio", "TextToSpeech", "create_tts"]
