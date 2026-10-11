"""Piper (VITS, ONNX) text-to-speech: fast on CPU, one voice per language."""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

from app.tts.base import SpeechAudio

log = logging.getLogger(__name__)


def _ascii_safe_espeak_dir() -> Path:
    """espeak-ng (C library inside Piper) cannot open non-ASCII paths on Windows, e.g. a project
    living in "C:/Users/x/Desktop/akınRobotics". Use the 8.3 short path or a temp copy instead."""
    import piper

    data_dir = Path(piper.__file__).parent / "espeak-ng-data"
    if str(data_dir).isascii() or sys.platform != "win32":
        return data_dir
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(str(data_dir), buf, len(buf)) and buf.value.isascii():
        return Path(buf.value)
    copy = Path(tempfile.gettempdir()) / "piper-espeak-ng-data"
    if not copy.exists():
        shutil.copytree(data_dir, copy)
    return copy


class PiperTTS:
    def __init__(self, voices: dict[str, str], voices_dir: Path, length_scale: float = 1.0):
        """`voices` maps a language code to a Piper voice name, e.g. {"tr": "tr_TR-dfki-medium"}."""
        self.voices: dict[str, PiperVoice] = {}
        espeak_dir = _ascii_safe_espeak_dir()
        for language, name in voices.items():
            path = voices_dir / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"Piper voice not found: {path} (run scripts/download_models.py)")
            self.voices[language] = PiperVoice.load(path, espeak_data_dir=espeak_dir)
            log.info("piper voice loaded", extra={"data": {"language": language, "voice": name}})
        self.fallback = next(iter(self.voices))
        self.config = SynthesisConfig(length_scale=length_scale)

    def synthesize(self, text: str, language: str) -> SpeechAudio:
        start = time.perf_counter()
        voice = self.voices.get(language) or self.voices[self.fallback]
        chunks = [c.audio_int16_array for c in voice.synthesize(text, syn_config=self.config)]
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return SpeechAudio(
            pcm16=pcm.astype(np.int16).tobytes(),
            sample_rate=voice.config.sample_rate,
            processing_s=time.perf_counter() - start,
        )
