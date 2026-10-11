"""Whisper (CTranslate2 / faster-whisper) speech-to-text tuned for short, live utterances.

Whisper was trained on 30 s windows and the stock pipeline always pads audio to
30 s, then runs the encoder twice when the language is unknown (once for
language ID, once for decoding). For a 2-3 s voice command on a 4-core laptop
CPU that is ~10 s with `small`. Here we instead:

1. run the encoder ONCE on a window that is only as long as the utterance plus
   2 s of padding (the padding lets the decoder see where speech ends), and
   reuse that encoder output for both language ID and decoding (~10x faster);
2. restrict language ID to the supported languages;
3. guard against the failure modes of shortened windows and of Whisper in
   general: a token budget proportional to the audio length, repetition
   trimming, a no-speech check and a list of known silence hallucinations.

On real speech (FLEURS, see scripts/benchmark_stt.py) this keeps the WER of the
stock pipeline while being ~4x faster on 10 s utterances and more on short ones.

Set STT_DYNAMIC_WINDOW=false to fall back to full 30 s windows.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer

from app.stt.base import Transcript

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FULL_WINDOW_FRAMES = 3000  # 30 s of 10 ms mel frames
PAD_FRAMES = 200  # 2 s

# Phrases Whisper is known to hallucinate on silence/noise (subtitle credits seen in training data).
_HALLUCINATIONS = re.compile(
    r"^(altyaz[ıi].*|thank(s| you) for watching.*|abone ol.*|izlediğiniz için teşekkürler.*|"
    r"subtitles by.*|www\..*|\[?müzik\]?|\[?music\]?)$",
    re.IGNORECASE,
)
_REPEAT = re.compile(r"(\S.{2,}?)(?:[\s,.!?]+\1){2,}", re.IGNORECASE)
_LONG_REPEAT = re.compile(r"(\S.{11,}?)[\s,.!?]+\1", re.IGNORECASE)


def trim_repetitions(text: str) -> str:
    """Remove decoder loops.

    'Merhaba Merhaba Merhaba' -> 'Merhaba' (short phrase, 3+ times), and
    'Merhaba, sesim geliyor mu? Merhaba, sesim geliyor mu? Merhaba, ses' -> 'Merhaba, sesim geliyor mu?'
    (a 12+ character phrase said twice in a row never happens in a short voice command; whatever
    follows the repetition is the loop's tail and is dropped too)."""
    text = _REPEAT.sub(r"\1", text).strip()
    if m := _LONG_REPEAT.search(text):
        end = m.end(1)
        while end < len(text) and text[end] in ".!?…":  # keep the phrase's own punctuation
            end += 1
        text = text[:end]
    return text.strip()


class FasterWhisperSTT:
    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 1,
        hotwords: str | None = None,
        dynamic_window: bool = True,
        repetition_penalty: float = 1.0,
        download_root: Path | None = None,
    ):
        self.beam_size = beam_size
        self.repetition_penalty = repetition_penalty
        self.hotwords = hotwords or ""
        self.dynamic_window = dynamic_window
        self.whisper = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=str(download_root) if download_root else None,
        )
        self._tokenizers: dict[str, Tokenizer] = {}
        log.info(
            "faster-whisper loaded",
            extra={"data": {"model": model, "compute_type": compute_type, "dynamic_window": dynamic_window}},
        )

    def _tokenizer(self, language: str) -> Tokenizer:
        if language not in self._tokenizers:
            self._tokenizers[language] = Tokenizer(
                self.whisper.hf_tokenizer, multilingual=True, task="transcribe", language=language
            )
        return self._tokenizers[language]

    def _prompt(self, tok: Tokenizer) -> list[int]:
        prompt: list[int] = []
        if self.hotwords:  # domain vocabulary as "previous text" biases spelling of product names
            prompt = [tok.sot_prev, *tok.encode(" " + self.hotwords.strip())[-200:]]
        return prompt + list(tok.sot_sequence) + [tok.no_timestamps]

    def transcribe(self, audio: np.ndarray, languages: list[str]) -> Transcript:
        start = time.perf_counter()
        audio_s = len(audio) / SAMPLE_RATE
        model = self.whisper.model

        features = self.whisper.feature_extractor(audio)
        frames = FULL_WINDOW_FRAMES
        if self.dynamic_window:
            frames = min(FULL_WINDOW_FRAMES, int(features.shape[-1]) + PAD_FRAMES)
        window = np.ascontiguousarray(pad_or_trim(features, frames)[None], dtype=np.float32)
        encoder_output = model.encode(ctranslate2.StorageView.from_array(window), to_cpu=False)

        probs = {token[2:-2]: p for token, p in model.detect_language(encoder_output)[0]}
        candidates = languages or list(probs)
        language = max(candidates, key=lambda code: probs.get(code, 0.0))

        tok = self._tokenizer(language)
        prompt = self._prompt(tok)
        max_new_tokens = int(10 + 12 * audio_s)  # speech rarely exceeds ~8 tokens/s
        result = model.generate(
            encoder_output,
            [prompt],
            beam_size=self.beam_size,
            max_length=min(448, len(prompt) + max_new_tokens),
            repetition_penalty=self.repetition_penalty,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=True,
        )[0]
        tokens = [t for t in result.sequences_ids[0] if t < tok.eot]
        text = trim_repetitions(tok.decode(tokens).strip())

        if result.no_speech_prob > 0.6 and result.scores[0] < -1.0:
            log.info("dropped: no speech", extra={"data": {"text": text, "p": round(result.no_speech_prob, 2)}})
            text = ""
        stripped = text.strip(" .!,?")
        if _HALLUCINATIONS.match(stripped) or (stripped and stripped.lower() in self.hotwords.lower()):
            log.info("dropped likely STT hallucination", extra={"data": {"text": text}})
            text = ""

        return Transcript(
            text=text,
            language=language,
            language_prob=round(probs.get(language, 0.0), 3),
            audio_s=audio_s,
            processing_s=time.perf_counter() - start,
        )
