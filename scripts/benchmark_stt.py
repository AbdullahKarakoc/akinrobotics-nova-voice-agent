"""STT accuracy (WER) and speed (RTF) on real human speech: Google FLEURS test set (CC-BY 4.0).

    python -m scripts.benchmark_stt [--n 25] [--models small base] [--out reports/stt.json]

Compares, for each Whisper size:
  stock    - faster-whisper's standard transcribe() with automatic language ID
             (30 s padded window; the encoder runs twice when the language is unknown)
  single30 - this project's decoder: one encoder pass shared by language ID and decoding, 30 s window
  dynamic  - this project's default: one encoder pass over only the utterance + 2 s

The first N utterances of the TR and EN test splits are streamed from the
Hugging Face hub (the archives are not downloaded completely) and cached in
MODELS_DIR/fleurs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import sys
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.text import turkish_lower  # noqa: E402
from app.stt.faster_whisper_stt import FasterWhisperSTT  # noqa: E402

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}/{path}"
LANGS = {"tr": "tr_tr", "en": "en_us"}


def fetch_fleurs(lang: str, n: int, cache: Path) -> list[tuple[np.ndarray, str]]:
    """Return n (audio float32 16 kHz, reference transcript) pairs."""
    folder = cache / LANGS[lang]
    folder.mkdir(parents=True, exist_ok=True)
    tsv = folder / "test.tsv"
    if not tsv.exists():
        tsv.write_bytes(httpx.get(FLEURS.format(lang=LANGS[lang], path="test.tsv"), follow_redirects=True, timeout=60).content)
    refs = {row[1]: row[2] for row in csv.reader(tsv.open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}

    have = sorted(folder.glob("*.wav"))
    if len(have) < n:
        url = FLEURS.format(lang=LANGS[lang], path="audio/test.tar.gz")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            stream = _ResponseReader(resp.iter_bytes())
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if not member.isfile() or not name.endswith(".wav") or name not in refs:
                        continue
                    (folder / name).write_bytes(tar.extractfile(member).read())
                    if len(list(folder.glob("*.wav"))) >= n:
                        break
        have = sorted(folder.glob("*.wav"))

    # FLEURS wavs are 32-bit float, which the stdlib `wave` module can't read.
    return [(decode_audio(str(path), sampling_rate=16000), refs[path.name]) for path in have[:n]]


class _ResponseReader(io.RawIOBase):
    """File-like wrapper so tarfile can read an HTTP stream sequentially."""

    def __init__(self, chunks):
        self.chunks, self.buf = chunks, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", turkish_lower(text)).split()


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="utterances per language")
    parser.add_argument("--models", nargs="+", default=["small", "base"])
    parser.add_argument("--variants", nargs="+", default=["stock", "single30", "dynamic"],
                        help="subset of: stock single30 dynamic (stock is slow for bigger models)")  # fmt: skip
    parser.add_argument("--langs", nargs="+", default=list(LANGS))
    parser.add_argument("--threads", type=int, default=None, help="default: STT_CPU_THREADS")
    parser.add_argument("--beam", type=int, default=None, help="default: STT_BEAM_SIZE")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    settings = get_settings()

    data = {lang: fetch_fleurs(lang, args.n, settings.models_dir / "fleurs") for lang in args.langs}
    durations = [len(a) / 16000 for pairs in data.values() for a, _ in pairs]
    print(f"utterances: {sum(map(len, data.values()))}, mean length {statistics.fmean(durations):.1f} s\n")

    results = []
    for size in args.models:
        stt = FasterWhisperSTT(model=size, cpu_threads=args.threads or settings.stt_cpu_threads,
                               beam_size=args.beam or settings.stt_beam_size, hotwords=settings.stt_hotwords,
                               repetition_penalty=settings.stt_repetition_penalty,
                               download_root=settings.models_dir / "whisper")  # fmt: skip
        stt.transcribe(next(iter(data.values()))[0][0], ["tr", "en"])  # warm-up

        def stock(audio, langs):
            start = time.perf_counter()
            segments, info = stt.whisper.transcribe(audio, beam_size=1, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments)
            return text, info.language, time.perf_counter() - start

        def ours(dynamic):
            def run(audio, langs):
                stt.dynamic_window = dynamic
                t = stt.transcribe(audio, langs)
                return t.text, t.language, t.processing_s

            return run

        variants = {"stock": stock, "single30": ours(False), "dynamic": ours(True)}
        for variant, fn in ((v, variants[v]) for v in args.variants):
            row = {"model": size, "variant": variant}
            for lang, pairs in data.items():
                errors = words = correct_lang = 0
                proc = audio_s = 0.0
                for audio, ref in pairs:
                    text, detected, seconds = fn(audio, ["tr", "en"])
                    e, w = wer(ref, text)
                    errors, words = errors + e, words + w
                    correct_lang += detected == lang
                    proc, audio_s = proc + seconds, audio_s + len(audio) / 16000
                row[lang] = {
                    "wer": round(errors / words, 3),
                    "lang_id": f"{correct_lang}/{len(pairs)}",
                    "rtf": round(proc / audio_s, 3),
                    "ms_per_utt": round(proc / len(pairs) * 1000),
                }
            results.append(row)
            print(f"{size:6} {variant:9} " + "  ".join(
                f"{lang}: WER {row[lang]['wer']:.1%} LID {row[lang]['lang_id']} RTF {row[lang]['rtf']:.2f} ({row[lang]['ms_per_utt']} ms)"
                for lang in data))  # fmt: skip

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"n_per_language": args.n, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
