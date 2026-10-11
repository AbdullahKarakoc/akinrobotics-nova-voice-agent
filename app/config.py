"""Central configuration.

Every tunable lives here and can be overridden with an environment variable
(or a `.env` file) of the same name in upper case, e.g. `LLM_MODEL=qwen2.5:1.5b`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False

    models_dir: Path = ROOT_DIR / "models"
    data_dir: Path = ROOT_DIR / "data"
    web_dir: Path = ROOT_DIR / "web"

    # Languages the agent understands and answers in. The first one is the fallback.
    languages: list[str] = Field(default_factory=lambda: ["tr", "en"])

    # --- VAD / turn-taking --------------------------------------------------
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500  # silence that ends a user turn
    vad_min_speech_ms: int = 200  # shorter blips are ignored
    vad_speech_pad_ms: int = 300  # audio kept before speech onset
    vad_max_utterance_s: float = 25.0
    barge_in: bool = True  # user may interrupt the agent while it speaks
    barge_in_min_speech_ms: int = 350  # stricter onset while the agent is speaking

    # --- STT --------------------------------------------------------------
    stt_backend: str = "faster_whisper"
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_cpu_threads: int = 4  # 8 was faster natively on Windows (1.25 -> 1.10 s) but slower in Docker/WSL2 (1.27 -> 1.91 s)
    stt_dynamic_window: bool = True  # encode only the utterance (+2 s), not a padded 30 s window
    stt_repetition_penalty: float = 1.0
    # Optional domain words given to Whisper as "previous text". Off by default: on real Turkish speech
    # (FLEURS) it raised WER from 21% to 36%. Product names are fixed by app/stt/domain_terms.py instead.
    stt_hotwords: str = ""

    # --- LLM --------------------------------------------------------------
    llm_backend: str = "ollama"  # "ollama" (native API) or "openai" (any OpenAI-compatible server)
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    # Explicit 4-bit (Q4_K_M) tag: ~1.3 GB instead of ~3.4 GB in fp16, and much faster on CPU.
    # (Ollama's plain "qwen3:1.7b" points to the same Q4_K_M file; the tag makes it explicit.)
    llm_model: str = "qwen3:1.7b-q4_K_M"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 200
    llm_num_ctx: int = 3072  # longest prompt ~1.0k tokens + 4 history turns; smaller KV cache = more layers on a 4 GB GPU
    llm_num_thread: int | None = None  # Ollama CPU threads (None = Ollama's default: physical cores)
    llm_keep_alive: str = "30m"
    llm_timeout_s: float = 90.0
    # Ollama "think" flag: false skips the reasoning phase of hybrid models such as qwen3
    # (other models ignore it). None = don't send the field.
    llm_think: bool | None = False
    max_history_turns: int = 4
    max_tool_rounds: int = 2

    # --- knowledge ----------------------------------------------------------
    # "rag": retrieved chunks go into each prompt (scales to any corpus size).
    # "full": the whole knowledge base sits in the cached system prompt (lowest latency for small corpora).
    knowledge_mode: str = "rag"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = 2
    # Hybrid score below this is "not relevant". Calibrated on eval/: knowledge questions
    # score >= 0.62, out-of-domain ones <= 0.43.
    rag_min_score: float = 0.45
    rag_relative_cut: float = 0.15  # also drop chunks this far below the best one
    rag_chunk_chars: int = 600

    # --- TTS --------------------------------------------------------------
    tts_backend: str = "piper"
    tts_voice_tr: str = "tr_TR-dfki-medium"
    tts_voice_en: str = "en_US-ryan-medium"
    tts_length_scale: float = 1.0  # < 1 speaks faster

    # --- tools ------------------------------------------------------------
    db_path: Path = ROOT_DIR / "data" / "nova.db"
    http_tool_timeout_s: float = 5.0
    timezone: str = "Europe/Istanbul"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def default_language(self) -> str:
        return self.languages[0]

    def voice_for(self, language: str) -> str:
        return {"tr": self.tts_voice_tr, "en": self.tts_voice_en}.get(language, self.tts_voice_tr)


@lru_cache
def get_settings() -> Settings:
    return Settings()
