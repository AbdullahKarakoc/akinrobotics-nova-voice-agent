"""Logging with per-session / per-turn context.

`session_id` and `turn_id` are stored in context variables so that every log
line emitted while handling a turn carries them automatically, without passing
loggers around.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        record.turn_id = turn_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "session": record.session_id,
            "turn": record.turn_id,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extra = getattr(record, "data", None)
        if extra:
            line += " " + json.dumps(extra, ensure_ascii=False)
        return line


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            _TextFormatter(
                "%(asctime)s %(levelname)-5s [%(session_id)s/%(turn_id)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "faster_whisper", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
