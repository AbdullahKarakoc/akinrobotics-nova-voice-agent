from __future__ import annotations

from app.config import Settings
from app.llm.base import ChatLLM, Done, LLMEvent, LLMStats, TextDelta, ToolCall, ToolCalls


def create_llm(settings: Settings) -> ChatLLM:
    if settings.llm_backend == "ollama":
        from app.llm.ollama_llm import OllamaLLM

        return OllamaLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            num_ctx=settings.llm_num_ctx,
            num_thread=settings.llm_num_thread,
            keep_alive=settings.llm_keep_alive,
            timeout_s=settings.llm_timeout_s,
            think=settings.llm_think,
        )
    if settings.llm_backend == "openai":
        from app.llm.openai_llm import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")


__all__ = ["ChatLLM", "Done", "LLMEvent", "LLMStats", "TextDelta", "ToolCall", "ToolCalls", "create_llm"]
