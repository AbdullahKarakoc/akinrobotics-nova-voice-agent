"""Agent orchestration with a scripted LLM (no model or server needed)."""

import json

from app.agent import Agent, AgentTurn, Conversation
from app.config import Settings
from app.knowledge.documents import Chunk
from app.knowledge.retriever import Hit
from app.llm import Done, LLMStats, TextDelta, ToolCall, ToolCalls
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    model = "scripted"

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def stream_chat(self, messages, tools=None):
        self.requests.append({"messages": messages, "tools": tools})
        for event in self.rounds.pop(0):
            yield event
        yield Done(LLMStats(prompt_tokens=10, completion_tokens=5, prompt_eval_ms=1, eval_ms=1))

    async def aclose(self):
        pass


class FakeRetriever:
    def __init__(self, hits):
        self.chunks = [h.chunk for h in hits]
        self._hits = hits

    def search(self, query, top_k=3, min_score=0.0):
        return [h for h in self._hits if h.score >= min_score][:top_k]


CHUNK = Chunk(id="urun.md#2", source="urun.md", title="Rehber R2", text="Pil ömrü 8 saattir.")


def make_agent(rounds, hits=(), mode="rag"):
    registry = ToolRegistry()

    @registry.tool("get_order_status", "Order status", {"order_number": {"type": "string"}})
    def get_order_status(order_number: str) -> dict:
        return {"found": True, "status_en": "shipped", "order_number": order_number}

    settings = Settings(knowledge_mode=mode)
    return Agent(ScriptedLLM(rounds), FakeRetriever(list(hits)), registry, settings)


async def collect(agent, question, language="tr", conversation=None):
    conversation = conversation or Conversation(4)
    turn = AgentTurn()
    deltas = [d async for d in agent.respond(question, language, conversation, turn)]
    return "".join(deltas), turn, conversation


async def test_context_is_injected_and_answer_streamed():
    agent = make_agent([[TextDelta("Pil ömrü "), TextDelta("8 saattir.")]], hits=[Hit(CHUNK, 0.8, 0.8, 1.0)])
    text, turn, conversation = await collect(agent, "Rehber R2 pil ömrü?")
    assert text == "Pil ömrü 8 saattir."
    user_msg = agent.llm.requests[0]["messages"][-1]["content"]
    assert "Pil ömrü 8 saattir." in user_msg and "Türkçe cevap ver" in user_msg
    assert turn.ungrounded == [] and turn.sources()[0]["id"] == "urun.md#2"
    assert conversation.messages[-1] == {"role": "assistant", "content": "Pil ömrü 8 saattir."}


async def test_no_context_is_explicit_and_invented_numbers_are_flagged():
    agent = make_agent([[TextDelta("Pil ömrü 12 saattir.")]])
    _, turn, _ = await collect(agent, "C3 kaç kilo?")
    assert "(no relevant information found)" in agent.llm.requests[0]["messages"][-1]["content"]
    assert turn.ungrounded == ["12"]


async def test_llm_tool_call_runs_tool_speaks_filler_and_feeds_result_back():
    call = ToolCall(id="c1", name="get_order_status", arguments={"order_number": "1003"})
    agent = make_agent([[ToolCalls([call])], [TextDelta("Siparişiniz yolda.")]])
    text, turn, _ = await collect(agent, "siparişim nerede, numarası bin üç")
    assert text == "Bir saniye, kontrol ediyorum. Siparişiniz yolda."
    second = agent.llm.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and json.loads(second[-1]["content"])["status_en"] == "shipped"
    assert turn.answer == "Siparişiniz yolda."  # filler is spoken but not stored in history


async def test_tool_call_written_as_text_is_executed_not_spoken():
    leaked = ',{"name": "get_order_status", "arguments": {"order_number": "1003"}}'
    agent = make_agent([[TextDelta(leaked[:20]), TextDelta(leaked[20:])], [TextDelta("Siparişiniz yolda.")]])
    text, turn, _ = await collect(agent, "siparişim nerede")
    assert "{" not in text and text.endswith("Siparişiniz yolda.")
    assert [r.call.name for r in turn.tool_results] == ["get_order_status"]


async def test_text_that_only_looks_like_json_is_still_spoken():
    agent = make_agent([[TextDelta("[Not] bilgi yok.")]])
    text, _, _ = await collect(agent, "x")
    assert text == "[Not] bilgi yok."


async def test_full_knowledge_mode_puts_corpus_in_cached_system_prompt():
    agent = make_agent([[TextDelta("8 saat.")], [TextDelta("Bilmiyorum.")]], hits=[Hit(CHUNK, 0.8, 0.8, 1.0)], mode="full")
    await collect(agent, "pil?")
    messages = agent.llm.requests[0]["messages"]
    assert "[K1] Pil ömrü 8 saattir." in messages[0]["content"]
    user = messages[-1]["content"]
    assert "CONTEXT" not in user and "RELEVANT KNOWLEDGE: K1" in user

    agent.retriever._hits = []  # nothing relevant -> explicit "none" signal instead of passages
    await collect(agent, "genel müdür kim?")
    assert "RELEVANT KNOWLEDGE: none" in agent.llm.requests[1]["messages"][-1]["content"]


async def test_llm_failure_yields_spoken_apology():
    class Broken(ScriptedLLM):
        async def stream_chat(self, messages, tools=None):
            raise RuntimeError("server down")
            yield  # pragma: no cover

    agent = make_agent([])
    agent.llm = Broken([])
    text, turn, _ = await collect(agent, "merhaba", language="en")
    assert text.startswith("Sorry") and turn.error
