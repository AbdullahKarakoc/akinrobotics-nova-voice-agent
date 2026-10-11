"""Grounding / hallucination evaluation of the agent (text in, text out).

    python -m scripts.evaluate [--url http://localhost:8000] [--out reports/eval.json]

For every question in eval/questions.json (fresh conversation each):
  * expect  - at least one of the strings must appear in the answer (accuracy)
  * refuse  - the answer is NOT in the knowledge sources: the agent must say it
              doesn't know instead of inventing an answer (hallucination test);
              optional "forbid" lists strings a refusal must not contain (e.g. the answer itself)
  * tool    - the named tool must have been called
  * lang    - the answer must be in the same language as the question
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.speech_text import detect_text_language  # noqa: E402
from app.knowledge.text import turkish_lower  # noqa: E402

REFUSAL_MARKERS = [
    "bilgi", "maalesef", "üzgünüm", "bilmiyorum", "yardımcı olabil", "sahip değilim", "bulunmuyor", "yalnızca",
    "sadece", "don't have", "do not have", "sorry", "only help", "can only", "unable", "no information",
    "not available", "don't know", "cannot provide", "unrelated",
]  # fmt: skip


def contains(answer: str, needles: list[str]) -> bool:
    a = turkish_lower(answer)
    return any(turkish_lower(n) in a for n in needles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    client = httpx.Client(base_url=args.url, timeout=180)
    health = client.get("/health").json()
    print(f"model: {health['llm_model']}  cases: {len(cases)}\n")

    rows = []
    for case in cases:
        start = time.perf_counter()
        r = client.post("/api/chat", json={"text": case["q"], "session_id": uuid.uuid4().hex}).json()
        wall = (time.perf_counter() - start) * 1000
        answer = r["answer"]
        tools = [s["id"] for s in r["sources"] if s["type"] == "tool"]
        checks = {}
        if "expect" in case:
            checks["fact"] = contains(answer, case["expect"])
        if case.get("refuse"):
            checks["refusal"] = (contains(answer, REFUSAL_MARKERS) and not r["ungrounded_numbers"]
                                 and not contains(answer, case.get("forbid", [])))  # fmt: skip
        if "tool" in case:
            checks["tool"] = case["tool"] in tools
        checks["language"] = detect_text_language(answer, case["lang"]) == case["lang"]
        ok = all(checks.values())
        rows.append({**case, "answer": answer, "tools": tools, "checks": checks, "ok": ok, "ms": round(wall),
                     "ungrounded_numbers": r["ungrounded_numbers"], "metrics": r["metrics"]})  # fmt: skip
        mark = "PASS" if ok else "FAIL"
        failed = ",".join(k for k, v in checks.items() if not v)
        print(f"[{mark}] {wall/1000:5.1f}s  {case['q']}\n        -> {answer}" + (f"\n        failed: {failed}" if failed else ""))

    def rate(key: str) -> str:
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    lat = [r["ms"] for r in rows]
    ttft = [r["metrics"]["llm_ttft_ms"] for r in rows if r["metrics"].get("llm_ttft_ms")]
    summary = {
        "model": health["llm_model"],
        "passed": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
        "fact_accuracy": rate("fact"),
        "correct_refusals": rate("refusal"),
        "tool_selection": rate("tool"),
        "answer_language": rate("language"),
        "answers_with_ungrounded_numbers": sum(bool(r["ungrounded_numbers"]) for r in rows),
        "latency_ms_p50": round(statistics.median(lat)),
        "llm_ttft_ms_p50": round(statistics.median(ttft)) if ttft else None,
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
