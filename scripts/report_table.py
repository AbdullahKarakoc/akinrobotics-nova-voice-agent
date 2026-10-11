"""Print a Markdown comparison table of reports/eval_*.json (for the README).

    python -m scripts.report_table
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = []
    for path in sorted((ROOT / "reports").glob("eval_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["summary"]
        mode = path.stem.rsplit("_", 1)[-1]
        rows.append((s, mode))

    print("| LLM | Bilgi modu | Geçen | Olgu doğruluğu | Doğru ret | Araç seçimi | Cevap dili | Kaynaksız sayı içeren cevap | LLM ilk token p50 | Toplam cevap p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, mode in rows:
        print(
            f"| {s['model']} | {mode} | {s['passed']} | {s['fact_accuracy']} | {s['correct_refusals']} | "
            f"{s['tool_selection']} | {s['answer_language']} | {s['answers_with_ungrounded_numbers']} | "
            f"{s['llm_ttft_ms_p50'] / 1000:.1f} sn | {s['latency_ms_p50'] / 1000:.1f} sn |"
        )


if __name__ == "__main__":
    main()
