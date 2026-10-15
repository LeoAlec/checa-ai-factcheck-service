"""Create a clearly labelled third-pass LLM review of a golden-set sample."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.evaluation.metrics import CONCLUSIONS  # noqa: E402

SYSTEM_PROMPT = """\
Você é um anotador independente de relações entre alegação e evidência.
Use SOMENTE o laudo fornecido, sem conhecimento externo, e escolha:
- sustentada: o laudo confirma a mesma alegação;
- contradita: o laudo refuta a mesma alegação;
- inconclusiva: entidade, número, data ou fato diferem, ou a evidência é insuficiente.
Na dúvida, escolha inconclusiva. Responda somente JSON com
{"conclusion":"sustentada|contradita|inconclusiva","reason":"justificativa curta"}.
"""


def parse_review(raw: str) -> tuple[str, str]:
    payload = json.loads(raw)
    conclusion = payload.get("conclusion")
    reason = str(payload.get("reason", "")).strip()
    if conclusion not in CONCLUSIONS or not reason:
        raise ValueError("revisão fora do contrato")
    return conclusion, reason


def review_item(client: Any, item: dict[str, Any], model: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ALEGAÇÃO:\n{item['claim']}\n\n"
                    f"LAUDO:\n{item['evidence']['titulo']}\n{item['evidence']['texto']}"
                ),
            },
        ],
    )
    return parse_review(response.choices[0].message.content or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden_claims.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "golden_claims_llm_review.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "evals" / "golden_claims_llm_review_report.json")
    parser.add_argument("--model", default=os.environ.get("GOLDEN_REVIEW_MODEL", "gpt-4.1"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY ausente.", file=sys.stderr)
        return 2
    import openai

    items = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = random.Random(args.seed).sample(items, min(args.sample_size, len(items)))
    client = openai.OpenAI(timeout=90, max_retries=3)
    rows: list[dict[str, Any]] = []
    for item in sample:
        conclusion, reason = review_item(client, item, args.model)
        rows.append({
            "id": item["id"],
            "claim": item["claim"],
            "evidence": item["evidence"]["texto"],
            "gold_conclusion": item["gold_conclusion"],
            "reviewer_conclusion": conclusion,
            "agrees_with_gold": conclusion == item["gold_conclusion"],
            "review_reason": reason,
            "reviewer_type": "llm",
            "reviewer_model": args.model,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agreements = sum(bool(row["agrees_with_gold"]) for row in rows)
    report = {
        "reviewer_type": "llm",
        "reviewer_model": args.model,
        "sample_size": len(rows),
        "agreements": agreements,
        "agreement_rate": agreements / len(rows) if rows else None,
        "seed": args.seed,
        "not_human_validation": True,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
