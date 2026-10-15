"""
paraphrase_benchmark.py
─────────────────────────────────────────────────────────────────────
Reescreve as consultas `matched` do benchmark no estilo de mensagem de
WhatsApp, para que a consulta NÃO repita o título do laudo indexado
(evita vazamento lexical que favorece BM25 e infla o recall).

Também atribui um split estratificado e determinístico (calibration/test)
por status, para calibrar o limiar em uma metade e reportar na outra.

Uso:
    python scripts/paraphrase_benchmark.py
Exige OPENAI_API_KEY. Casos `abstained` não são alterados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DEFAULT_IN = ROOT / "data" / "eval" / "retrieval_benchmark_120.jsonl"
DEFAULT_OUT = ROOT / "data" / "eval" / "retrieval_benchmark_120_paraphrased.jsonl"

STYLES = [
    "informal, com abreviações (vc, tbm, pq) e tom de indignação",
    "em CAIXA ALTA com pontuação exagerada, como corrente de WhatsApp",
    "como pergunta de alguém em dúvida pedindo confirmação",
    "como relato de terceiro ('meu tio mandou no grupo que...')",
]

SYSTEM = """\
Você reescreve alegações como circulariam no WhatsApp em português do Brasil.
Regras:
- Preserve o MESMO fato, entidades e números da alegação original.
- NÃO copie a frase original: troque vocabulário e estrutura; evite repetir sequências de 3+ palavras.
- Não adicione fatos novos nem desminta a alegação; ela deve soar como o boato circulando.
- Remova marcas de checagem ('#boato', 'É falso que', 'É #FAKE que').
- Uma a duas frases. Responda SOMENTE JSON: {"text": str}
"""


def split_for(case_id: str) -> str:
    digest = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16)
    return "calibration" if digest % 2 == 0 else "test"


def assign_balanced_splits(cases: list[dict]) -> None:
    """Split 50/50 estratificado por status, ordenado por hash (determinístico)."""
    for status in {c["expected_status"] for c in cases}:
        group = sorted(
            (c for c in cases if c["expected_status"] == status),
            key=lambda c: hashlib.sha256(c["id"].encode("utf-8")).hexdigest(),
        )
        for i, case in enumerate(group):
            case["split"] = "calibration" if i % 2 == 0 else "test"


def paraphrase(client, model: str, text: str, style: str) -> str | None:
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Estilo: {style}\nAlegação original: {text}"},
        ],
    )
    try:
        out = json.loads(response.choices[0].message.content)["text"].strip()
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI()
    cases = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    assign_balanced_splits(cases)

    for i, case in enumerate(c for c in cases if c["expected_status"] == "matched"):
        style = STYLES[i % len(STYLES)]
        new = paraphrase(client, args.model, case["query"], style)
        case["original_query"] = case["query"]
        case["paraphrase_style"] = style
        if new:
            case["query"] = new
        else:
            case["paraphrase_failed"] = True
        print(f"[{i + 1:02d}] {case['original_query'][:60]!r}\n     -> {case['query'][:90]!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
