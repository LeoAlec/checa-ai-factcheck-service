"""
build_golden_claims.py
─────────────────────────────────────────────────────────────────────
Constrói o golden set de 100 alegações em português com conclusão evidencial
esperada (`sustentada` | `contradita` | `inconclusiva`) e um laudo de
referência, usando DUAS passagens de LLM:

  1. GERAÇÃO: um modelo escreve alegações no estilo WhatsApp + o laudo que as
     sustenta/contradiz/não resolve, por (rótulo, fenômeno linguístico).
  2. VERIFICAÇÃO CEGA: outro modelo (idealmente diferente) lê só alegação+laudo,
     sem ver o rótulo pretendido, e decide a conclusão. Só entram no golden os
     itens em que gerador e juiz concordam; o resto é descartado e regenerado.

Limites (documentados no relatório): é um conjunto SINTÉTICO verificado por LLM,
não por humanos — mede consistência alegação↔laudo (tipo NLI), não verdade no
mundo. Os laudos são fictícios (`*.example`). Use `--sample-for-humans` para
exportar uma amostra a revisar manualmente e estimar o erro do juiz.

Uso:
    python scripts/build_golden_claims.py            # 100 alegações
    python scripts/build_golden_claims.py --size 20 --generator-model gpt-4o --judge-model gpt-4o-mini
Exige OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.evaluation.metrics import CONCLUSIONS, cohen_kappa  # noqa: E402

DEFAULT_OUT = ROOT / "evals" / "golden_claims.jsonl"
DEFAULT_REPORT = ROOT / "evals" / "golden_claims_report.json"

# Distribuição-alvo (soma 100): inconclusiva é a classe mais difícil e mais perigosa.
DEFAULT_QUOTAS = {"contradita": 35, "sustentada": 30, "inconclusiva": 35}
TOPICS = [
    "saúde e vacinas", "eleições e urnas", "economia e impostos", "golpes e fraudes digitais",
    "meio ambiente e clima", "tecnologia e redes sociais", "segurança pública", "educação",
    "alimentação e remédios caseiros", "benefícios sociais e previdência", "energia e combustíveis",
    "ciência e espaço",
]
PHENOMENA = {
    "negation": "a alegação contém negação ou dupla negação (ex.: 'não é verdade que…')",
    "paraphrase": "a alegação parafraseia o laudo com vocabulário diferente, sem repetir seus termos",
    "caps": "mensagem em CAIXA ALTA, estilo corrente de WhatsApp, com pontuação exagerada",
    "informal": "linguagem informal com abreviações (vc, tbm, pq, msm) e emojis em texto",
    "numeric": "a alegação depende de números, valores ou datas específicos",
    "entity": "a alegação nomeia uma instituição/sigla específica (TSE, Anvisa, INSS…)",
    "plain": "alegação curta e direta",
}
LABEL_DEFS = {
    "contradita": (
        "O laudo REFUTA diretamente a alegação: trata do mesmo fato, mesma entidade, mesmos números "
        "e conclui que ela é falsa."
    ),
    "sustentada": (
        "O laudo CONFIRMA a alegação: trata do mesmo fato, mesma entidade e mesmos números e "
        "conclui que ela é verdadeira."
    ),
    "inconclusiva": (
        "O laudo NÃO permite concluir: é do mesmo assunto, mas trata de outra entidade, outro número/"
        "data/período, é parcial ou ressalva que não há dados — de modo que a alegação específica "
        "não pode ser confirmada nem refutada com ele. Varie: entidade diferente, número diferente, "
        "evidência parcial, dados insuficientes."
    ),
}

GEN_SYSTEM = """\
Você é anotador sênior de um corpus de checagem de fatos em português do Brasil.
Escreva exemplos SINTÉTICOS (fictícios) para avaliar um verificador de alegações.
Regras:
- Cada exemplo tem: "claim" (a alegação como circularia no WhatsApp), "evidence_title" e
  "evidence_text" (um laudo curto de 2 a 4 frases, tom jornalístico, com a fonte institucional
  citada no texto) e "tags" (lista de fenômenos presentes).
- O laudo deve ser a ÚNICA base de decisão. Não dependa de conhecimento externo.
- Não atribua falas ou crimes a pessoas físicas reais. Instituições públicas genéricas são aceitas.
- Variedade: assuntos e estruturas diferentes entre os exemplos, sem repetir frases.
- Responda SOMENTE JSON: {"items": [{"claim": str, "evidence_title": str, "evidence_text": str, "tags": [str]}]}
"""

JUDGE_SYSTEM = """\
Você é um verificador de alegações rigoroso. Receberá uma ALEGAÇÃO e um LAUDO.
Use SOMENTE o laudo (sem conhecimento externo) e classifique a relação:
- "sustentada": o laudo confirma a alegação (mesma entidade, números e fato).
- "contradita": o laudo refuta a alegação (mesma entidade, números e fato).
- "inconclusiva": o laudo não trata da mesma entidade/número/data/fato, é parcial ou insuficiente.
Na dúvida entre decidir e não decidir, escolha "inconclusiva".
Responda SOMENTE JSON: {"conclusion": "sustentada|contradita|inconclusiva", "reason": str}
"""


# ─── Funções puras (testáveis) ───────────────────────────────────────
def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", stripped))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def is_duplicate(claim: str, existing: list[str], threshold: float = 0.7) -> bool:
    return any(jaccard(claim, other) >= threshold for other in existing)


def build_generation_prompt(label: str, phenomenon: str, topic: str, n: int) -> str:
    return (
        f"Gere {n} exemplos DIFERENTES em que a conclusão correta é \"{label}\".\n"
        f"Definição: {LABEL_DEFS[label]}\n"
        f"Assunto: {topic}.\n"
        f"Fenômeno linguístico obrigatório em todos: {PHENOMENA[phenomenon]}.\n"
        f"Inclua \"{phenomenon}\" em tags."
    )


def parse_items(raw: str) -> list[dict[str, Any]]:
    try:
        items = json.loads(raw).get("items", [])
    except (ValueError, AttributeError):
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if all(isinstance(item.get(k), str) and item[k].strip() for k in ("claim", "evidence_title", "evidence_text")):
            valid.append(item)
    return valid


def parse_judgement(raw: str) -> str | None:
    try:
        conclusion = json.loads(raw).get("conclusion")
    except (ValueError, AttributeError):
        return None
    return conclusion if conclusion in CONCLUSIONS else None


def plan_batches(remaining: dict[str, int], rng: random.Random, batch_size: int = 5) -> list[tuple[str, str, str, int]]:
    """Lotes (rótulo, fenômeno, assunto, n) cobrindo a cota restante, com fenômenos rotativos."""
    phenomena = list(PHENOMENA)
    batches = []
    for label, missing in remaining.items():
        while missing > 0:
            n = min(batch_size, missing)
            batches.append((label, rng.choice(phenomena), rng.choice(TOPICS), n))
            missing -= n
    rng.shuffle(batches)
    return batches


# ─── Chamadas ao LLM ─────────────────────────────────────────────────
def make_chat(client, model: str, temperature: float) -> Callable[[str, str], str]:
    def chat(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return chat


def build_golden(
    generate: Callable[[str, str], str],
    judge: Callable[[str, str], str],
    quotas: dict[str, int],
    seed: int = 13,
    max_rounds: int = 8,
    batch_size: int = 5,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    kept: list[dict] = []
    stats = {"generated": 0, "duplicates": 0, "disagreements": 0, "judge_errors": 0}
    intended_all: list[str] = []
    judged_all: list[str] = []
    claims: list[str] = []

    for _round in range(max_rounds):
        counts = Counter(item["gold_conclusion"] for item in kept)
        remaining = {label: quotas[label] - counts[label] for label in quotas if quotas[label] > counts[label]}
        if not remaining:
            break
        # Pede ~30% a mais para compensar descartes do juiz.
        asked = {label: int(missing * 1.3) + 1 for label, missing in remaining.items()}
        for label, phenomenon, topic, n in plan_batches(asked, rng, batch_size):
            for item in parse_items(generate(GEN_SYSTEM, build_generation_prompt(label, phenomenon, topic, n))):
                stats["generated"] += 1
                if is_duplicate(item["claim"], claims):
                    stats["duplicates"] += 1
                    continue
                verdict = parse_judgement(
                    judge(JUDGE_SYSTEM, f"ALEGAÇÃO:\n{item['claim']}\n\nLAUDO:\n{item['evidence_title']}\n{item['evidence_text']}")
                )
                if verdict is None:
                    stats["judge_errors"] += 1
                    continue
                intended_all.append(label)
                judged_all.append(verdict)
                if verdict != label:
                    stats["disagreements"] += 1
                    continue
                if sum(1 for k in kept if k["gold_conclusion"] == label) >= quotas[label]:
                    continue
                claims.append(item["claim"])
                tags = sorted({t for t in item.get("tags", []) if isinstance(t, str)} | {phenomenon})
                kept.append(
                    {
                        "id": f"gc-{len(kept) + 1:03d}",
                        "claim": item["claim"].strip(),
                        "gold_conclusion": label,
                        "topic": topic,
                        "tags": tags,
                        "evidence": {
                            "titulo": item["evidence_title"].strip(),
                            "texto": item["evidence_text"].strip(),
                            "dominio": "laudo.example",
                            "url": f"https://laudo.example/gc-{len(kept) + 1:03d}",
                            "data_publicacao": "2024-05-10",
                        },
                        "synthetic": True,
                        "verified_by": "llm-blind-judge",
                    }
                )
    report = {
        **stats,
        "kept": len(kept),
        "by_label": dict(Counter(k["gold_conclusion"] for k in kept)),
        "quotas": quotas,
        "judge_agreement": (
            sum(1 for a, b in zip(intended_all, judged_all) if a == b) / len(intended_all) if intended_all else None
        ),
        "cohen_kappa_generator_vs_judge": cohen_kappa(intended_all, judged_all) if intended_all else None,
        "complete": all(Counter(k["gold_conclusion"] for k in kept)[l] >= q for l, q in quotas.items()),
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o golden set de alegações verificado por LLM")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--generator-model", default=os.environ.get("GOLDEN_GENERATOR_MODEL", "gpt-4o"))
    parser.add_argument("--judge-model", default=os.environ.get("GOLDEN_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-for-humans", type=int, default=20, help="n de itens para revisão humana (CSV)")
    parser.add_argument(
        "--human-review-out",
        type=Path,
        help="CSV de revisão; padrão: <nome do --out>_human_review.csv",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY ausente.", file=sys.stderr)
        return 2
    import openai

    client = openai.OpenAI(timeout=90, max_retries=3)
    total = sum(DEFAULT_QUOTAS.values())
    quotas = {label: round(n * args.size / total) for label, n in DEFAULT_QUOTAS.items()}
    quotas["inconclusiva"] += args.size - sum(quotas.values())

    kept, report = build_golden(
        make_chat(client, args.generator_model, 0.9), make_chat(client, args.judge_model, 0.0), quotas, args.seed
    )
    report.update(generator_model=args.generator_model, judge_model=args.judge_model, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.sample_for_humans and kept:
        sample = random.Random(args.seed).sample(kept, min(args.sample_for_humans, len(kept)))
        csv_path = args.human_review_out or args.out.with_name(f"{args.out.stem}_human_review.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "claim", "evidence", "gold_conclusion", "human_conclusion"])
            for k in sample:
                writer.writerow([k["id"], k["claim"], k["evidence"]["texto"], k["gold_conclusion"], ""])
        print(f"Amostra para revisão humana: {csv_path}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
