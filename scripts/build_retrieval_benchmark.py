"""
build_retrieval_benchmark.py
─────────────────────────────────────────────────────────────────────
Gera um benchmark ampliado e balanceado (120 casos) para avaliação
estatística rigorosa de retrieval (Recall@k, MRR, nDCG, calibração
de limiar e abstenção).

Composição:
  - 60 casos 'matched': consultas e alegações com laudo de checagem real
    confirmado no índice Qdrant / base Silver, cobrindo saúde, política,
    ciência, mundo e economia.
  - 60 casos 'abstained':
      * 12 consultas benignas fora de domínio (receitas, programação, clima).
      * 28 hard negatives de notícias reais/fatos verídicos sem laudo no corpus.
      * 20 hard negatives de boatos virais fora do corpus (sem laudo na base).

Uso:
    python scripts/build_retrieval_benchmark.py
    python scripts/build_retrieval_benchmark.py --output data/eval/retrieval_benchmark_120.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "data" / "eval" / "retrieval_benchmark_120.jsonl"
DEFAULT_MANIFEST = ROOT / "reports" / "retrieval" / "benchmark_120_manifest.json"

# Caminhos padrão do corpus Silver (local ou no repo irmão checa-ai-ml)
SILVER_CANDIDATES = [
    ROOT / "data" / "processed" / "fact_checks_silver.jsonl",
    ROOT.parent / "checa-ai-ml" / "data" / "processed" / "fact_checks_silver.jsonl",
]

ORIG_EVAL_CANDIDATES = [
    ROOT / "data" / "eval" / "retrieval_eval.jsonl",
    ROOT.parent / "checa-ai-ml" / "data" / "eval" / "retrieval_eval.jsonl",
]

BENIGN_QUERIES = [
    ("benign_weather_01", "Vai chover amanhã em São Paulo ou Rio de Janeiro?"),
    ("benign_weather_02", "Qual a previsão do tempo para o fim de semana em Curitiba?"),
    ("benign_recipe_01", "Qual é a melhor receita de bolo de cenoura com cobertura de chocolate?"),
    ("benign_recipe_02", "Como preparar risoto de cogumelos tradicional passo a passo"),
    ("benign_code_01", "Como configurar conexão assíncrona com SQLAlchemy e FastAPI no Python 3.12"),
    ("benign_code_02", "Qual a diferença entre busca vetorial densa e esparsa BM25?"),
    ("benign_sports_01", "Quem venceu a Libertadores da América no ano passado?"),
    ("benign_sports_02", "Qual a escalação da seleção brasileira no último amistoso internacional?"),
    ("benign_science_01", "Qual a distância média entre a Terra e a Lua em quilômetros?"),
    ("benign_science_02", "Como funciona a fotossíntese nas plantas durante o dia?"),
    ("benign_daily_01", "Dicas para melhorar a qualidade do sono e dormir mais rápido"),
    ("benign_daily_02", "Quais documentos são necessários para renovar a carteira de habilitação CNH?"),
]


def find_existing_file(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def clean_claim(title: str) -> str:
    title = str(title).strip()
    title = re.sub(r"\s*#boato\b.*|\?$", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^[\s\"']+|[\s\"']+$", "", title)
    title = re.sub(r"^([EÉ]\s+#?FAKE\s+que\s+)", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^([EÉ]\s+falso\s+que\s+)", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"\s+", " ", title)
    return title


def load_silver_urls(silver_path: Path) -> set[str]:
    urls = set()
    with open(silver_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u = json.loads(line).get("url")
                if u:
                    urls.add(u.strip())
    return urls


def build_benchmark(
    silver_path: Path,
    recogna_path: Path,
    faketrue_path: Path,
    orig_eval_path: Path | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    random.seed(seed)
    silver_urls = load_silver_urls(silver_path)

    # 1. Carrega casos originais se existirem
    orig_matched = []
    seen_urls = set()
    if orig_eval_path and orig_eval_path.exists():
        with open(orig_eval_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    if c.get("expected_status") == "matched" and c.get("expected_url") in silver_urls:
                        orig_matched.append(c)
                        seen_urls.add(c["expected_url"])

    # 2. Amostra novos casos 'matched' a partir de FakeRecogna
    import pandas as pd

    df_recogna = pd.read_excel(recogna_path)
    fakes_in_silver = df_recogna[
        (df_recogna["Classe"] == 0) & (df_recogna["URL"].str.strip().isin(silver_urls))
    ].copy()

    target_matched = 60
    needed_matched = target_matched - len(orig_matched)
    new_matched = []

    cats = list(fakes_in_silver["Categoria"].dropna().unique())
    random.shuffle(cats)

    for cat in cats:
        cat_df = fakes_in_silver[fakes_in_silver["Categoria"] == cat]
        for _, row in cat_df.sample(frac=1.0, random_state=seed).iterrows():
            url = row["URL"].strip()
            if url in seen_urls:
                continue
            claim = clean_claim(row["Titulo"])
            if len(claim) < 18 or len(claim) > 220:
                continue

            seen_urls.add(url)
            case_id = f"recogna_matched_{str(cat).lower()}_{len(new_matched):03d}"
            new_matched.append(
                {
                    "id": case_id,
                    "query": claim,
                    "expected_url": url,
                    "expected_status": "matched",
                    "category": str(cat),
                }
            )
            if len(new_matched) >= needed_matched:
                break
        if len(new_matched) >= needed_matched:
            break

    total_matched = orig_matched + new_matched

    # 3. Casos 'abstained'
    # a) Benignos fora de domínio
    abstained_cases = []
    for b_id, b_query in BENIGN_QUERIES:
        abstained_cases.append(
            {
                "id": b_id,
                "query": b_query,
                "expected_url": None,
                "expected_status": "abstained",
                "category": "benign_out_of_domain",
            }
        )

    # b) Hard negatives 1: Notícias reais/verídicas sem laudo no corpus
    true_news = df_recogna[
        (df_recogna["Classe"] == 1) & (~df_recogna["URL"].str.strip().isin(silver_urls))
    ].copy()
    hard_true = []
    n_sample_true = min(len(true_news), 28)
    if n_sample_true > 0:
        for _, row in true_news.sample(n=n_sample_true, random_state=seed).iterrows():
            title = clean_claim(row["Titulo"])
            if 20 <= len(title) <= 220:
                hard_true.append(
                    {
                        "id": f"hard_neg_true_news_{len(hard_true):03d}",
                        "query": title,
                        "expected_url": None,
                        "expected_status": "abstained",
                        "category": str(row.get("Categoria", "noticia_real")),
                    }
                )
    abstained_cases.extend(hard_true[:28])

    # c) Hard negatives 2: Boatos reais virais sem laudo no corpus
    df_faketrue = pd.read_csv(faketrue_path)
    df_faketrue["link_clean"] = df_faketrue["link_f"].dropna().str.strip()
    faketrue_not_in_silver = df_faketrue[~df_faketrue["link_clean"].isin(silver_urls)].copy()

    hard_rumors = []
    n_sample_rumors = min(len(faketrue_not_in_silver), 20)
    if n_sample_rumors > 0:
        for _, row in faketrue_not_in_silver.sample(n=n_sample_rumors, random_state=seed).iterrows():
            title = clean_claim(str(row["title_fake"]))
            if 20 <= len(title) <= 220:
                hard_rumors.append(
                    {
                        "id": f"hard_neg_out_corpus_{len(hard_rumors):03d}",
                        "query": title,
                        "expected_url": None,
                        "expected_status": "abstained",
                        "category": "rumor_sem_laudo_no_corpus",
                    }
                )
    abstained_cases.extend(hard_rumors[:20])

    # Combinação final e embaralhamento reproduzível
    all_cases = total_matched + abstained_cases
    random.shuffle(all_cases)

    manifest = {
        "total_cases": len(all_cases),
        "matched_cases": len(total_matched),
        "abstained_cases": len(abstained_cases),
        "abstained_composition": {
            "benign_out_of_domain": len(BENIGN_QUERIES),
            "hard_negatives_true_news": len(hard_true[:28]),
            "hard_negatives_out_of_corpus_rumors": len(hard_rumors[:20]),
        },
        "matched_from_orig_eval": len(orig_matched),
        "matched_from_fakerecogna": len(new_matched),
        "seed": seed,
    }

    return all_cases, manifest


def main():
    parser = argparse.ArgumentParser(description="Gera benchmark de retrieval de 120 casos")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Caminho do JSONL de saída")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Caminho do manifesto JSON")
    parser.add_argument("--silver", type=Path, default=find_existing_file(SILVER_CANDIDATES))
    parser.add_argument("--recogna", type=Path, default=ROOT / "data" / "raw" / "FakeRecogna.xlsx")
    parser.add_argument("--faketrue", type=Path, default=ROOT / "data" / "raw" / "FakeTrueBr_corpus.csv")
    parser.add_argument("--orig-eval", type=Path, default=find_existing_file(ORIG_EVAL_CANDIDATES))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.silver or not args.silver.exists():
        print(f"Erro: Arquivo silver não encontrado em {SILVER_CANDIDATES}", file=sys.stderr)
        sys.exit(1)

    print(f"Usando base Silver: {args.silver}")
    cases, manifest = build_benchmark(
        silver_path=args.silver,
        recogna_path=args.recogna,
        faketrue_path=args.faketrue,
        orig_eval_path=args.orig_eval,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"Benchmark gerado com sucesso em: {args.output}")
    print(f"Manifesto salvo em: {args.manifest}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
