"""
src/rag/hybrid.py
─────────────────────────────────────────────────────────────────────
Fusão híbrida lexical + densa sobre os candidatos já recuperados.

O denso (embeddings) é fraco para entidades, números e datas, que dominam a
checagem de fatos. Aqui os candidatos do Qdrant são repontuados com BM25
(estatísticas calculadas sobre o próprio conjunto de candidatos) e fundidos
com a ordem densa/cluster por Reciprocal Rank Fusion (RRF).

Importante: o `score` de similaridade (usado no limiar e nas métricas) NUNCA
é alterado; só o `rerank_score`, que define a ordem de seleção.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from typing import Sequence

from src.rag.retriever import Evidence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a o as os um uma uns umas de da do das dos em no na nos nas por para com sem e ou que "
    "se ao aos foi ser sao esta estao vai vao ja mais muito sobre como mas tem ter ha".split()
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, sem stopwords; mantém números (R$ 5 mil → 5, mil)."""
    base = unicodedata.normalize("NFKD", text or "")
    base = "".join(ch for ch in base if not unicodedata.combining(ch)).lower()
    return [t for t in _TOKEN_RE.findall(base) if t not in _STOPWORDS and len(t) > 1 or t.isdigit()]


def bm25_scores(
    query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """BM25 Okapi da `query` contra cada documento (IDF só sobre o conjunto dado)."""
    docs = [tokenize(d) for d in documents]
    q_terms = set(tokenize(query))
    if not docs or not q_terms:
        return [0.0] * len(docs)
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(term for d in docs for term in set(d) if term in q_terms)
    scores = []
    for tokens in docs:
        tf = Counter(tokens)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(tokens) / avg_len))
        scores.append(score)
    return scores


def _ranks(values: Sequence[float]) -> list[int]:
    """Posição (1 = maior valor); empates dividem a mesma posição (competition ranking).

    Assim um sinal sem informação (ex.: BM25 todo zero) soma o mesmo a todos
    os candidatos e preserva a ordem dos demais sinais.
    """
    ranks = []
    for value in values:
        ranks.append(1 + sum(1 for other in values if other > value))
    return ranks


def rrf_fuse(rankings: Sequence[Sequence[int]], weights: Sequence[float] | None = None, k: int = RRF_K) -> list[float]:
    """RRF: Σ w_r / (k + posição_r). `rankings[r][i]` é a posição do item i na lista r."""
    weights = weights or [1.0] * len(rankings)
    size = len(rankings[0]) if rankings else 0
    return [sum(w / (k + r[i]) for r, w in zip(rankings, weights)) for i in range(size)]


def hybrid_rerank(
    query: str,
    evidences: list[Evidence],
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[Evidence]:
    """Preenche `rerank_score` ∈ [0,1] com a fusão RRF (denso/cluster + BM25).

    Chunks sem nenhum termo em comum com a consulta não são penalizados além da
    própria posição lexical. O `score` original é preservado.
    """
    if len(evidences) < 2:
        return evidences
    dense = [ev.rerank_score if ev.rerank_score is not None else ev.score for ev in evidences]
    lexical = bm25_scores(query, [f"{ev.titulo} {ev.texto}" for ev in evidences])
    fused = rrf_fuse([_ranks(dense), _ranks(lexical)], [dense_weight, lexical_weight])
    top = max(fused) or 1.0
    return [replace(ev, rerank_score=value / top) for ev, value in zip(evidences, fused)]
