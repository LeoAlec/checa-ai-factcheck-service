"""
src/rag/entities.py
─────────────────────────────────────────────────────────────────────
Extração de entidades da alegação para a salvaguarda "a fonte fala da mesma
entidade?". Backend configurável (`NER_BACKEND`):

  - "spacy": usa o modelo `pt_core_news_*` (se instalado);
  - "heuristic": maiúsculas + siglas, robusto a mensagens em CAIXA ALTA;
  - "auto" (padrão): spaCy quando disponível, senão heurística.

A comparação com o texto-fonte é por fronteira de palavra, sem acento/caixa,
e aceita aliases conhecidos (ex.: "STF" ↔ "Supremo Tribunal Federal").
"""

from __future__ import annotations

import logging
import re
import unicodedata

from src.config import NER_BACKEND, NER_SPACY_MODEL

logger = logging.getLogger("checa-ai.entities")

_PARTICLES = {"da", "das", "de", "do", "dos", "e"}
_ARTICLES = {"a", "o", "os", "as", "um", "uma"}
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_ACRONYM_RE = re.compile(r"^[A-ZÀ-Ý]{2,6}$")
# Siglas muito comuns que não identificam uma entidade específica.
_ACRONYM_STOP = {"OK", "AI", "TV"}
_SHOUTING_RATIO = 0.6

ALIASES: dict[str, set[str]] = {
    "stf": {"supremo tribunal federal"},
    "tse": {"tribunal superior eleitoral"},
    "tcu": {"tribunal de contas da uniao"},
    "stj": {"superior tribunal de justica"},
    "pf": {"policia federal"},
    "prf": {"policia rodoviaria federal"},
    "bc": {"banco central"},
    "bcb": {"banco central"},
    "oms": {"organizacao mundial da saude"},
    "onu": {"organizacao das nacoes unidas"},
    "inss": {"instituto nacional do seguro social"},
    "anvisa": {"agencia nacional de vigilancia sanitaria"},
    "sus": {"sistema unico de saude"},
    "eua": {"estados unidos"},
}
_REVERSE_ALIASES: dict[str, set[str]] = {}
for _short, _longs in ALIASES.items():
    for _long in _longs:
        _REVERSE_ALIASES.setdefault(_long, set()).add(_short)

_nlp = None
_nlp_failed = False


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return " ".join(
        "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().split()
    )


def _is_shouting(tokens: list[str]) -> bool:
    letters = [t for t in tokens if len(t) > 1]
    return bool(letters) and sum(t.isupper() for t in letters) / len(letters) >= _SHOUTING_RATIO


def heuristic_entities(claim: str) -> list[str]:
    """Nomes próprios multi-palavra e siglas. Em CAIXA ALTA não infere nada (evita falso bloqueio)."""
    tokens = _WORD_RE.findall(claim)
    if _is_shouting(tokens):
        return []
    entities: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ACRONYM_RE.match(token) and token not in _ACRONYM_STOP:
            entities.append(token)
            index += 1
            continue
        if not token[:1].isupper() or token.lower() in _ARTICLES:
            index += 1
            continue
        parts, capitals, cursor = [token], 1, index + 1
        while cursor < len(tokens):
            candidate = tokens[cursor]
            if candidate[:1].isupper() and not _ACRONYM_RE.match(candidate):
                parts.append(candidate)
                capitals += 1
                cursor += 1
            elif (
                candidate.lower() in _PARTICLES
                and cursor + 1 < len(tokens)
                and tokens[cursor + 1][:1].isupper()
            ):
                parts.append(candidate)
                cursor += 1
            else:
                break
        if capitals >= 2:
            entities.append(" ".join(parts))
        index = max(cursor, index + 1)
    return _dedupe(entities)


def _get_spacy():
    global _nlp, _nlp_failed
    if _nlp is None and not _nlp_failed:
        try:
            import spacy

            _nlp = spacy.load(NER_SPACY_MODEL, disable=["parser", "lemmatizer"])
        except Exception:  # noqa: BLE001
            _nlp_failed = True
            logger.info("spaCy/%s indisponível; usando heurística de entidades.", NER_SPACY_MODEL)
    return _nlp


def spacy_entities(claim: str) -> list[str] | None:
    """Entidades PER/ORG/LOC/MISC do spaCy; `None` se o modelo não está disponível."""
    nlp = _get_spacy()
    if nlp is None:
        return None
    labels = {"PER", "ORG", "LOC", "MISC", "GPE"}
    return _dedupe([ent.text for ent in nlp(claim).ents if ent.label_ in labels])


def _dedupe(entities: list[str]) -> list[str]:
    return sorted({normalize(e) for e in entities if e.strip()}, key=len, reverse=True)


def extract_entities(claim: str, backend: str = NER_BACKEND) -> list[str]:
    """Entidades normalizadas (maiores primeiro) segundo o backend configurado."""
    if backend in {"spacy", "auto"}:
        found = spacy_entities(claim)
        if found is not None:
            # União com a heurística: o NER perde siglas e nomes em minúsculas/curtos.
            return _dedupe(found + heuristic_entities(claim))
    return heuristic_entities(claim)


def _variants(entity: str) -> set[str]:
    return {entity} | ALIASES.get(entity, set()) | _REVERSE_ALIASES.get(entity, set())


def entity_in_text(entity: str, normalized_text: str) -> bool:
    """Fronteira de palavra, aceitando aliases (texto já normalizado)."""
    return any(
        re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized_text)
        for variant in _variants(entity)
    )


def all_entities_present(claim: str, source_text: str, backend: str = NER_BACKEND) -> bool:
    entities = extract_entities(claim, backend)
    if not entities:
        return True
    text = normalize(source_text)
    return all(entity_in_text(entity, text) for entity in entities)
