"""Compatibility implementation for the serialized Fake.br TF-IDF pipeline.

The original artifact serializes ``checa_ai.preprocessing.PortugueseTextPreprocessor``.
This module supplies the same public callable under that historical import path at load
time, without depending on a writable checkout of the training repository.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

import snowballstemmer

PORTUGUESE_STOPWORDS = frozenset(
    """
    a à às agora ainda alguém algum alguma algumas alguns ampla amplas amplo amplos
    ante antes ao aos após aquela aquelas aquele aqueles aquilo as até através cada
    coisa coisas com como contra contudo da daquela daquelas daquele daqueles das de
    dela delas dele depois dessa dessas desse desta deste deve devem devendo dever
    deveria deveriam devia deviam disse disso disto diz dizem do dos e é ela elas ele
    eles em enquanto entre era eram essa essas esse esta está estamos estão estas estava
    este eu fazendo fazer feita feitas feito foi for foram fosse fossem grande há isso
    isto já la lá lhe lhes lo mas me mesma mesmo meu meus minha muita muitos na não nas
    nem nenhum nessa neste ninguém no nos nós nossa nosso num numa nunca o os ou outra
    outro para pela pelas pelo pequena pequeno per perante pode pude podendo poder
    poderia poderiam podia pois por porém porque posso pouca primeiro própria próprio
    quais qual quando quanto quantos que quem são se seja sejam sem sempre sendo ser
    será seu si sido só sob sobre sua talvez também tampouco te tem tendo tenha ter teu
    ti tido tinha toda todas todavia todo todos tu tua tudo um uma umas uns vendo ver
    vez vindo você vocês vos
    """.split()
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip_accents(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


_STOPWORDS_ASCII = frozenset(_strip_accents(word) for word in PORTUGUESE_STOPWORDS)


class PortugueseTextPreprocessor:
    """Sklearn-compatible historical preprocessor used by the supplied pipeline."""

    def __init__(self, mode: str = "light") -> None:
        if mode not in {"light", "stem", "lemmatize"}:
            raise ValueError("mode deve ser 'light', 'stem' ou 'lemmatize'")
        self.mode = mode
        self._nlp: Any | None = None
        self._stemmer = snowballstemmer.stemmer("portuguese")

    def fit(self, X: Any, y: Any = None) -> "PortugueseTextPreprocessor":
        return self

    def transform(self, X: Any) -> list[str]:
        return [self(text) for text in X]

    def __call__(self, text: object) -> str:
        cleaned = html.unescape(str(text))
        cleaned = _URL_RE.sub(" URL ", cleaned)
        cleaned = _EMAIL_RE.sub(" EMAIL ", cleaned)
        cleaned = cleaned.casefold()
        if self.mode == "lemmatize":
            raise RuntimeError("artefato legado requer dependência spaCy para lematização")
        tokens = [_strip_accents(token) for token in _TOKEN_RE.findall(cleaned)]
        tokens = [token for token in tokens if token not in _STOPWORDS_ASCII]
        if self.mode == "stem":
            tokens = [self._stemmer.stemWord(token) for token in tokens]
        return " ".join(tokens)
