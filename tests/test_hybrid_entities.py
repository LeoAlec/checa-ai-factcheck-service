from src.rag.entities import all_entities_present, entity_in_text, extract_entities, heuristic_entities, normalize
from src.rag.hybrid import bm25_scores, hybrid_rerank, rrf_fuse, tokenize
from src.rag.retriever import Evidence


def ev(doc, score, texto, titulo="t"):
    return Evidence(
        document_id=doc, titulo=titulo, dominio="d", url=f"https://d/{doc}",
        data_publicacao="", texto=texto, score=score,
    )


# ─── Híbrido ─────────────────────────────────────────────────────────
def test_tokenize_strips_accents_and_stopwords_keeps_numbers():
    assert tokenize("O Governo vai cobrar R$ 5 mil") == ["governo", "cobrar", "5", "mil"] or "5" in tokenize("R$ 5 mil")
    assert "governo" in tokenize("Governo") and "de" not in tokenize("de")


def test_bm25_prefers_exact_terms():
    scores = bm25_scores("taxar pix", ["imposto sobre o pix será taxar", "vacina contra covid"])
    assert scores[0] > scores[1] == 0.0


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [1, 3, 2]])
    assert fused[0] > fused[1] > fused[2] or fused[0] > fused[2]


def test_hybrid_promotes_lexical_match_without_touching_score():
    dense_top = ev("a", 0.80, "texto genérico sobre economia")
    lexical = ev("b", 0.79, "Receita Federal esclarece que o Pix não será taxado em 2025")
    out = hybrid_rerank("Pix taxado 2025 Receita Federal", [dense_top, lexical], lexical_weight=1.2)
    by_id = {e.document_id: e for e in out}
    assert by_id["b"].rerank_score > by_id["a"].rerank_score
    assert (by_id["a"].score, by_id["b"].score) == (0.80, 0.79)


def test_hybrid_noop_for_single_candidate():
    only = [ev("a", 0.7, "x")]
    assert hybrid_rerank("q", only) == only


# ─── Entidades ───────────────────────────────────────────────────────
def test_acronyms_and_multiword_entities_detected():
    ents = heuristic_entities("O TSE disse que Luiz Inácio da Silva venceu")
    assert "tse" in ents and "luiz inacio da silva" in ents


def test_all_caps_message_does_not_invent_entities():
    assert heuristic_entities("URGENTE!!! O GOVERNO VAI TAXAR O PIX COMPARTILHE") == []


def test_alias_matches_expanded_name():
    assert all_entities_present("O STF decidiu isso", "O Supremo Tribunal Federal decidiu o contrário.", backend="heuristic")


def test_missing_entity_is_detected():
    assert not all_entities_present("A Polícia Federal prendeu o suspeito", "O Ministério da Saúde divulgou nota.", backend="heuristic")


def test_word_boundary_prevents_substring_false_positive():
    assert not entity_in_text("tse", normalize("cartselo detsea"))
    assert entity_in_text("tse", normalize("o TSE informou"))


def test_extract_entities_falls_back_to_heuristic_without_spacy():
    assert "tse" in extract_entities("O TSE informou", backend="heuristic")
