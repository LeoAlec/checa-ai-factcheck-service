from src.rag.entities import (
    all_entities_present,
    entity_in_text,
    extract_entities,
    heuristic_entities,
    normalize,
)


def test_acronyms_and_multiword_entities_detected():
    entities = heuristic_entities("O TSE disse que Luiz Inácio da Silva venceu")
    assert "tse" in entities
    assert "luiz inacio da silva" in entities


def test_all_caps_message_does_not_invent_entities():
    assert heuristic_entities("URGENTE O GOVERNO VAI TAXAR O PIX") == []


def test_alias_matches_expanded_name():
    assert all_entities_present(
        "O STF decidiu isso",
        "O Supremo Tribunal Federal decidiu o contrário.",
        backend="heuristic",
    )


def test_missing_entity_is_detected():
    assert not all_entities_present(
        "A Polícia Federal prendeu o suspeito",
        "O Ministério da Saúde divulgou nota.",
        backend="heuristic",
    )


def test_word_boundary_prevents_substring_false_positive():
    assert not entity_in_text("tse", normalize("cartselo detsea"))
    assert entity_in_text("tse", normalize("o TSE informou"))


def test_extract_entities_uses_heuristic_backend():
    assert "tse" in extract_entities("O TSE informou", backend="heuristic")
