from scripts.build_ner_anchor_benchmark import POSITIVE_IDS, build_cases


def test_build_cases_has_balanced_anchor_labels():
    items = [
        {"id": case_id, "claim": case_id, "evidence": {"titulo": "t", "texto": "e"}}
        for case_id in POSITIVE_IDS
    ]
    cases = build_cases(items)
    assert len(cases) == len(POSITIVE_IDS) * 2
    assert sum(case["expected_anchor"] for case in cases) == len(POSITIVE_IDS)
    assert sum(not case["expected_anchor"] for case in cases) == len(POSITIVE_IDS)
