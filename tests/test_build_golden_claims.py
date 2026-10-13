import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_golden_claims as bgc  # noqa: E402


def fake_generate_factory():
    counter = {"n": 0}

    def generate(system, user):
        label = next(l for l in bgc.DEFAULT_QUOTAS if f'"{l}"' in user)
        items = []
        for _ in range(8):
            counter["n"] += 1
            n = counter["n"]
            items.append(
                {
                    "claim": f"alegacao numero {n} sobre tema {label} distinto alfa{n} beta{n * 7}",
                    "evidence_title": f"Laudo {n}",
                    "evidence_text": f"{label} texto {n}",
                    "tags": ["plain"],
                }
            )
        return json.dumps({"items": items})

    return generate


def judge_that_agrees(system, user):
    for label in bgc.CONCLUSIONS:
        if f"\n{label} texto" in user:
            return json.dumps({"conclusion": label, "reason": "ok"})
    return json.dumps({"conclusion": "inconclusiva", "reason": "?"})


def test_build_golden_fills_quotas_with_agreeing_judge():
    quotas = {"contradita": 4, "sustentada": 3, "inconclusiva": 3}
    kept, report = bgc.build_golden(fake_generate_factory(), judge_that_agrees, quotas)
    assert report["complete"] and len(kept) == 10
    assert report["by_label"] == quotas
    assert report["judge_agreement"] == 1.0
    assert len({k["id"] for k in kept}) == 10
    assert all(k["synthetic"] for k in kept)


def test_disagreeing_judge_discards_items_and_reports_incomplete():
    quotas = {"contradita": 2, "sustentada": 2, "inconclusiva": 2}
    always_inconclusive = lambda s, u: json.dumps({"conclusion": "inconclusiva", "reason": "x"})  # noqa: E731
    kept, report = bgc.build_golden(fake_generate_factory(), always_inconclusive, quotas, max_rounds=2)
    assert {k["gold_conclusion"] for k in kept} == {"inconclusiva"}
    assert report["disagreements"] > 0 and not report["complete"]


def test_duplicates_are_removed():
    assert bgc.is_duplicate("A vacina tem microchip", ["a vacina tem microchip!"])
    assert not bgc.is_duplicate("Pix será taxado", ["vacina tem microchip"])


def test_parsers_reject_malformed_output():
    assert bgc.parse_items("não é json") == []
    assert bgc.parse_items(json.dumps({"items": [{"claim": "x"}]})) == []
    assert bgc.parse_judgement(json.dumps({"conclusion": "talvez"})) is None
