import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from review_golden_claims_with_llm import parse_review  # noqa: E402


def test_parse_review_accepts_complete_contract() -> None:
    assert parse_review('{"conclusion":"inconclusiva","reason":"datas diferentes"}') == (
        "inconclusiva",
        "datas diferentes",
    )


@pytest.mark.parametrize(
    "payload",
    ['{"conclusion":"talvez","reason":"x"}', '{"conclusion":"sustentada","reason":""}'],
)
def test_parse_review_rejects_invalid_contract(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_review(payload)
