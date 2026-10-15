import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import analyze_retrieval_ab as ab  # noqa: E402
import paraphrase_benchmark as pb  # noqa: E402


def test_sign_test_symmetric_and_extreme():
    assert ab.sign_test_p(4, 5) == 1.0
    assert ab.sign_test_p(0, 0) == 1.0
    assert ab.sign_test_p(10, 0) < 0.01


def test_wrong_top1_counts_as_false_positive():
    cases = [
        {"expected_status": "matched", "best_score": 0.9, "rank": 2},
        {"expected_status": "matched", "best_score": 0.9, "rank": 1},
        {"expected_status": "abstained", "best_score": 0.7, "rank": None},
    ]
    c = ab.confusion(cases, 0.65)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 2, 1, 0)


def test_splits_are_balanced_per_status_and_deterministic():
    cases = [{"id": f"m{i}", "expected_status": "matched"} for i in range(10)]
    cases += [{"id": f"a{i}", "expected_status": "abstained"} for i in range(10)]
    pb.assign_balanced_splits(cases)
    first = [c["split"] for c in cases]
    pb.assign_balanced_splits(cases)
    assert first == [c["split"] for c in cases]
    for status in ("matched", "abstained"):
        group = [c for c in cases if c["expected_status"] == status]
        assert sum(c["split"] == "calibration" for c in group) == 5
