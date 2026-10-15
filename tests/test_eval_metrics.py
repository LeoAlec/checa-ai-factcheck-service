import math

from src.evaluation.metrics import (
    bootstrap_ci,
    calibrate_threshold,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_coverage_curve,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_mrr_ndcg_basic():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == 1 / 3
    assert reciprocal_rank(ranked, {"x"}) == 0.0
    assert ndcg_at_k(ranked, {"a"}, 3) == 1.0
    assert math.isclose(ndcg_at_k(ranked, {"b"}, 3), 1 / math.log2(3))
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_macro_f1_and_confusion():
    gold = ["sustentada", "contradita", "inconclusiva", "contradita"]
    pred = ["sustentada", "contradita", "contradita", "contradita"]
    matrix = confusion_matrix(gold, pred)
    assert matrix["inconclusiva"]["contradita"] == 1
    assert 0 < macro_f1(gold, pred) < 1
    assert macro_f1(gold, gold) == 1.0


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_bootstrap_ci_is_reproducible_and_brackets_estimate():
    values = [1.0] * 8 + [0.0] * 2
    ci = bootstrap_ci(values)
    assert ci == bootstrap_ci(values)
    assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert bootstrap_ci([])["n"] == 0


def test_small_sample_ci_is_wide():
    ci = bootstrap_ci([1.0] * 9 + [0.0])
    assert ci["high"] - ci["low"] > 0.3


def test_paired_bootstrap_macro_f1():
    gold = ["sustentada"] * 10 + ["contradita"] * 10 + ["inconclusiva"] * 10
    ci = paired_bootstrap_ci(gold, gold)
    assert ci["estimate"] == 1.0 and ci["low"] == 1.0


def test_calibrate_threshold_picks_lowest_threshold_meeting_precision():
    scores = [0.9, 0.85, 0.8, 0.55, 0.5, 0.3]
    relevant = [True, True, True, False, False, False]
    best = calibrate_threshold(scores, relevant, candidates=[0.4, 0.6, 0.8], min_precision=1.0)
    assert best["threshold"] == 0.6 and best["met_target"]


def test_calibrate_threshold_reports_unmet_target():
    best = calibrate_threshold([0.9, 0.8], [False, False], candidates=[0.5], min_precision=0.9)
    assert best["met_target"] is False


def test_precision_coverage_curve():
    curve = precision_coverage_curve([0.9, 0.5], [True, False], [0.4, 0.7])
    assert curve[0] == {"threshold": 0.4, "coverage": 1.0, "precision": 0.5}
    assert curve[1]["coverage"] == 0.5 and curve[1]["precision"] == 1.0
