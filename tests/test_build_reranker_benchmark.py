from scripts.build_reranker_benchmark import build_rows, normalize_url
from scripts.evaluate_reranker_offline import exact_sign_p_value, paired_bootstrap_delta


def test_normalize_url_removes_query_fragment_and_duplicate_slashes():
    assert normalize_url("HTTPS://Example.COM//a/b/?x=1#z") == "https://example.com/a/b"


def test_build_rows_joins_candidate_payload_by_normalized_url():
    benchmark = [{"id": "one", "query": "alegação", "expected_status": "matched", "expected_url": "https://x.test/a"}]
    report = {"cases": [{"id": "one", "ranked_urls": ["https://x.test//a/"]}]}
    corpus = [{"url": "https://x.test/a", "titulo": "Título", "texto": "Laudo"}]
    rows, missing = build_rows(benchmark, report, corpus)
    assert missing == []
    assert rows[0]["candidates"][0]["text"] == "Laudo"


def test_paired_statistics_are_deterministic():
    delta = paired_bootstrap_delta([1.0, 0.5, 0.0], [1.0, 1.0, 0.0], iterations=1_000)
    assert delta["estimate"] == 1 / 6
    assert delta["low"] <= delta["estimate"] <= delta["high"]
    assert exact_sign_p_value(4, 5) == 1.0
