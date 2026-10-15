from types import SimpleNamespace

from src.rag import nli


def test_normalize_model_labels():
    assert nli._normalize_label("ENTAILMENT") == "entailment"
    assert nli._normalize_label("contradiction") == "contradiction"
    assert nli._normalize_label("LABEL_NEUTRAL") == "neutral"


def test_resolve_device_prefers_cuda():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    assert nli._resolve_device(fake_torch, "auto") == 0


def test_resolve_device_falls_back_to_cpu():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    assert nli._resolve_device(fake_torch, "auto") == -1
