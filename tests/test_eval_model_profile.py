from pathlib import Path

import pytest

from scripts.eval_model_profile import load_profile, profile_environment


ROOT = Path(__file__).resolve().parent.parent


def test_openai_profile_keeps_generation_remote_without_embedding_changes():
    base = {
        "OPENAI_API_KEY": "secret",
        "EMBEDDING_PROVIDER": "openai",
        "QDRANT_COLLECTION": "fact_checks_pt_v2_openai",
    }

    result = profile_environment("openai-gpt-4o-mini", base)

    assert result["LLM_PROVIDER"] == "openai"
    assert result["LLM_MODEL"] == "gpt-4o-mini"
    assert result["LLM_BASE_URL"] == ""
    assert result["OPENAI_API_KEY"] == "secret"
    assert result["EMBEDDING_PROVIDER"] == "openai"
    assert result["QDRANT_COLLECTION"] == "fact_checks_pt_v2_openai"


def test_ollama_profile_uses_host_endpoint_for_local_evaluation():
    result = profile_environment("ollama-qwen3-8b", {"OPENAI_API_KEY": "judge-key"})

    assert result["LLM_PROVIDER"] == "ollama"
    assert result["LLM_MODEL"] == "qwen3:8b"
    assert result["LLM_BASE_URL"] == "http://localhost:11434/v1/"
    assert result["LLM_API_KEY"] == "ollama"
    assert result["OPENAI_API_KEY"] == "judge-key"


def test_profiles_never_define_real_provider_secrets():
    for path in sorted((ROOT / "config" / "llm").glob("*.env")):
        profile = load_profile(path)
        assert "OPENAI_API_KEY" not in profile
        assert "TAVILY_API_KEY" not in profile


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Perfil de LLM desconhecido"):
        profile_environment("missing", {})
