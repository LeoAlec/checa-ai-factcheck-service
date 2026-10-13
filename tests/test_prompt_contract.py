"""Contrato do prompt: roda em todo PR e impede mudanças silenciosas."""

import json
from pathlib import Path

from src.rag import prompts

LOCK = json.loads((Path(prompts.__file__).parent / "prompts.lock.json").read_text())


def test_prompt_text_change_requires_version_bump_and_lock_update():
    assert (
        LOCK["schema_version"] == 2
        and prompts.content_hash() == LOCK["sha256"]
        and prompts.prompt_manifest() == LOCK["prompts"]
    ), (
        "O texto de src/rag/prompts.py mudou. Incremente PROMPT_VERSION e rode "
        "`python scripts/sync_prompts.py --write-lock`."
    )


def test_lock_hash_covers_claim_analysis_prompts(monkeypatch):
    original_hash = prompts.content_hash()
    monkeypatch.setattr(
        prompts,
        "CLAIM_ANALYSIS_SYSTEM_PROMPT",
        prompts.CLAIM_ANALYSIS_SYSTEM_PROMPT + "\nMudança de teste.",
    )

    assert prompts.content_hash() != original_hash


def test_each_prompt_has_an_independent_name_version_and_hash():
    manifest = prompts.prompt_manifest()

    assert set(manifest) == {"evidence-flow", "claim-analysis"}
    assert all(item["version"] and len(item["sha256"]) == 64 for item in manifest.values())


def test_user_prompt_has_required_variables():
    assert "{{query}}" in prompts.USER_PROMPT and "{{contexto}}" in prompts.USER_PROMPT


def test_system_prompt_keeps_safety_rules():
    s = prompts.SYSTEM_PROMPT
    assert "APENAS as evidências" in s          # anti-alucinação
    assert "não confiáveis" in s                # prompt injection nos laudos
    assert "fontes_usadas" in s and "JSON" in s  # saída estruturada
    assert "laudos_tratam_alegacao" in s        # abstenção pelo modelo
    assert "Fonte:" in s                        # crédito às fontes


def test_prompts_render_without_leftover_placeholders():
    out = prompts.render(prompts.USER_PROMPT, query="q", contexto="c")
    assert "{{" not in out
