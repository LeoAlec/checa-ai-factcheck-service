"""Executa o golden set com um perfil de geração versionado e sem segredos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "config" / "llm"
ALLOWED_KEYS = {
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_REASONING_EFFORT",
    "LLM_TIMEOUT_SECONDS",
    "EVAL_LLM_BASE_URL",
}
REQUIRED_KEYS = {"LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"}


def load_profile(path: Path) -> dict[str, str]:
    """Lê um arquivo KEY=VALUE simples e rejeita chaves fora do contrato."""
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Linha inválida em {path}:{number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_KEYS:
            raise ValueError(f"Chave não permitida no perfil: {key}")
        values[key] = value.strip()
    missing = REQUIRED_KEYS - values.keys()
    if missing:
        raise ValueError(f"Perfil incompleto; faltam: {', '.join(sorted(missing))}")
    return values


def available_profiles() -> dict[str, Path]:
    return {path.stem: path for path in sorted(PROFILE_DIR.glob("*.env"))}


def profile_environment(name: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    profiles = available_profiles()
    if name not in profiles:
        raise ValueError(f"Perfil de LLM desconhecido: {name}")
    profile = load_profile(profiles[name])
    environment = dict(base or os.environ)
    environment.update({key: value for key, value in profile.items() if key != "EVAL_LLM_BASE_URL"})
    if "EVAL_LLM_BASE_URL" in profile:
        environment["LLM_BASE_URL"] = profile["EVAL_LLM_BASE_URL"]
    return environment


def main() -> int:
    profiles = available_profiles()
    parser = argparse.ArgumentParser(description="Avalia um perfil de geração no golden set")
    parser.add_argument("profile", choices=sorted(profiles))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-faithfulness-judge", action="store_true")
    parser.add_argument("--contract", choices=("evidence", "backend-v1"), default="backend-v1")
    args = parser.parse_args()

    report = args.report or ROOT / "reports" / "eval" / f"generation-{args.profile}.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "eval_generation.py"),
        "--report",
        str(report),
        "--contract",
        args.contract,
    ]
    if args.skip_faithfulness_judge:
        command.append("--skip-faithfulness-judge")
    return subprocess.run(command, cwd=ROOT, env=profile_environment(args.profile), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
