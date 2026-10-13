"""Artifact inventory utilities. Model files remain outside Git."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactInfo:
    name: str
    path: str
    kind: str
    available: bool
    size_bytes: int | None
    version: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def artifact_version(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def inspect_artifact(name: str, path: str | Path | None, kind: str) -> ArtifactInfo:
    normalized = Path(path).expanduser().resolve() if path else None
    available = bool(normalized and normalized.exists())
    return ArtifactInfo(
        name=name,
        path=str(normalized) if normalized else "",
        kind=kind,
        available=available,
        size_bytes=artifact_size(normalized) if available and normalized else None,
        version=artifact_version(normalized) if available and normalized else None,
    )


def inventory(
    tfidf_path: str | Path | None,
    bertimbau_path: str | Path | None,
    *,
    tfidf_metrics_path: str | Path | None = None,
    tfidf_predictions_path: str | Path | None = None,
    bertimbau_metrics_path: str | Path | None = None,
    bertimbau_predictions_path: str | Path | None = None,
) -> list[ArtifactInfo]:
    return [
        inspect_artifact("tfidf_model", tfidf_path, "joblib"),
        inspect_artifact("tfidf_metrics", tfidf_metrics_path, "json"),
        inspect_artifact("tfidf_predictions", tfidf_predictions_path, "csv"),
        inspect_artifact("bertimbau_model", bertimbau_path, "huggingface_directory"),
        inspect_artifact("bertimbau_metrics", bertimbau_metrics_path, "json"),
        inspect_artifact("bertimbau_predictions", bertimbau_predictions_path, "csv"),
    ]
