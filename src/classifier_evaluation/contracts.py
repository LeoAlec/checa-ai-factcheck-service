"""Validated contracts for the external classifier evaluation."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TruthLabel(StrEnum):
    TRUE = "verdadeiro"
    FALSE = "falso"


class DatasetSplit(StrEnum):
    CALIBRATION = "calibration"
    TEST = "test"


class ThresholdBand(StrEnum):
    LOW = "baixo"
    MEDIUM = "medio"
    HIGH = "alto"


class DatasetRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=5, max_length=5_000)
    label: TruthLabel
    topic: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=1_000)
    published_at: date
    split: DatasetSplit

    @field_validator("text")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("text contém caracteres de controle inválidos")
        return value

    @property
    def binary_label(self) -> int:
        return int(self.label is TruthLabel.FALSE)


class ClassifierOutput(BaseModel):
    """Stable output contract reserved for a future production integration."""

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    probability_false: float = Field(ge=0, le=1)
    calibrated: bool
    threshold_band: ThresholdBand
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def from_probability(
        cls,
        *,
        model_name: str,
        model_version: str,
        probability_false: float,
        calibrated: bool,
        limitations: list[str] | None = None,
        low_max: float = 0.35,
        high_min: float = 0.65,
    ) -> "ClassifierOutput":
        if not 0 <= low_max < high_min <= 1:
            raise ValueError("limites das faixas devem satisfazer 0 <= baixo < alto <= 1")
        band = (
            ThresholdBand.LOW
            if probability_false < low_max
            else ThresholdBand.HIGH
            if probability_false >= high_min
            else ThresholdBand.MEDIUM
        )
        return cls(
            model_name=model_name,
            model_version=model_version,
            probability_false=probability_false,
            calibrated=calibrated,
            threshold_band=band,
            limitations=limitations or [],
        )
