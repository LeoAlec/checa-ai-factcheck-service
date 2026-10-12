"""Offline, reproducible evaluation for candidate text classifiers."""

from .contracts import ClassifierOutput, DatasetRecord, DatasetSplit, TruthLabel

__all__ = ["ClassifierOutput", "DatasetRecord", "DatasetSplit", "TruthLabel"]
