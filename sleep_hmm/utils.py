from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_session_name(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", name.strip())
    sanitized = sanitized.strip(" ._")
    return sanitized or "session"


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = (matrix - mean) / std
    return scaled, mean, std


def minmax_scale_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = matrix.min(axis=0)
    scale = matrix.max(axis=0) - minimum
    scale = np.where(scale < 1e-8, 1.0, scale)
    scaled = (matrix - minimum) / scale
    return scaled, minimum, scale


def inverse_threshold(threshold_scaled: float, mean: float, std: float) -> float:
    return threshold_scaled * std + mean


def dataframe_from_matrix(matrix: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=columns)


def compute_confusion_matrix(reference: np.ndarray, other: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for ref_label, other_label in zip(reference.astype(int), other.astype(int), strict=True):
        matrix[ref_label, other_label] += 1
    return matrix


def pairwise_distance_matrix(features: np.ndarray) -> np.ndarray:
    return cdist(features, features, metric="euclidean")


def stable_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def choose_feature_subset(columns: list[str], max_features: int) -> list[str]:
    if len(columns) <= max_features:
        return columns
    priority = [
        "eeg_energy",
        "emg_energy",
        "eeg_peak_to_peak",
        "emg_peak_to_peak",
        "eeg_zcr",
        "emg_zcr",
        "eeg_peak_count",
        "emg_peak_count",
        "eeg_dominant_frequency",
        "eeg_spectral_entropy",
    ]
    ordered = [name for name in priority if name in columns]
    ordered.extend(name for name in columns if name not in ordered)
    return ordered[:max_features]
