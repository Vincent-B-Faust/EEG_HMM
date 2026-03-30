from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SignalBundle:
    eeg: np.ndarray
    emg: np.ndarray | None
    fs: float
    time: np.ndarray


@dataclass
class PreprocessResult:
    raw: SignalBundle
    filtered: SignalBundle


@dataclass
class WindowResult:
    eeg_windows: np.ndarray
    emg_windows: np.ndarray | None
    start_indices: np.ndarray
    start_times: np.ndarray
    window_sec: float
    overlap_sec: float


@dataclass
class FeatureResult:
    raw_table: pd.DataFrame
    scaled_table: pd.DataFrame
    feature_names: list[str]
    scale_mean: np.ndarray
    scale_std: np.ndarray
    freqs: np.ndarray
    eeg_spectra: np.ndarray
    metadata: pd.DataFrame


@dataclass
class ClusterMethodResult:
    labels: np.ndarray
    metrics: dict[str, float]
    average_spectra: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterResult:
    methods: dict[str, ClusterMethodResult]


@dataclass
class AlignmentMethodResult:
    aligned_labels: np.ndarray
    mapping: dict[int, int]
    confusion_before: np.ndarray
    confusion_after: np.ndarray


@dataclass
class AlignmentResult:
    reference_method: str
    methods: dict[str, AlignmentMethodResult]


@dataclass
class HMMResult:
    n_states: int
    hidden_states: np.ndarray
    transition_matrix: np.ndarray
    initial_distribution: np.ndarray
    stationary_distribution: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihood: float
    bic: float
    durations_sec: dict[int, np.ndarray]
    run_lengths: dict[int, np.ndarray]


@dataclass
class ManifoldResult:
    method_requested: str
    method_used: str
    embedding: np.ndarray
    eigenvalues: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeNode:
    node_id: int
    depth: int
    class_counts: np.ndarray
    predicted_class: int
    impurity: float
    n_samples: int
    feature_index: int | None = None
    threshold_scaled: float | None = None
    threshold_raw: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class ExplainResult:
    tree_root: TreeNode
    feature_importance: pd.DataFrame
    thresholds: pd.DataFrame
    rules_text: str
    fidelity: float
    node_records: list[dict[str, Any]]


@dataclass
class PipelineResult:
    output_dir: Path
    preprocess: PreprocessResult
    windows: WindowResult
    features: FeatureResult
    clustering: ClusterResult
    alignment: AlignmentResult
    hmm: dict[int, HMMResult]
    manifold: ManifoldResult
    explain: dict[str, ExplainResult]
    artifact_paths: dict[str, list[Path]]
    runtime_info: dict[str, Any]
