from __future__ import annotations

import base64
import html
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from .types import ExplainResult, FeatureResult, HMMResult, PreprocessResult, TreeNode, WindowResult
from .utils import choose_feature_subset, ensure_dir


PALETTE = ["#0B4F6C", "#01BAEF", "#FBFBFF", "#B80C09", "#6D7278", "#F4B942"]


def _finalize(fig: plt.Figure, path: Path, dpi: int) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_signal_comparison(
    result: PreprocessResult,
    output_dir: Path,
    dpi: int,
    duration_sec: float = 20.0,
    prefix: str = "preprocess",
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    max_samples = int(duration_sec * result.raw.fs)
    time = result.raw.time[:max_samples]

    fig, axes = plt.subplots(2 if result.raw.emg is not None else 1, 1, figsize=(12, 6), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes[0].plot(time, result.raw.eeg[:max_samples], color="#B80C09", linewidth=0.8, label="Raw EEG")
    axes[0].plot(time, result.filtered.eeg[:max_samples], color="#0B4F6C", linewidth=0.8, label="Filtered EEG")
    axes[0].set_ylabel("EEG")
    axes[0].legend(loc="upper right")
    if result.raw.emg is not None and result.filtered.emg is not None:
        axes[1].plot(time, result.raw.emg[:max_samples], color="#F4B942", linewidth=0.8, label="Raw EMG")
        axes[1].plot(time, result.filtered.emg[:max_samples], color="#01BAEF", linewidth=0.8, label="Filtered EMG")
        axes[1].set_ylabel("EMG")
        axes[1].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    return [_finalize(fig, output_dir / f"{prefix}_signal_compare.png", dpi)]


def plot_window_boundaries(
    bundle: PreprocessResult,
    windows: WindowResult,
    output_dir: Path,
    dpi: int,
    prefix: str = "windowing",
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    max_samples = min(bundle.filtered.eeg.size, int(max(60.0, windows.window_sec * 3) * bundle.filtered.fs))
    time = bundle.filtered.time[:max_samples]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, bundle.filtered.eeg[:max_samples], color="#0B4F6C", linewidth=0.8)
    for start in windows.start_times:
        if start > time[-1]:
            break
        ax.axvline(start, color="#B80C09", linewidth=0.9, alpha=0.7)
    ax.set_title("Signal with window boundaries")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filtered EEG")
    return [_finalize(fig, output_dir / f"{prefix}_boundaries.png", dpi)]


def plot_feature_overview(
    features: FeatureResult,
    output_dir: Path,
    dpi: int,
    prefix: str = "features",
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []

    if features.eeg_spectra.size:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(features.freqs, features.eeg_spectra[0], color="#0B4F6C")
        ax.set_title("Single-window EEG spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        paths.append(_finalize(fig, output_dir / f"{prefix}_single_spectrum.png", dpi))

        fig, ax = plt.subplots(figsize=(10, 5))
        mesh = ax.imshow(features.eeg_spectra.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_yticks(np.linspace(0, len(features.freqs) - 1, min(6, len(features.freqs)), dtype=int))
        ax.set_yticklabels([f"{features.freqs[idx]:.1f}" for idx in ax.get_yticks().astype(int)])
        ax.set_xlabel("Window index")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("EEG spectral heatmap")
        fig.colorbar(mesh, ax=ax, label="Power")
        paths.append(_finalize(fig, output_dir / f"{prefix}_spectrum_heatmap.png", dpi))

    subset = choose_feature_subset(features.feature_names, 8)
    fig, axes = plt.subplots(len(subset), 1, figsize=(10, 2.2 * len(subset)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, feature_name in zip(axes, subset, strict=True):
        ax.hist(features.raw_table[feature_name], bins=30, color="#01BAEF", alpha=0.85)
        ax.set_title(feature_name)
        ax.set_ylabel("Count")
    axes[-1].set_xlabel("Feature value")
    paths.append(_finalize(fig, output_dir / f"{prefix}_distributions.png", dpi))
    return paths


def plot_cluster_outputs(
    method_name: str,
    labels: np.ndarray,
    metrics: dict[str, float],
    features: FeatureResult,
    average_spectra: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []
    time_axis = features.metadata["start_time_sec"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(time_axis, labels, where="post", color="#0B4F6C")
    ax.set_title(f"{method_name} cluster timeline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cluster")
    paths.append(_finalize(fig, output_dir / f"{method_name}_timeline.png", dpi))

    subset = choose_feature_subset(features.feature_names, 6)
    fig, axes = plt.subplots(len(subset), 1, figsize=(10, 2.2 * len(subset)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, feature_name in zip(axes, subset, strict=True):
        grouped = [features.raw_table.loc[labels == state, feature_name].to_numpy(dtype=float) for state in np.unique(labels)]
        ax.boxplot(grouped, tick_labels=[f"S{state}" for state in np.unique(labels)])
        ax.set_title(feature_name)
    axes[-1].set_xlabel("Cluster")
    paths.append(_finalize(fig, output_dir / f"{method_name}_feature_boxplots.png", dpi))

    fig, ax = plt.subplots(figsize=(8, 4))
    for state, spectrum in enumerate(average_spectra):
        ax.plot(features.freqs, spectrum, linewidth=1.5, label=f"S{state}")
    ax.set_title(f"{method_name} average spectra")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend(loc="upper right")
    paths.append(_finalize(fig, output_dir / f"{method_name}_average_spectra.png", dpi))

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(metrics)
    values = [metrics[key] for key in keys]
    ax.bar(range(len(keys)), values, color=["#0B4F6C", "#01BAEF", "#F4B942", "#B80C09"][: len(keys)])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_title(f"{method_name} quality metrics")
    paths.append(_finalize(fig, output_dir / f"{method_name}_quality.png", dpi))
    return paths


def plot_alignment_matrices(
    method_name: str,
    confusion_before: np.ndarray,
    confusion_after: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []
    for label, matrix in [("before", confusion_before), ("after", confusion_after)]:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        mesh = ax.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Candidate label")
        ax.set_ylabel("Reference label")
        ax.set_title(f"{method_name} confusion {label} alignment")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
        paths.append(_finalize(fig, output_dir / f"{method_name}_alignment_{label}.png", dpi))
    return paths


def plot_hmm_outputs(
    model_name: str,
    hmm_result: HMMResult,
    time_axis: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(time_axis, hmm_result.hidden_states, where="post", color="#0B4F6C")
    ax.set_title(f"{model_name} hidden states")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hidden state")
    paths.append(_finalize(fig, output_dir / f"{model_name}_hidden_states.png", dpi))

    fig, ax = plt.subplots(figsize=(5, 4))
    mesh = ax.imshow(hmm_result.transition_matrix, cmap="magma", vmin=0, vmax=1)
    for i in range(hmm_result.transition_matrix.shape[0]):
        for j in range(hmm_result.transition_matrix.shape[1]):
            ax.text(j, i, f"{hmm_result.transition_matrix[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title(f"{model_name} transition matrix")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    paths.append(_finalize(fig, output_dir / f"{model_name}_transition_heatmap.png", dpi))

    n_states = hmm_result.transition_matrix.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    positions = np.c_[np.cos(angles), np.sin(angles)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=1200, c="#0B4F6C", alpha=0.9)
    for state, (x, y) in enumerate(positions):
        ax.text(x, y, f"S{state}", color="white", ha="center", va="center", fontsize=12)
    for i in range(n_states):
        for j in range(n_states):
            prob = hmm_result.transition_matrix[i, j]
            if prob < 0.05:
                continue
            start = positions[i]
            end = positions[j]
            if i == j:
                ax.add_patch(plt.Circle((start[0], start[1] + 0.14), 0.14, fill=False, color="#B80C09", linewidth=1 + 3 * prob))
            else:
                arrow = FancyArrowPatch(
                    posA=(start[0], start[1]),
                    posB=(end[0], end[1]),
                    arrowstyle="->",
                    mutation_scale=12,
                    linewidth=1 + 4 * prob,
                    color="#B80C09",
                    alpha=0.7,
                    connectionstyle="arc3,rad=0.15",
                )
                ax.add_patch(arrow)
    ax.set_title(f"{model_name} state transition graph")
    ax.axis("off")
    paths.append(_finalize(fig, output_dir / f"{model_name}_transition_graph.png", dpi))

    fig, axes = plt.subplots(n_states, 1, figsize=(8, 2.2 * n_states))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for state, ax in enumerate(axes):
        durations = hmm_result.durations_sec.get(state, np.array([], dtype=float))
        ax.hist(durations, bins=20, color="#F4B942", alpha=0.85)
        ax.set_title(f"State {state} duration")
        ax.set_ylabel("Count")
    axes[-1].set_xlabel("Duration (s)")
    paths.append(_finalize(fig, output_dir / f"{model_name}_durations.png", dpi))
    return paths


def plot_hmm_model_comparison(
    comparison_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(comparison_frame["n_states"], comparison_frame["log_likelihood"], marker="o", color="#0B4F6C")
    axes[0].set_title("Gaussian HMM log likelihood")
    axes[0].set_xlabel("Number of states")
    axes[0].set_ylabel("Log likelihood")

    axes[1].plot(comparison_frame["n_states"], comparison_frame["bic"], marker="o", color="#B80C09")
    axes[1].set_title("Gaussian HMM BIC")
    axes[1].set_xlabel("Number of states")
    axes[1].set_ylabel("BIC")
    paths.append(_finalize(fig, output_dir / "hmm_model_comparison.png", dpi))
    return paths


def plot_manifold_outputs(
    embedding: np.ndarray,
    labels_by_method: dict[str, np.ndarray],
    method_used: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []
    for method_name, labels in labels_by_method.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        for state in np.unique(labels):
            mask = labels == state
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=18, alpha=0.8, label=f"S{state}")
        ax.set_title(f"{method_name} manifold scatter ({method_used})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="best")
        paths.append(_finalize(fig, output_dir / f"{method_name}_manifold_scatter.png", dpi))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(embedding[:, 0], embedding[:, 1], color="#6D7278", linewidth=0.8, alpha=0.7)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=16, cmap="tab10")
        ax.set_title(f"{method_name} manifold trajectory")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        paths.append(_finalize(fig, output_dir / f"{method_name}_manifold_trajectory.png", dpi))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=10, cmap="tab10", alpha=0.6)
        stride = max(len(embedding) // 120, 1)
        for start in range(0, len(embedding) - stride, stride):
            stop = start + stride
            ax.annotate(
                "",
                xy=(embedding[stop, 0], embedding[stop, 1]),
                xytext=(embedding[start, 0], embedding[start, 1]),
                arrowprops={"arrowstyle": "->", "color": "#B80C09", "alpha": 0.4, "linewidth": 0.8},
            )
        ax.set_title(f"{method_name} transition arrows")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        paths.append(_finalize(fig, output_dir / f"{method_name}_manifold_arrows.png", dpi))

    fig, axes = plt.subplots(1, len(labels_by_method), figsize=(6 * len(labels_by_method), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (method_name, labels) in zip(axes, labels_by_method.items(), strict=True):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=16, cmap="tab10", alpha=0.8)
        ax.set_title(method_name)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    paths.append(_finalize(fig, output_dir / "manifold_method_comparison.png", dpi))
    return paths


def _tree_layout(node: TreeNode, x: float, y: float, width: float, positions: dict[int, tuple[float, float]]) -> None:
    positions[node.node_id] = (x, y)
    if node.left is not None:
        _tree_layout(node.left, x - width / 2.0, y - 1.0, width / 2.0, positions)
    if node.right is not None:
        _tree_layout(node.right, x + width / 2.0, y - 1.0, width / 2.0, positions)


def plot_explain_outputs(
    method_name: str,
    explain_result: ExplainResult,
    raw_table: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    top_features: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []

    positions: dict[int, tuple[float, float]] = {}
    _tree_layout(explain_result.tree_root, 0.0, 0.0, 4.0, positions)
    fig, ax = plt.subplots(figsize=(12, 7))
    for record in explain_result.node_records:
        x, y = positions[int(record["node_id"])]
        label = f"S{record['predicted_class']}\nN={record['n_samples']}"
        if record["feature"] is not None:
            label = f"{record['feature']}\n<= {record['threshold_raw']:.3f}\n{label}"
        ax.text(x, y, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.3", "fc": "#FBFBFF", "ec": "#0B4F6C"})
    def _draw_edges(node: TreeNode) -> None:
        if node.left is not None:
            x0, y0 = positions[node.node_id]
            x1, y1 = positions[node.left.node_id]
            ax.plot([x0, x1], [y0, y1], color="#6D7278")
            _draw_edges(node.left)
        if node.right is not None:
            x0, y0 = positions[node.node_id]
            x1, y1 = positions[node.right.node_id]
            ax.plot([x0, x1], [y0, y1], color="#6D7278")
            _draw_edges(node.right)
    _draw_edges(explain_result.tree_root)
    ax.set_title(f"{method_name} decision tree")
    ax.axis("off")
    paths.append(_finalize(fig, output_dir / f"{method_name}_decision_tree.png", dpi))

    pivot = explain_result.thresholds.pivot(index="cluster", columns="feature", values="threshold_raw")
    feature_subset = list(explain_result.feature_importance.head(top_features)["feature"])
    pivot = pivot.loc[:, [feature for feature in feature_subset if feature in pivot.columns]]
    fig, ax = plt.subplots(figsize=(1.6 * max(len(pivot.columns), 1), 1.2 * max(len(pivot.index), 1) + 2))
    mesh = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"S{idx}" for idx in pivot.index])
    ax.set_title(f"{method_name} threshold heatmap")
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    paths.append(_finalize(fig, output_dir / f"{method_name}_threshold_heatmap.png", dpi))

    fig, ax = plt.subplots(figsize=(8, 4))
    importance = explain_result.feature_importance.head(top_features)
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#0B4F6C")
    ax.set_title(f"{method_name} feature importance")
    ax.set_xlabel("Importance")
    paths.append(_finalize(fig, output_dir / f"{method_name}_feature_importance.png", dpi))

    selected = list(importance["feature"])
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.2 * len(selected)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, feature_name in zip(axes, selected, strict=True):
        values = raw_table[feature_name].to_numpy(dtype=float)
        ax.hist(values, bins=30, color="#01BAEF", alpha=0.7)
        thresholds = explain_result.thresholds.loc[explain_result.thresholds["feature"] == feature_name]
        for _, row in thresholds.iterrows():
            ax.axvline(row["threshold_raw"], linestyle="--", linewidth=1.0, label=f"S{int(row['cluster'])} {row['direction']}")
        ax.set_title(feature_name)
    if len(selected) > 0:
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Feature value")
    paths.append(_finalize(fig, output_dir / f"{method_name}_threshold_overlays.png", dpi))
    return paths


def plot_comparison_outputs(
    quality_frame: pd.DataFrame,
    transition_matrices: dict[str, np.ndarray],
    embedding: np.ndarray,
    labels_by_method: dict[str, np.ndarray],
    threshold_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(quality_frame))
    width = 0.18
    metrics = [column for column in quality_frame.columns if column != "method"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + idx * width, quality_frame[metric], width=width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2.0)
    ax.set_xticklabels(quality_frame["method"])
    ax.set_title("Method quality comparison")
    ax.legend(loc="best")
    paths.append(_finalize(fig, output_dir / "comparison_quality.png", dpi))

    fig, axes = plt.subplots(1, len(transition_matrices), figsize=(5 * len(transition_matrices), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (method_name, matrix) in zip(axes, transition_matrices.items(), strict=True):
        mesh = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1)
        ax.set_title(method_name)
        ax.set_xlabel("Next")
        ax.set_ylabel("Current")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7)
    paths.append(_finalize(fig, output_dir / "comparison_transitions.png", dpi))

    fig, axes = plt.subplots(1, len(labels_by_method), figsize=(6 * len(labels_by_method), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (method_name, labels) in zip(axes, labels_by_method.items(), strict=True):
        ax.plot(embedding[:, 0], embedding[:, 1], color="#6D7278", linewidth=0.8, alpha=0.6)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=15, cmap="tab10")
        ax.set_title(method_name)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    paths.append(_finalize(fig, output_dir / "comparison_manifold.png", dpi))

    if not threshold_summary.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        methods = threshold_summary["method"].unique().tolist()
        features = threshold_summary["feature"].unique().tolist()
        grid = np.full((len(methods), len(features)), np.nan)
        for _, row in threshold_summary.iterrows():
            i = methods.index(row["method"])
            j = features.index(row["feature"])
            grid[i, j] = row["mean_abs_threshold"]
        mesh = ax.imshow(grid, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=40, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title("Threshold comparison by method")
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        paths.append(_finalize(fig, output_dir / "comparison_thresholds.png", dpi))
    return paths


def _encode_int16_base64(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<i2").tobytes()).decode("ascii")


def _quantize_signal(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    scale = value_max - value_min
    if scale <= 1e-12:
        return np.zeros(values.shape, dtype=np.int16)
    normalized = (values - value_min) / scale
    quantized = np.round(normalized * 65535.0 - 32768.0)
    return np.clip(quantized, -32768, 32767).astype(np.int16)


def _signal_minmax(signal: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = int(np.ceil(signal.size / block_size))
    if n_blocks <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    padded_length = n_blocks * block_size
    if padded_length == signal.size:
        padded = signal
    else:
        padded = np.pad(signal, (0, padded_length - signal.size), mode="edge")
    reshaped = padded.reshape(n_blocks, block_size)
    return reshaped.min(axis=1), reshaped.max(axis=1)


def _build_signal_levels(signal: np.ndarray, max_bins: int = 180_000, n_levels: int = 4) -> dict[str, object]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        return {
            "sample_count": 0,
            "value_min": 0.0,
            "value_max": 0.0,
            "levels": [],
        }

    value_min = float(np.min(signal))
    value_max = float(np.max(signal))
    finest_block = max(1, int(np.ceil(signal.size / max_bins)))
    block_sizes: list[int] = []
    block_size = finest_block
    for _ in range(n_levels):
        if block_size in block_sizes:
            break
        block_sizes.append(block_size)
        if block_size >= signal.size:
            break
        block_size *= 4

    levels = []
    for block_size in block_sizes:
        mins, maxs = _signal_minmax(signal, block_size)
        levels.append(
            {
                "block_size_samples": int(block_size),
                "mins_b64": _encode_int16_base64(_quantize_signal(mins, value_min, value_max)),
                "maxs_b64": _encode_int16_base64(_quantize_signal(maxs, value_min, value_max)),
            }
        )

    return {
        "sample_count": int(signal.size),
        "value_min": value_min,
        "value_max": value_max,
        "levels": levels,
    }


def create_interactive_session_view(
    session_name: str,
    source_name: str,
    preprocess_result: PreprocessResult,
    windows: WindowResult,
    labels_by_method: dict[str, np.ndarray],
    output_dir: Path,
    default_method: str,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    all_labels = np.concatenate([labels.astype(int) for labels in labels_by_method.values()]) if labels_by_method else np.array([0], dtype=int)
    n_states = int(all_labels.max()) + 1 if all_labels.size else 1
    palette = [PALETTE[idx % len(PALETTE)] for idx in range(max(n_states, 1))]

    payload = {
        "session_name": session_name,
        "source_name": source_name,
        "fs": float(preprocess_result.raw.fs),
        "duration_sec": float(preprocess_result.raw.eeg.size / preprocess_result.raw.fs) if preprocess_result.raw.eeg.size else 0.0,
        "sample_count": int(preprocess_result.raw.eeg.size),
        "window_sec": float(windows.window_sec),
        "epoch_starts_sec": windows.start_times.astype(float).round(6).tolist(),
        "epoch_centers_sec": (windows.start_times.astype(float) + windows.window_sec / 2.0).round(6).tolist(),
        "method_order": list(labels_by_method.keys()),
        "default_background_method": default_method if default_method in labels_by_method else next(iter(labels_by_method), "kmeans"),
        "cluster_colors": palette,
        "methods": {name: labels.astype(int).tolist() for name, labels in labels_by_method.items()},
        "signals": {
            "eeg": {
                "title": "EEG",
                **_build_signal_levels(preprocess_result.raw.eeg),
            },
            "emg": None
            if preprocess_result.raw.emg is None
            else {
                "title": "EMG",
                **_build_signal_levels(preprocess_result.raw.emg),
            },
        },
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #10212b;
      --muted: #5f6b76;
      --border: #d6dde3;
      --accent: #0b4f6c;
      --accent-2: #b80c09;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #eff5f6 0%, var(--bg) 45%, #fbfaf4 100%);
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .page {
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px 18px 28px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .title h1 {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
    }
    .title p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    .toolbar, .info-card, .panel {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 24px rgba(16, 33, 43, 0.06);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
      padding: 12px 14px;
      margin-bottom: 12px;
    }
    .toolbar button, .toolbar select {
      border: 1px solid #bfd0d8;
      border-radius: 10px;
      background: white;
      padding: 8px 12px;
      color: var(--ink);
      font-size: 13px;
      cursor: pointer;
    }
    .toolbar .range {
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      gap: 14px;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
    .panel {
      padding: 10px;
    }
    .panel canvas {
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #fffdfa;
    }
    .panel-label {
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .info-card {
      padding: 14px;
      align-self: start;
      position: sticky;
      top: 18px;
    }
    .info-card h2 {
      margin: 0 0 10px;
      font-size: 18px;
    }
    .info-card dl {
      margin: 0;
      display: grid;
      gap: 10px;
    }
    .info-card dt {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .info-card dd {
      margin: 3px 0 0;
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }
    .method-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
    }
    .method-table th, .method-table td {
      padding: 6px 8px;
      border-bottom: 1px solid #e5eaee;
      text-align: left;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .swatch {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
    }
    .hint {
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }
    @media (max-width: 1100px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .info-card {
        position: static;
      }
      .toolbar .range {
        margin-left: 0;
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="title">
        <h1 id="session-title">Interactive EEG/EMG Session</h1>
        <p id="session-subtitle"></p>
      </div>
    </div>

    <div class="toolbar">
      <button id="zoom-in">Zoom In</button>
      <button id="zoom-out">Zoom Out</button>
      <button id="pan-left">Pan Left</button>
      <button id="pan-right">Pan Right</button>
      <button id="show-full">Show Full</button>
      <button id="go-selected">Center Selected Epoch</button>
      <label for="method-select">Background Cluster</label>
      <select id="method-select"></select>
      <div class="range" id="range-label"></div>
    </div>

    <div class="grid">
      <div class="stack">
        <div class="panel">
          <p class="panel-label">Overview</p>
          <canvas id="overview-canvas" height="110"></canvas>
        </div>
        <div class="panel">
          <p class="panel-label">EEG</p>
          <canvas id="eeg-canvas" height="230"></canvas>
        </div>
        <div class="panel" id="emg-panel">
          <p class="panel-label">EMG</p>
          <canvas id="emg-canvas" height="200"></canvas>
        </div>
        <div class="panel">
          <p class="panel-label">Cluster Timeline</p>
          <canvas id="cluster-canvas" height="210"></canvas>
        </div>
      </div>

      <aside class="info-card">
        <h2>Epoch Traceback</h2>
        <dl>
          <div>
            <dt>Session</dt>
            <dd id="info-session"></dd>
          </div>
          <div>
            <dt>Selected Epoch</dt>
            <dd id="info-epoch"></dd>
          </div>
          <div>
            <dt>Epoch Range</dt>
            <dd id="info-range"></dd>
          </div>
          <div>
            <dt>Current View</dt>
            <dd id="info-view"></dd>
          </div>
        </dl>
        <table class="method-table">
          <thead>
            <tr><th>Method</th><th>State</th></tr>
          </thead>
          <tbody id="method-state-body"></tbody>
        </table>
        <div class="hint">
          Click an epoch on the cluster timeline to trace it back to the raw EEG/EMG panels. Mouse wheel zooms around the cursor position, and dragging inside EEG/EMG or cluster panels pans the time range synchronously.
        </div>
      </aside>
    </div>
  </div>

  <script>
    const DATA = __SESSION_DATA__;

    function decodeInt16(base64Text) {
      const binary = atob(base64Text);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      return new Int16Array(bytes.buffer);
    }

    function hexToRgba(hex, alpha) {
      const clean = hex.replace("#", "");
      const r = parseInt(clean.slice(0, 2), 16);
      const g = parseInt(clean.slice(2, 4), 16);
      const b = parseInt(clean.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function formatTime(totalSeconds) {
      const seconds = Math.max(0, Number(totalSeconds) || 0);
      const hh = Math.floor(seconds / 3600);
      const mm = Math.floor((seconds % 3600) / 60);
      const ss = seconds % 60;
      return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${ss.toFixed(1).padStart(4, "0")}`;
    }

    function lowerBound(values, target) {
      let low = 0;
      let high = values.length;
      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (values[mid] < target) low = mid + 1;
        else high = mid;
      }
      return low;
    }

    function nearestEpochIndex(sec) {
      if (!DATA.epoch_centers_sec.length) return 0;
      const idx = lowerBound(DATA.epoch_centers_sec, sec);
      if (idx <= 0) return 0;
      if (idx >= DATA.epoch_centers_sec.length) return DATA.epoch_centers_sec.length - 1;
      const left = DATA.epoch_centers_sec[idx - 1];
      const right = DATA.epoch_centers_sec[idx];
      return Math.abs(sec - left) <= Math.abs(sec - right) ? idx - 1 : idx;
    }

    function getLabels(methodName) {
      return DATA.methods[methodName] || DATA.methods[DATA.method_order[0]];
    }

    function dequantize(q, signal) {
      const span = signal.value_max - signal.value_min;
      if (Math.abs(span) < 1e-12) return signal.value_min;
      return signal.value_min + ((q + 32768) / 65535) * span;
    }

    function decodeSignalLevels(signal) {
      if (!signal) return null;
      signal.levels.forEach((level) => {
        level.mins = decodeInt16(level.mins_b64);
        level.maxs = decodeInt16(level.maxs_b64);
        delete level.mins_b64;
        delete level.maxs_b64;
      });
      return signal;
    }

    DATA.signals.eeg = decodeSignalLevels(DATA.signals.eeg);
    DATA.signals.emg = decodeSignalLevels(DATA.signals.emg);

    const defaultSpan = Math.min(DATA.duration_sec || DATA.window_sec, Math.max(600, DATA.window_sec * 20));
    const state = {
      backgroundMethod: DATA.default_background_method,
      selectedEpoch: 0,
      viewStartSec: 0,
      viewEndSec: Math.min(DATA.duration_sec, defaultSpan),
      isDragging: false,
      dragOriginX: 0,
      dragViewStart: 0,
      dragViewEnd: 0,
      dragCanvas: null,
    };

    const canvases = {
      overview: document.getElementById("overview-canvas"),
      eeg: document.getElementById("eeg-canvas"),
      emg: document.getElementById("emg-canvas"),
      cluster: document.getElementById("cluster-canvas"),
    };

    const emgPanel = document.getElementById("emg-panel");
    if (!DATA.signals.emg) emgPanel.style.display = "none";

    document.getElementById("session-title").textContent = `Interactive Session: ${DATA.session_name}`;
    document.getElementById("session-subtitle").textContent = `${DATA.source_name} | Duration ${formatTime(DATA.duration_sec)} | Window ${DATA.window_sec.toFixed(1)} s`;
    document.getElementById("info-session").textContent = DATA.session_name;

    const methodSelect = document.getElementById("method-select");
    DATA.method_order.forEach((methodName) => {
      const option = document.createElement("option");
      option.value = methodName;
      option.textContent = methodName;
      if (methodName === state.backgroundMethod) option.selected = true;
      methodSelect.appendChild(option);
    });
    methodSelect.addEventListener("change", (event) => {
      state.backgroundMethod = event.target.value;
      renderAll();
    });

    function setupCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const bounds = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
      canvas.height = Math.max(1, Math.floor((canvas.getAttribute("height") || bounds.height || 200) * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, width: bounds.width, height: canvas.height / ratio };
    }

    function clampView(startSec, endSec) {
      const minSpan = Math.max(DATA.window_sec * 1.2, Math.min(DATA.duration_sec, 10));
      const duration = Math.max(DATA.duration_sec, minSpan);
      let span = Math.max(minSpan, endSec - startSec);
      if (span > duration) span = duration;
      let start = startSec;
      let end = start + span;
      if (start < 0) {
        start = 0;
        end = span;
      }
      if (end > duration) {
        end = duration;
        start = Math.max(0, end - span);
      }
      state.viewStartSec = start;
      state.viewEndSec = end;
    }

    function setView(startSec, endSec) {
      clampView(startSec, endSec);
      renderAll();
    }

    function zoom(factor, centerSec) {
      const center = Number.isFinite(centerSec) ? centerSec : (state.viewStartSec + state.viewEndSec) / 2;
      const span = state.viewEndSec - state.viewStartSec;
      const newSpan = span * factor;
      const ratio = span > 0 ? (center - state.viewStartSec) / span : 0.5;
      setView(center - newSpan * ratio, center + newSpan * (1 - ratio));
    }

    function pan(fraction) {
      const span = state.viewEndSec - state.viewStartSec;
      const shift = span * fraction;
      setView(state.viewStartSec + shift, state.viewEndSec + shift);
    }

    function ensureEpochVisible(epochIndex) {
      const start = DATA.epoch_starts_sec[epochIndex] || 0;
      const end = start + DATA.window_sec;
      const span = state.viewEndSec - state.viewStartSec;
      if (start >= state.viewStartSec && end <= state.viewEndSec) return;
      const center = (start + end) / 2;
      setView(center - span / 2, center + span / 2);
    }

    function selectEpoch(epochIndex, keepView = false) {
      state.selectedEpoch = Math.max(0, Math.min(DATA.epoch_starts_sec.length - 1, epochIndex));
      if (!keepView) ensureEpochVisible(state.selectedEpoch);
      else renderAll();
    }

    function currentPlotRect(width, height, leftMargin) {
      return {
        left: leftMargin,
        right: width - 12,
        top: 14,
        bottom: height - 22,
        width: Math.max(10, width - leftMargin - 12),
        height: Math.max(10, height - 36),
      };
    }

    function drawEpochBackgrounds(ctx, plot, labels) {
      const startIdx = Math.max(0, lowerBound(DATA.epoch_starts_sec, state.viewStartSec + 1e-9) - 1);
      const endIdx = Math.min(DATA.epoch_starts_sec.length - 1, lowerBound(DATA.epoch_starts_sec, state.viewEndSec + DATA.window_sec));
      for (let idx = startIdx; idx <= endIdx; idx += 1) {
        const epochStart = DATA.epoch_starts_sec[idx];
        const epochEnd = epochStart + DATA.window_sec;
        const x0 = plot.left + ((epochStart - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
        const x1 = plot.left + ((epochEnd - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
        ctx.fillStyle = hexToRgba(DATA.cluster_colors[labels[idx] % DATA.cluster_colors.length], idx === state.selectedEpoch ? 0.22 : 0.11);
        ctx.fillRect(x0, plot.top, Math.max(1, x1 - x0), plot.height);
      }
      if (state.selectedEpoch < DATA.epoch_starts_sec.length) {
        const selectedStart = DATA.epoch_starts_sec[state.selectedEpoch];
        const selectedEnd = selectedStart + DATA.window_sec;
        const x0 = plot.left + ((selectedStart - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
        const x1 = plot.left + ((selectedEnd - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
        ctx.strokeStyle = hexToRgba("#b80c09", 0.9);
        ctx.lineWidth = 2;
        ctx.strokeRect(x0, plot.top + 1, Math.max(1, x1 - x0), plot.height - 2);
      }
    }

    function pickSignalLevel(signal, pixelWidth) {
      const visibleSamples = Math.max(1, Math.ceil((state.viewEndSec - state.viewStartSec) * DATA.fs));
      const targetBars = Math.max(250, Math.floor(pixelWidth * 1.35));
      for (const level of signal.levels) {
        const visibleBins = Math.ceil(visibleSamples / level.block_size_samples);
        if (visibleBins <= targetBars) return level;
      }
      return signal.levels[signal.levels.length - 1];
    }

    function drawSignalPanel(canvas, signal) {
      const { ctx, width, height } = setupCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      const plot = currentPlotRect(width, height, 62);
      const labels = getLabels(state.backgroundMethod);
      drawEpochBackgrounds(ctx, plot, labels);

      const level = pickSignalLevel(signal, plot.width);
      const startSample = Math.max(0, Math.floor(state.viewStartSec * DATA.fs));
      const endSample = Math.min(signal.sample_count, Math.ceil(state.viewEndSec * DATA.fs));
      const firstBlock = Math.max(0, Math.floor(startSample / level.block_size_samples));
      const lastBlock = Math.min(level.mins.length - 1, Math.ceil(endSample / level.block_size_samples));

      let visibleMin = Number.POSITIVE_INFINITY;
      let visibleMax = Number.NEGATIVE_INFINITY;
      for (let idx = firstBlock; idx <= lastBlock; idx += 1) {
        const localMin = dequantize(level.mins[idx], signal);
        const localMax = dequantize(level.maxs[idx], signal);
        if (localMin < visibleMin) visibleMin = localMin;
        if (localMax > visibleMax) visibleMax = localMax;
      }
      if (!Number.isFinite(visibleMin) || !Number.isFinite(visibleMax)) {
        visibleMin = signal.value_min;
        visibleMax = signal.value_max;
      }
      const rawSpan = Math.max(visibleMax - visibleMin, 1e-6);
      visibleMin -= rawSpan * 0.05;
      visibleMax += rawSpan * 0.05;

      ctx.strokeStyle = "#0f1720";
      ctx.lineWidth = 1;
      for (let idx = firstBlock; idx <= lastBlock; idx += 1) {
        const blockStart = idx * level.block_size_samples;
        const blockEnd = Math.min(signal.sample_count, blockStart + level.block_size_samples);
        const centerSec = ((blockStart + blockEnd) * 0.5) / DATA.fs;
        const x = plot.left + ((centerSec - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
        const yMin = plot.bottom - ((dequantize(level.mins[idx], signal) - visibleMin) / (visibleMax - visibleMin)) * plot.height;
        const yMax = plot.bottom - ((dequantize(level.maxs[idx], signal) - visibleMin) / (visibleMax - visibleMin)) * plot.height;
        ctx.beginPath();
        ctx.moveTo(x, yMin);
        ctx.lineTo(x, yMax);
        ctx.stroke();
      }

      if (visibleMin < 0 && visibleMax > 0) {
        const zeroY = plot.bottom - ((0 - visibleMin) / (visibleMax - visibleMin)) * plot.height;
        ctx.strokeStyle = "rgba(11,79,108,0.3)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plot.left, zeroY);
        ctx.lineTo(plot.right, zeroY);
        ctx.stroke();
      }

      ctx.fillStyle = "#10212b";
      ctx.font = "12px Segoe UI";
      ctx.fillText(signal.title, plot.left, 12);
      ctx.fillStyle = "#5f6b76";
      ctx.fillText(`Visible amplitude: ${visibleMin.toFixed(2)} to ${visibleMax.toFixed(2)}`, plot.left + 80, 12);
      ctx.fillText(formatTime(state.viewStartSec), plot.left, height - 4);
      ctx.fillText(formatTime(state.viewEndSec), Math.max(plot.left, plot.right - 68), height - 4);
    }

    function drawClusterPanel() {
      const { ctx, width, height } = setupCanvas(canvases.cluster);
      ctx.clearRect(0, 0, width, height);
      const left = 92;
      const plot = currentPlotRect(width, height, left);
      const rowHeight = plot.height / Math.max(DATA.method_order.length, 1);
      const selectedX = plot.left + ((DATA.epoch_centers_sec[state.selectedEpoch] - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;

      ctx.strokeStyle = "rgba(11,79,108,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(selectedX, plot.top);
      ctx.lineTo(selectedX, plot.bottom);
      ctx.stroke();

      const startIdx = Math.max(0, lowerBound(DATA.epoch_centers_sec, state.viewStartSec) - 1);
      const endIdx = Math.min(DATA.epoch_centers_sec.length - 1, lowerBound(DATA.epoch_centers_sec, state.viewEndSec + DATA.window_sec));

      DATA.method_order.forEach((methodName, rowIndex) => {
        const yCenter = plot.top + rowHeight * (rowIndex + 0.5);
        ctx.fillStyle = methodName === state.backgroundMethod ? "#10212b" : "#5f6b76";
        ctx.font = methodName === state.backgroundMethod ? "bold 12px Segoe UI" : "12px Segoe UI";
        ctx.fillText(methodName, 12, yCenter + 4);

        ctx.strokeStyle = "rgba(95,107,118,0.15)";
        ctx.beginPath();
        ctx.moveTo(plot.left, yCenter);
        ctx.lineTo(plot.right, yCenter);
        ctx.stroke();

        const labels = DATA.methods[methodName];
        for (let idx = startIdx; idx <= endIdx; idx += 1) {
          const sec = DATA.epoch_centers_sec[idx];
          if (sec < state.viewStartSec - DATA.window_sec || sec > state.viewEndSec + DATA.window_sec) continue;
          const x = plot.left + ((sec - state.viewStartSec) / (state.viewEndSec - state.viewStartSec)) * plot.width;
          const color = DATA.cluster_colors[labels[idx] % DATA.cluster_colors.length];
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, yCenter, idx === state.selectedEpoch ? 5.5 : 4, 0, Math.PI * 2);
          ctx.fill();
          if (idx === state.selectedEpoch) {
            ctx.strokeStyle = "#b80c09";
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      });
      ctx.fillStyle = "#5f6b76";
      ctx.font = "12px Segoe UI";
      ctx.fillText(formatTime(state.viewStartSec), plot.left, height - 4);
      ctx.fillText(formatTime(state.viewEndSec), Math.max(plot.left, plot.right - 68), height - 4);
    }

    function drawOverview() {
      const { ctx, width, height } = setupCanvas(canvases.overview);
      ctx.clearRect(0, 0, width, height);
      const plot = currentPlotRect(width, height, 12);
      const labels = getLabels(state.backgroundMethod);
      for (let idx = 0; idx < DATA.epoch_starts_sec.length; idx += 1) {
        const x0 = plot.left + (DATA.epoch_starts_sec[idx] / DATA.duration_sec) * plot.width;
        const x1 = plot.left + ((DATA.epoch_starts_sec[idx] + DATA.window_sec) / DATA.duration_sec) * plot.width;
        ctx.fillStyle = hexToRgba(DATA.cluster_colors[labels[idx] % DATA.cluster_colors.length], 0.55);
        ctx.fillRect(x0, plot.top + 10, Math.max(1, x1 - x0), plot.height - 20);
      }
      const selectedStart = DATA.epoch_starts_sec[state.selectedEpoch];
      const selectedCenterX = plot.left + ((selectedStart + DATA.window_sec / 2) / DATA.duration_sec) * plot.width;
      ctx.strokeStyle = "#b80c09";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(selectedCenterX, plot.top);
      ctx.lineTo(selectedCenterX, plot.bottom);
      ctx.stroke();

      const viewX0 = plot.left + (state.viewStartSec / DATA.duration_sec) * plot.width;
      const viewX1 = plot.left + (state.viewEndSec / DATA.duration_sec) * plot.width;
      ctx.strokeStyle = "rgba(16,33,43,0.92)";
      ctx.lineWidth = 2;
      ctx.strokeRect(viewX0, plot.top + 2, Math.max(2, viewX1 - viewX0), plot.height - 4);
      ctx.fillStyle = "#5f6b76";
      ctx.font = "12px Segoe UI";
      ctx.fillText("Click to center the visible window", 12, 12);
      ctx.fillText(formatTime(0), plot.left, height - 4);
      ctx.fillText(formatTime(DATA.duration_sec), Math.max(plot.left, plot.right - 74), height - 4);
    }

    function updateInfo() {
      const epochIdx = state.selectedEpoch;
      const epochStart = DATA.epoch_starts_sec[epochIdx] || 0;
      const epochEnd = epochStart + DATA.window_sec;
      document.getElementById("range-label").textContent = `View ${formatTime(state.viewStartSec)} - ${formatTime(state.viewEndSec)} | Span ${(state.viewEndSec - state.viewStartSec).toFixed(1)} s`;
      document.getElementById("info-epoch").textContent = `#${epochIdx}`;
      document.getElementById("info-range").textContent = `${formatTime(epochStart)} - ${formatTime(epochEnd)}`;
      document.getElementById("info-view").textContent = `${formatTime(state.viewStartSec)} - ${formatTime(state.viewEndSec)}`;
      const body = document.getElementById("method-state-body");
      body.innerHTML = "";
      DATA.method_order.forEach((methodName) => {
        const row = document.createElement("tr");
        const stateCell = document.createElement("td");
        const methodCell = document.createElement("td");
        methodCell.textContent = methodName;
        const badge = document.createElement("span");
        badge.className = "badge";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        const clusterState = DATA.methods[methodName][epochIdx];
        swatch.style.background = DATA.cluster_colors[clusterState % DATA.cluster_colors.length];
        const label = document.createElement("span");
        label.textContent = `S${clusterState}`;
        badge.appendChild(swatch);
        badge.appendChild(label);
        stateCell.appendChild(badge);
        row.appendChild(methodCell);
        row.appendChild(stateCell);
        body.appendChild(row);
      });
    }

    function renderAll() {
      drawOverview();
      drawSignalPanel(canvases.eeg, DATA.signals.eeg);
      if (DATA.signals.emg) drawSignalPanel(canvases.emg, DATA.signals.emg);
      drawClusterPanel();
      updateInfo();
    }

    function timeFromCanvasEvent(canvas, event, leftMargin) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const plotWidth = Math.max(10, rect.width - leftMargin - 12);
      const clamped = Math.min(Math.max(leftMargin, x), leftMargin + plotWidth);
      const ratio = (clamped - leftMargin) / plotWidth;
      return state.viewStartSec + ratio * (state.viewEndSec - state.viewStartSec);
    }

    function attachPanZoom(canvas, leftMargin) {
      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        zoom(event.deltaY < 0 ? 0.75 : 1.35, timeFromCanvasEvent(canvas, event, leftMargin));
      }, { passive: false });

      canvas.addEventListener("mousedown", (event) => {
        state.isDragging = true;
        state.dragCanvas = canvas;
        state.dragOriginX = event.clientX;
        state.dragViewStart = state.viewStartSec;
        state.dragViewEnd = state.viewEndSec;
      });

      canvas.addEventListener("dblclick", (event) => {
        const sec = timeFromCanvasEvent(canvas, event, leftMargin);
        selectEpoch(nearestEpochIndex(sec));
      });
    }

    window.addEventListener("mousemove", (event) => {
      if (!state.isDragging || !state.dragCanvas) return;
      const rect = state.dragCanvas.getBoundingClientRect();
      const leftMargin = state.dragCanvas === canvases.cluster ? 92 : 62;
      const plotWidth = Math.max(10, rect.width - leftMargin - 12);
      const dx = event.clientX - state.dragOriginX;
      const shiftSec = -(dx / plotWidth) * (state.dragViewEnd - state.dragViewStart);
      clampView(state.dragViewStart + shiftSec, state.dragViewEnd + shiftSec);
      renderAll();
    });

    window.addEventListener("mouseup", () => {
      state.isDragging = false;
      state.dragCanvas = null;
    });

    canvases.cluster.addEventListener("click", (event) => {
      const sec = timeFromCanvasEvent(canvases.cluster, event, 92);
      selectEpoch(nearestEpochIndex(sec), true);
      ensureEpochVisible(state.selectedEpoch);
    });

    canvases.overview.addEventListener("click", (event) => {
      const rect = canvases.overview.getBoundingClientRect();
      const x = Math.min(Math.max(12, event.clientX - rect.left), rect.width - 12);
      const ratio = (x - 12) / Math.max(10, rect.width - 24);
      const center = ratio * DATA.duration_sec;
      const span = state.viewEndSec - state.viewStartSec;
      setView(center - span / 2, center + span / 2);
    });

    attachPanZoom(canvases.eeg, 62);
    if (DATA.signals.emg) attachPanZoom(canvases.emg, 62);
    attachPanZoom(canvases.cluster, 92);

    document.getElementById("zoom-in").addEventListener("click", () => zoom(0.5));
    document.getElementById("zoom-out").addEventListener("click", () => zoom(2.0));
    document.getElementById("pan-left").addEventListener("click", () => pan(-0.5));
    document.getElementById("pan-right").addEventListener("click", () => pan(0.5));
    document.getElementById("show-full").addEventListener("click", () => setView(0, DATA.duration_sec));
    document.getElementById("go-selected").addEventListener("click", () => {
      const center = DATA.epoch_centers_sec[state.selectedEpoch] || 0;
      const span = Math.max(DATA.window_sec * 3, Math.min(state.viewEndSec - state.viewStartSec, 600));
      setView(center - span / 2, center + span / 2);
    });

    window.addEventListener("resize", renderAll);
    renderAll();
  </script>
</body>
</html>
"""

    html_text = html_template.replace("__SESSION_DATA__", payload_json).replace("__TITLE__", html.escape(f"{session_name} Interactive Session"))
    html_path = output_dir / "session_view.html"
    html_path.write_text(html_text, encoding="utf-8")
    return [html_path]
