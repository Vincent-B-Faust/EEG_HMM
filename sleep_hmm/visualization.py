from __future__ import annotations

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
