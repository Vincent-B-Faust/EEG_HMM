from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .acceleration import resolve_acceleration
from .alignment import align_cluster_labels
from .clustering import cluster
from .config import PipelineConfig
from .explainability import explain
from .features import extract_features, window_signals
from .hmm import hmm_analysis
from .preprocessing import preprocess
from .types import PipelineResult, SignalBundle
from .utils import ensure_dir, save_json
from .visualization import (
    plot_alignment_matrices,
    plot_cluster_outputs,
    plot_comparison_outputs,
    plot_explain_outputs,
    plot_feature_overview,
    plot_hmm_model_comparison,
    plot_hmm_outputs,
    plot_manifold_outputs,
    plot_signal_comparison,
    plot_window_boundaries,
)
from .manifold import manifold


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    config: PipelineConfig,
    cluster_metrics: pd.DataFrame,
    hmm_metrics: pd.DataFrame,
    explain_results: dict[str, object],
    manifold_method: str,
    runtime_info: dict[str, object],
) -> Path:
    lines = [
        "# Sleep Stage Unsupervised Analysis Report",
        "",
        "## Configuration",
        f"- Window length: {config.windowing.window_sec:.1f} s",
        f"- Overlap: {config.windowing.overlap_sec:.1f} s",
        f"- Number of clusters: {config.clustering.n_clusters}",
        f"- Manifold method requested: {config.manifold.method}",
        f"- Manifold method used: {manifold_method}",
        f"- Acceleration backend used: {runtime_info.get('backend_used', 'numpy')}",
        f"- Acceleration device used: {runtime_info.get('device_used', 'cpu')}",
        f"- Acceleration note: {runtime_info.get('reason', '')}",
        "",
        "## Clustering Summary",
        _markdown_table(cluster_metrics),
        "",
        "## Gaussian HMM Model Comparison",
        _markdown_table(hmm_metrics),
        "",
        "## Explainability",
    ]
    for method_name, explain_result in explain_results.items():
        lines.extend(
            [
                f"### {method_name}",
                f"- Decision-tree fidelity: {explain_result.fidelity:.3f}",
                "",
                "```text",
                explain_result.rules_text,
                "```",
                "",
            ]
        )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(
    eeg: np.ndarray,
    fs: float,
    emg: np.ndarray | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    output_dir = ensure_dir(cfg.output.output_dir)
    artifacts: dict[str, list[Path]] = {}
    runtime = resolve_acceleration(cfg.acceleration)

    preprocess_result = preprocess(eeg=eeg, emg=emg, fs=fs, config=cfg.filters)
    windows = window_signals(preprocess_result.filtered, cfg.windowing)
    feature_result = extract_features(windows, fs=fs, config=cfg.features, runtime=runtime)
    clustering = cluster(feature_result, cfg.clustering, runtime=runtime)
    alignment = align_cluster_labels(clustering, cfg.clustering)
    hmm_results = hmm_analysis(
        features=feature_result.scaled_table.to_numpy(dtype=float),
        window_sec=cfg.windowing.window_sec,
        config=cfg.hmm,
    )

    manifold_result = manifold(feature_result.scaled_table.to_numpy(dtype=float), cfg.manifold, runtime=runtime)

    explain_results = {}
    for method_name, alignment_result in alignment.methods.items():
        labels = alignment_result.aligned_labels
        explain_results[method_name] = explain(
            raw_table=feature_result.raw_table,
            scaled_table=feature_result.scaled_table,
            labels=labels,
            feature_names=feature_result.feature_names,
            scale_mean=feature_result.scale_mean,
            scale_std=feature_result.scale_std,
            config=cfg.explain,
        )

    save_json(cfg.to_dict(), output_dir / "config.json")
    save_json(runtime.info(), output_dir / "acceleration.json")
    feature_result.raw_table.to_csv(output_dir / "features_raw.csv", index=False)
    feature_result.scaled_table.to_csv(output_dir / "features_scaled.csv", index=False)
    feature_result.metadata.to_csv(output_dir / "window_metadata.csv", index=False)
    np.savez(
        output_dir / "spectra.npz",
        freqs=feature_result.freqs,
        eeg_spectra=feature_result.eeg_spectra,
    )

    cluster_rows: list[dict[str, float | str]] = []
    threshold_compare_rows: list[dict[str, float | str]] = []
    hmm_rows: list[dict[str, float | int]] = []
    for method_name, method_result in clustering.methods.items():
        aligned = alignment.methods[method_name].aligned_labels
        pd.DataFrame({"label": method_result.labels, "aligned_label": aligned}).to_csv(
            output_dir / f"{method_name}_labels.csv",
            index=False,
        )
        save_json(method_result.metrics, output_dir / f"{method_name}_metrics.json")
        np.savez(output_dir / f"{method_name}_average_spectra.npz", freqs=feature_result.freqs, spectra=method_result.average_spectra)
        cluster_rows.append(
            {
                "method": method_name,
                **method_result.metrics,
                "explain_fidelity": float(explain_results[method_name].fidelity),
            }
        )

        explain_result = explain_results[method_name]
        explain_result.feature_importance.to_csv(output_dir / f"{method_name}_feature_importance.csv", index=False)
        explain_result.thresholds.to_csv(output_dir / f"{method_name}_thresholds.csv", index=False)
        save_json(
            {
                "fidelity": explain_result.fidelity,
                "rules_text": explain_result.rules_text,
                "nodes": explain_result.node_records,
            },
            output_dir / f"{method_name}_explain.json",
        )

        top_thresholds = (
            explain_result.thresholds.groupby("feature", as_index=False)["threshold_raw"]
            .apply(lambda column: np.mean(np.abs(column)))
            .rename(columns={"threshold_raw": "mean_abs_threshold"})
        )
        top_thresholds["method"] = method_name
        threshold_compare_rows.extend(top_thresholds.to_dict(orient="records"))

    cluster_metrics = pd.DataFrame(cluster_rows)
    cluster_metrics.to_csv(output_dir / "method_comparison_metrics.csv", index=False)
    for n_states, hmm_result in hmm_results.items():
        model_name = f"gaussian_hmm_{n_states}_state"
        hmm_rows.append(
            {
                "n_states": int(n_states),
                "log_likelihood": float(hmm_result.log_likelihood),
                "bic": float(hmm_result.bic),
                "mean_self_transition": float(np.mean(np.diag(hmm_result.transition_matrix))),
            }
        )
        pd.DataFrame({"hidden_state": hmm_result.hidden_states}).to_csv(output_dir / f"{model_name}_hidden_states.csv", index=False)
        pd.DataFrame(hmm_result.transition_matrix).to_csv(output_dir / f"{model_name}_transition_matrix.csv", index=False)
        pd.DataFrame(hmm_result.means, columns=feature_result.feature_names).to_csv(output_dir / f"{model_name}_means.csv", index=False)
        pd.DataFrame(hmm_result.variances, columns=feature_result.feature_names).to_csv(output_dir / f"{model_name}_variances.csv", index=False)
        pd.DataFrame(
            {
                "state": np.repeat(list(hmm_result.durations_sec.keys()), [len(item) for item in hmm_result.durations_sec.values()]),
                "duration_sec": np.concatenate(list(hmm_result.durations_sec.values())) if hmm_result.durations_sec else np.array([]),
            }
        ).to_csv(output_dir / f"{model_name}_durations.csv", index=False)
        save_json(
            {
                "n_states": n_states,
                "log_likelihood": hmm_result.log_likelihood,
                "bic": hmm_result.bic,
                "initial_distribution": hmm_result.initial_distribution,
                "stationary_distribution": hmm_result.stationary_distribution,
            },
            output_dir / f"{model_name}_summary.json",
        )
    hmm_metrics = pd.DataFrame(hmm_rows).sort_values("n_states").reset_index(drop=True)
    hmm_metrics.to_csv(output_dir / "hmm_model_comparison.csv", index=False)
    manifold_frame = pd.DataFrame(manifold_result.embedding, columns=[f"dim_{idx+1}" for idx in range(manifold_result.embedding.shape[1])])
    manifold_frame.to_csv(output_dir / "manifold_embedding.csv", index=False)
    save_json(
        {
            "method_requested": manifold_result.method_requested,
            "method_used": manifold_result.method_used,
            "metadata": manifold_result.metadata,
            "eigenvalues": None if manifold_result.eigenvalues is None else manifold_result.eigenvalues.tolist(),
        },
        output_dir / "manifold_meta.json",
    )

    figures_dir = ensure_dir(output_dir / "figures")
    artifacts["figures"] = []
    artifacts["figures"] += plot_signal_comparison(preprocess_result, figures_dir, cfg.output.figure_dpi)
    artifacts["figures"] += plot_window_boundaries(preprocess_result, windows, figures_dir, cfg.output.figure_dpi)
    artifacts["figures"] += plot_feature_overview(feature_result, figures_dir, cfg.output.figure_dpi)
    time_axis = feature_result.metadata["start_time_sec"].to_numpy(dtype=float)
    for method_name, method_result in clustering.methods.items():
        artifacts["figures"] += plot_cluster_outputs(
            method_name=method_name,
            labels=alignment.methods[method_name].aligned_labels,
            metrics=method_result.metrics,
            features=feature_result,
            average_spectra=method_result.average_spectra,
            output_dir=figures_dir,
            dpi=cfg.output.figure_dpi,
        )
        artifacts["figures"] += plot_alignment_matrices(
            method_name=method_name,
            confusion_before=alignment.methods[method_name].confusion_before,
            confusion_after=alignment.methods[method_name].confusion_after,
            output_dir=figures_dir,
            dpi=cfg.output.figure_dpi,
        )
        artifacts["figures"] += plot_explain_outputs(
            method_name=method_name,
            explain_result=explain_results[method_name],
            raw_table=feature_result.raw_table,
            output_dir=figures_dir,
            dpi=cfg.output.figure_dpi,
            top_features=cfg.explain.top_features_to_plot,
        )
    for n_states, hmm_result in hmm_results.items():
        artifacts["figures"] += plot_hmm_outputs(
            model_name=f"gaussian_hmm_{n_states}_state",
            hmm_result=hmm_result,
            time_axis=time_axis,
            output_dir=figures_dir,
            dpi=cfg.output.figure_dpi,
        )
    artifacts["figures"] += plot_hmm_model_comparison(
        comparison_frame=hmm_metrics,
        output_dir=figures_dir,
        dpi=cfg.output.figure_dpi,
    )

    aligned_labels = {method_name: result.aligned_labels for method_name, result in alignment.methods.items()}
    artifacts["figures"] += plot_manifold_outputs(
        embedding=manifold_result.embedding,
        labels_by_method=aligned_labels,
        method_used=manifold_result.method_used,
        output_dir=figures_dir,
        dpi=cfg.output.figure_dpi,
    )
    artifacts["figures"] += plot_comparison_outputs(
        quality_frame=cluster_metrics,
        transition_matrices={f"hmm_{n_states}": result.transition_matrix for n_states, result in hmm_results.items()},
        embedding=manifold_result.embedding,
        labels_by_method=aligned_labels,
        threshold_summary=pd.DataFrame(threshold_compare_rows),
        output_dir=figures_dir,
        dpi=cfg.output.figure_dpi,
    )

    artifacts["report"] = [
        _write_report(
            output_dir=output_dir,
            config=cfg,
            cluster_metrics=cluster_metrics,
            hmm_metrics=hmm_metrics,
            explain_results=explain_results,
            manifold_method=manifold_result.method_used,
            runtime_info=runtime.info(),
        )
    ]

    return PipelineResult(
        output_dir=output_dir,
        preprocess=preprocess_result,
        windows=windows,
        features=feature_result,
        clustering=clustering,
        alignment=alignment,
        hmm=hmm_results,
        manifold=manifold_result,
        explain=explain_results,
        artifact_paths=artifacts,
        runtime_info=runtime.info(),
    )
