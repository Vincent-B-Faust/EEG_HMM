from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .interactive import run_file_pipeline
from .pipeline import run_pipeline


def _synthetic_demo(fs: float, duration_sec: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_sec)) / fs
    segment = t.size // 3
    eeg = np.empty_like(t)
    emg = np.empty_like(t)
    eeg[:segment] = 1.2 * np.sin(2 * np.pi * 2.0 * t[:segment]) + 0.15 * rng.standard_normal(segment)
    eeg[segment : 2 * segment] = 0.9 * np.sin(2 * np.pi * 6.0 * t[segment : 2 * segment]) + 0.2 * rng.standard_normal(segment)
    eeg[2 * segment :] = 0.6 * np.sin(2 * np.pi * 12.0 * t[2 * segment :]) + 0.35 * rng.standard_normal(t.size - 2 * segment)
    emg[:segment] = 0.15 * rng.standard_normal(segment)
    emg[segment : 2 * segment] = 0.35 * rng.standard_normal(segment)
    emg[2 * segment :] = 0.8 * rng.standard_normal(t.size - 2 * segment)
    return eeg, emg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unsupervised sleep stage clustering and dynamics analysis.")
    parser.add_argument("--input", "--filename", dest="filename", type=str, help="Path to .edf/.csv/.mat/.npz/.npy input file.")
    parser.add_argument("--fs", type=float, default=None, help="Sampling rate, required for CSV/NPY/MAT inputs when it is not stored in file.")
    parser.add_argument("--output", type=str, default="sessions", help="Output root directory; file inputs are saved under a session folder named after the input file.")
    parser.add_argument("--use-dask", action=argparse.BooleanOptionalAction, default=True, help="Notebook-compatible flag; current implementation uses vectorized NumPy path.")
    parser.add_argument("--k-user", "--clusters", dest="k_user", type=int, default=3, help="Number of clusters, compatible with the source notebook.")
    parser.add_argument("--window-size", type=int, default=None, help="Window size in samples, compatible with the source notebook.")
    parser.add_argument("--overlap", type=int, default=None, help="Window overlap in samples, compatible with the source notebook.")
    parser.add_argument("--window-strategy", type=str, default="seconds", choices=["seconds", "samples", "notebook_auto"], help="How to interpret window parameters.")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Window length in seconds.")
    parser.add_argument("--overlap-sec", type=float, default=0.0, help="Window overlap in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--manifold", type=str, default="umap", choices=["umap", "diffusion", "pca"], help="Manifold method.")
    parser.add_argument("--feature-mode", type=str, default="full", choices=["full", "legacy"], help="Feature set: full pipeline or notebook-compatible 4-feature mode.")
    parser.add_argument("--feature-scaling", type=str, default="zscore", choices=["zscore", "minmax"], help="Feature scaling method.")
    parser.add_argument("--acceleration-backend", type=str, default="auto", choices=["auto", "numpy", "torch"], help="Acceleration backend selection.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Acceleration device selection.")
    parser.add_argument("--min-windows-for-gpu", type=int, default=64, help="Minimum number of windows before enabling GPU acceleration.")
    parser.add_argument("--eeg-channel", type=str, default=None, help="EDF channel name or index for EEG.")
    parser.add_argument("--emg-channel", type=str, default=None, help="EDF channel name or index for EMG.")
    parser.add_argument("--mat-variable", type=str, default=None, help="MAT variable name to load.")
    parser.add_argument("--csv-eeg-column", type=str, default=None, help="CSV EEG column name or index.")
    parser.add_argument("--csv-emg-column", type=str, default=None, help="CSV EMG column name or index.")
    parser.add_argument("--csv-has-header", action=argparse.BooleanOptionalAction, default=None, help="Whether CSV contains a header row.")
    parser.add_argument("--csv-separator", type=str, default=None, help="CSV separator, auto-detected when omitted.")
    parser.add_argument("--demo", action="store_true", help="Run on built-in synthetic EEG/EMG.")
    parser.add_argument("--duration-sec", type=float, default=60.0 * 30, help="Synthetic demo duration in seconds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = PipelineConfig()
    cfg.output.output_dir = Path(args.output)
    cfg.windowing.window_sec = args.window_sec
    cfg.windowing.overlap_sec = args.overlap_sec
    cfg.execution.use_dask = args.use_dask
    cfg.execution.k_user = args.k_user
    cfg.execution.window_strategy = args.window_strategy
    cfg.execution.window_size = args.window_size
    cfg.execution.overlap = args.overlap
    cfg.clustering.n_clusters = args.k_user
    cfg.clustering.random_seed = args.seed
    cfg.manifold.method = args.manifold
    cfg.features.mode = args.feature_mode
    cfg.features.scaling = args.feature_scaling
    cfg.acceleration.backend = args.acceleration_backend
    cfg.acceleration.device = args.device
    cfg.acceleration.min_windows_for_gpu = args.min_windows_for_gpu
    cfg.input.filename = args.filename
    cfg.input.eeg_channel = args.eeg_channel
    cfg.input.emg_channel = args.emg_channel
    cfg.input.mat_variable = args.mat_variable
    cfg.input.csv_eeg_column = args.csv_eeg_column
    cfg.input.csv_emg_column = args.csv_emg_column
    cfg.input.csv_has_header = args.csv_has_header
    cfg.input.csv_separator = args.csv_separator

    if args.demo:
        fs = args.fs or 128.0
        eeg, emg = _synthetic_demo(fs=fs, duration_sec=args.duration_sec, seed=args.seed)
    else:
        if not args.filename:
            raise SystemExit("Either --input or --demo must be provided.")
        result = run_file_pipeline(args.filename, fs=args.fs, config=cfg)
        print(f"Analysis complete. Outputs saved to: {result.output_dir}")
        return

    result = run_pipeline(eeg=eeg, emg=emg, fs=fs, config=cfg)
    print(f"Analysis complete. Outputs saved to: {result.output_dir}")


if __name__ == "__main__":
    main()
