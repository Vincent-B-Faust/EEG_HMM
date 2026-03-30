from __future__ import annotations

from pathlib import Path

from .config import InputConfig, PipelineConfig
from .io import load_signals
from .pipeline import run_pipeline
from .types import PipelineResult
from .utils import sanitize_session_name


def apply_compatibility_settings(config: PipelineConfig, signal_length: int, fs: float) -> PipelineConfig:
    config.clustering.n_clusters = config.execution.k_user
    strategy = config.execution.window_strategy
    if strategy == "samples":
        if config.execution.window_size is None:
            raise ValueError("window_size must be provided when window_strategy='samples'.")
        overlap = 0 if config.execution.overlap is None else config.execution.overlap
        config.windowing.window_sec = config.execution.window_size / fs
        config.windowing.overlap_sec = overlap / fs
    elif strategy == "notebook_auto":
        recommended_window = int(signal_length / 20)
        window_size = min(recommended_window, int(fs * 0.1))
        overlap = int(window_size * 0.25)
        config.execution.window_size = window_size
        config.execution.overlap = overlap
        config.windowing.window_sec = window_size / fs
        config.windowing.overlap_sec = overlap / fs
    return config


def build_compatible_config(
    filename: str | None = None,
    fs: float = 2000.0,
    use_dask: bool = True,
    k_user: int = 3,
    window_size: int | None = None,
    overlap: int | None = None,
    output_dir: str | Path = "sessions",
    feature_mode: str = "full",
    feature_scaling: str = "zscore",
    manifold_method: str = "umap",
    window_strategy: str = "notebook_auto",
    eeg_channel: str | int | None = None,
    emg_channel: str | int | None = None,
    mat_variable: str | None = None,
    csv_eeg_column: str | int | None = None,
    csv_emg_column: str | int | None = None,
    csv_has_header: bool | None = None,
    csv_separator: str | None = None,
    acceleration_backend: str = "auto",
    acceleration_device: str = "auto",
    min_windows_for_gpu: int = 64,
) -> PipelineConfig:
    config = PipelineConfig()
    config.input = InputConfig(
        filename=filename,
        eeg_channel=eeg_channel,
        emg_channel=emg_channel,
        mat_variable=mat_variable,
        csv_has_header=csv_has_header,
        csv_separator=csv_separator,
        csv_eeg_column=csv_eeg_column,
        csv_emg_column=csv_emg_column,
    )
    config.execution.use_dask = use_dask
    config.execution.k_user = k_user
    config.execution.window_strategy = window_strategy  # type: ignore[assignment]
    config.execution.window_size = window_size
    config.execution.overlap = overlap
    config.features.mode = feature_mode  # type: ignore[assignment]
    config.features.scaling = feature_scaling  # type: ignore[assignment]
    config.manifold.method = manifold_method  # type: ignore[assignment]
    config.acceleration.backend = acceleration_backend  # type: ignore[assignment]
    config.acceleration.device = acceleration_device  # type: ignore[assignment]
    config.acceleration.min_windows_for_gpu = min_windows_for_gpu
    config.output.output_dir = Path(output_dir)
    config.clustering.n_clusters = k_user
    return config


def run_file_pipeline(filename: str | Path, fs: float | None = None, config: PipelineConfig | None = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    cfg.input.filename = str(filename)
    source = Path(filename)
    session_name = sanitize_session_name(source.stem)
    base_output_dir = Path(cfg.output.output_dir)
    if base_output_dir.name != session_name:
        cfg.output.output_dir = base_output_dir / session_name
    bundle = load_signals(filename, fs=fs, config=cfg.input)
    cfg = apply_compatibility_settings(cfg, signal_length=len(bundle.eeg), fs=bundle.fs)
    return run_pipeline(eeg=bundle.eeg, emg=bundle.emg, fs=bundle.fs, config=cfg)


def run_interactive(
    filename: str | Path,
    fs: float = 2000.0,
    use_dask: bool = True,
    k_user: int = 3,
    window_size: int | None = None,
    overlap: int | None = None,
    output_dir: str | Path = "sessions",
    feature_mode: str = "full",
    feature_scaling: str = "zscore",
    manifold_method: str = "umap",
    window_strategy: str = "notebook_auto",
    eeg_channel: str | int | None = None,
    emg_channel: str | int | None = None,
    mat_variable: str | None = None,
    csv_eeg_column: str | int | None = None,
    csv_emg_column: str | int | None = None,
    csv_has_header: bool | None = None,
    csv_separator: str | None = None,
    acceleration_backend: str = "auto",
    acceleration_device: str = "auto",
    min_windows_for_gpu: int = 64,
) -> PipelineResult:
    config = build_compatible_config(
        filename=str(filename),
        fs=fs,
        use_dask=use_dask,
        k_user=k_user,
        window_size=window_size,
        overlap=overlap,
        output_dir=output_dir,
        feature_mode=feature_mode,
        feature_scaling=feature_scaling,
        manifold_method=manifold_method,
        window_strategy=window_strategy,
        eeg_channel=eeg_channel,
        emg_channel=emg_channel,
        mat_variable=mat_variable,
        csv_eeg_column=csv_eeg_column,
        csv_emg_column=csv_emg_column,
        csv_has_header=csv_has_header,
        csv_separator=csv_separator,
        acceleration_backend=acceleration_backend,
        acceleration_device=acceleration_device,
        min_windows_for_gpu=min_windows_for_gpu,
    )
    return run_file_pipeline(filename, fs=fs, config=config)
