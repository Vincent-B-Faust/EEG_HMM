from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class InputConfig:
    filename: str | None = None
    eeg_channel: str | int | None = None
    emg_channel: str | int | None = None
    mat_variable: str | None = None
    csv_has_header: bool | None = None
    csv_separator: str | None = None
    csv_eeg_column: str | int | None = None
    csv_emg_column: str | int | None = None


@dataclass
class FilterConfig:
    eeg_band: tuple[float, float] = (0.5, 45.0)
    emg_band: tuple[float, float] = (10.0, 100.0)
    notch_freq: float | None = 50.0
    notch_quality: float = 30.0
    filter_order: int = 4
    standardize_signal: bool = True


@dataclass
class WindowConfig:
    window_sec: float = 30.0
    overlap_sec: float = 0.0


@dataclass
class FeatureConfig:
    mode: Literal["full", "legacy"] = "full"
    scaling: Literal["zscore", "minmax"] = "zscore"
    max_spectrum_hz: float = 30.0
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    peak_height_std: float = 0.75
    band_definitions: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "sigma": (12.0, 16.0),
            "beta": (16.0, 30.0),
        }
    )


@dataclass
class ClusterConfig:
    n_clusters: int = 3
    random_seed: int = 42
    kmeans_restarts: int = 10
    kmeans_iterations: int = 100
    gmm_restarts: int = 5
    gmm_iterations: int = 100
    gmm_tolerance: float = 1e-4
    hierarchical_linkage: Literal["ward", "average", "complete"] = "ward"
    metric_sample_size: int = 2000
    alignment_reference: str = "kmeans"
    smoothing_pseudocount: float = 1.0


@dataclass
class ManifoldConfig:
    method: Literal["umap", "diffusion", "pca"] = "umap"
    n_components: int = 2
    diffusion_neighbors: int = 15
    diffusion_alpha: float = 1.0
    diffusion_epsilon: float | None = None


@dataclass
class ExplainConfig:
    max_depth: int = 3
    min_samples_leaf: int = 10
    threshold_candidates: int = 32
    top_features_to_plot: int = 8


@dataclass
class HMMConfig:
    state_counts: tuple[int, ...] = (3, 4, 5)
    covariance_type: Literal["diag"] = "diag"
    max_iterations: int = 100
    tolerance: float = 1e-4
    n_restarts: int = 5
    regularization: float = 1e-3
    transition_pseudocount: float = 1e-3
    random_seed: int = 42


@dataclass
class ExecutionConfig:
    use_dask: bool = True
    k_user: int = 3
    window_strategy: Literal["seconds", "samples", "notebook_auto"] = "seconds"
    window_size: int | None = None
    overlap: int | None = None


@dataclass
class AccelerationConfig:
    enabled: bool = True
    backend: Literal["auto", "numpy", "torch"] = "auto"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    min_windows_for_gpu: int = 64
    accelerate_features: bool = True
    accelerate_clustering: bool = True
    accelerate_manifold: bool = True


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs")
    figure_dpi: int = 160
    image_format: str = "png"


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    windowing: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output"]["output_dir"] = str(self.output.output_dir)
        return payload
