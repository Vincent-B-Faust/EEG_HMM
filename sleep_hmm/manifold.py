from __future__ import annotations

import importlib.util

import numpy as np
from scipy.spatial.distance import cdist

from .acceleration import AccelerationRuntime
from .config import ManifoldConfig
from .types import ManifoldResult


def _pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ vt[:n_components].T
    eigenvalues = (singular_values**2) / max(features.shape[0] - 1, 1)
    return embedding, eigenvalues[:n_components]


def _pairwise_sqeuclidean_torch(features: object) -> object:
    return (features**2).sum(dim=1, keepdim=True) + (features**2).sum(dim=1).unsqueeze(0) - 2.0 * (features @ features.T)


def _pca_torch(features: np.ndarray, n_components: int, runtime: AccelerationRuntime) -> tuple[np.ndarray, np.ndarray]:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")
    features_t = runtime.tensor(features)
    centered = features_t - features_t.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    embedding = centered @ vh[:n_components].T
    eigenvalues = (singular_values**2) / max(features.shape[0] - 1, 1)
    return runtime.to_numpy(embedding), runtime.to_numpy(eigenvalues[:n_components])


def _diffusion_map(features: np.ndarray, cfg: ManifoldConfig) -> tuple[np.ndarray, np.ndarray]:
    distances = cdist(features, features, metric="sqeuclidean")
    if cfg.diffusion_epsilon is None:
        nonzero = distances[distances > 0]
        epsilon = float(np.median(nonzero)) if nonzero.size else 1.0
    else:
        epsilon = cfg.diffusion_epsilon
    epsilon = max(epsilon, 1e-6)
    kernel = np.exp(-distances / epsilon)
    row_sum = kernel.sum(axis=1, keepdims=True)
    transition = kernel / np.maximum(row_sum, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eig(transition)
    order = np.argsort(-np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    coords = eigenvectors[:, 1 : cfg.n_components + 1] * eigenvalues[1 : cfg.n_components + 1]
    return coords, eigenvalues[1 : cfg.n_components + 1]


def _diffusion_map_torch(features: np.ndarray, cfg: ManifoldConfig, runtime: AccelerationRuntime) -> tuple[np.ndarray, np.ndarray]:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")
    features_t = runtime.tensor(features)
    distances = torch.clamp(_pairwise_sqeuclidean_torch(features_t), min=0.0)
    if cfg.diffusion_epsilon is None:
        nonzero = runtime.to_numpy(distances[distances > 0])
        epsilon = float(np.median(nonzero)) if nonzero.size else 1.0
    else:
        epsilon = cfg.diffusion_epsilon
    epsilon = max(epsilon, 1e-6)
    kernel = torch.exp(-distances / epsilon)
    transition = kernel / torch.clamp(kernel.sum(dim=1, keepdim=True), min=1e-12)
    transition_np = runtime.to_numpy(transition)
    eigenvalues, eigenvectors = np.linalg.eig(transition_np)
    order = np.argsort(-np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    coords = eigenvectors[:, 1 : cfg.n_components + 1] * eigenvalues[1 : cfg.n_components + 1]
    return coords, eigenvalues[1 : cfg.n_components + 1]


def manifold(
    features: np.ndarray,
    config: ManifoldConfig | None = None,
    runtime: AccelerationRuntime | None = None,
) -> ManifoldResult:
    cfg = config or ManifoldConfig()
    method_requested = cfg.method

    if cfg.method == "pca":
        if runtime is not None and runtime.should_accelerate("manifold", features.shape[0]):
            try:
                embedding, eigenvalues = _pca_torch(features, cfg.n_components, runtime)
                method_used = "pca_torch"
                runtime.record_stage("manifold", True, "PCA executed on GPU.")
            except Exception as exc:
                embedding, eigenvalues = _pca(features, cfg.n_components)
                method_used = "pca"
                runtime.record_stage("manifold", False, f"GPU PCA fallback: {exc}")
        else:
            embedding, eigenvalues = _pca(features, cfg.n_components)
            method_used = "pca"
            if runtime is not None:
                runtime.record_stage("manifold", False, "CPU PCA path.")
    elif cfg.method == "diffusion":
        if runtime is not None and runtime.should_accelerate("manifold", features.shape[0]):
            try:
                embedding, eigenvalues = _diffusion_map_torch(features, cfg, runtime)
                method_used = "diffusion_torch"
                runtime.record_stage("manifold", True, "Diffusion-kernel construction executed on GPU.")
            except Exception as exc:
                embedding, eigenvalues = _diffusion_map(features, cfg)
                method_used = "diffusion"
                runtime.record_stage("manifold", False, f"GPU diffusion fallback: {exc}")
        else:
            embedding, eigenvalues = _diffusion_map(features, cfg)
            method_used = "diffusion"
            if runtime is not None:
                runtime.record_stage("manifold", False, "CPU diffusion path.")
    elif cfg.method == "umap":
        if importlib.util.find_spec("umap") is not None:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=cfg.n_components, random_state=42)
            embedding = reducer.fit_transform(features)
            eigenvalues = None
            method_used = "umap"
            if runtime is not None:
                runtime.record_stage("manifold", False, "UMAP executed through CPU library path.")
        else:
            if runtime is not None and runtime.should_accelerate("manifold", features.shape[0]):
                try:
                    embedding, eigenvalues = _diffusion_map_torch(features, cfg, runtime)
                    method_used = "diffusion_torch_fallback"
                    runtime.record_stage("manifold", True, "UMAP unavailable; used GPU diffusion fallback.")
                except Exception as exc:
                    embedding, eigenvalues = _diffusion_map(features, cfg)
                    method_used = "diffusion_fallback"
                    runtime.record_stage("manifold", False, f"GPU diffusion fallback failed: {exc}")
            else:
                embedding, eigenvalues = _diffusion_map(features, cfg)
                method_used = "diffusion_fallback"
                if runtime is not None:
                    runtime.record_stage("manifold", False, "UMAP unavailable; used CPU diffusion fallback.")
    else:
        raise ValueError(f"Unknown manifold method: {cfg.method}")

    return ManifoldResult(
        method_requested=method_requested,
        method_used=method_used,
        embedding=np.asarray(embedding, dtype=float),
        eigenvalues=None if eigenvalues is None else np.asarray(eigenvalues, dtype=float),
        metadata={"n_components": cfg.n_components},
    )
