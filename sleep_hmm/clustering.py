from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from .acceleration import AccelerationRuntime
from .config import ClusterConfig
from .types import ClusterMethodResult, ClusterResult, FeatureResult


def _initialize_centroids(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    first_idx = int(rng.integers(features.shape[0]))
    centroids = [features[first_idx]]
    for _ in range(1, k):
        distances = cdist(features, np.vstack(centroids), metric="sqeuclidean").min(axis=1)
        probs = distances / distances.sum() if distances.sum() > 0 else np.full(features.shape[0], 1.0 / features.shape[0])
        next_idx = int(rng.choice(features.shape[0], p=probs))
        centroids.append(features[next_idx])
    return np.vstack(centroids)


def _kmeans(features: np.ndarray, k: int, cfg: ClusterConfig) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(cfg.random_seed)
    best_labels = None
    best_centroids = None
    best_inertia = math.inf
    for _ in range(cfg.kmeans_restarts):
        centroids = _initialize_centroids(features, k, rng)
        for _ in range(cfg.kmeans_iterations):
            distances = cdist(features, centroids, metric="sqeuclidean")
            labels = np.argmin(distances, axis=1)
            new_centroids = centroids.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = features[mask].mean(axis=0)
                else:
                    farthest = np.argmax(distances.min(axis=1))
                    new_centroids[cluster_id] = features[farthest]
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        distances = cdist(features, centroids, metric="sqeuclidean")
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(features.shape[0]), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, best_inertia


def _pairwise_sqeuclidean_torch(features: object, centroids: object) -> object:
    return (features**2).sum(dim=1, keepdim=True) + (centroids**2).sum(dim=1).unsqueeze(0) - 2.0 * (features @ centroids.T)


def _initialize_centroids_torch(features: object, k: int, rng: np.random.Generator, runtime: AccelerationRuntime) -> object:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")
    first_idx = int(rng.integers(features.shape[0]))
    centroids = [features[first_idx : first_idx + 1]]
    for _ in range(1, k):
        existing = torch.cat(centroids, dim=0)
        distances = torch.clamp(_pairwise_sqeuclidean_torch(features, existing), min=0.0).min(dim=1).values
        probs = runtime.to_numpy(distances / torch.clamp(distances.sum(), min=1e-12))
        next_idx = int(rng.choice(features.shape[0], p=probs if probs.sum() > 0 else None))
        centroids.append(features[next_idx : next_idx + 1])
    return torch.cat(centroids, dim=0)


def _kmeans_torch(features: np.ndarray, k: int, cfg: ClusterConfig, runtime: AccelerationRuntime) -> tuple[np.ndarray, np.ndarray, float]:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")
    rng = np.random.default_rng(cfg.random_seed)
    features_t = runtime.tensor(features)
    best_labels = None
    best_centroids = None
    best_inertia = math.inf
    for _ in range(cfg.kmeans_restarts):
        centroids = _initialize_centroids_torch(features_t, k, rng, runtime)
        for _ in range(cfg.kmeans_iterations):
            distances = torch.clamp(_pairwise_sqeuclidean_torch(features_t, centroids), min=0.0)
            labels = torch.argmin(distances, dim=1)
            new_centroids = centroids.clone()
            min_distances = distances.min(dim=1).values
            for cluster_id in range(k):
                mask = labels == cluster_id
                if bool(torch.any(mask)):
                    new_centroids[cluster_id] = features_t[mask].mean(dim=0)
                else:
                    farthest = int(torch.argmax(min_distances).item())
                    new_centroids[cluster_id] = features_t[farthest]
            if bool(torch.allclose(new_centroids, centroids, atol=1e-4, rtol=1e-4)):
                centroids = new_centroids
                break
            centroids = new_centroids
        distances = torch.clamp(_pairwise_sqeuclidean_torch(features_t, centroids), min=0.0)
        labels = torch.argmin(distances, dim=1)
        row_indices = torch.arange(features_t.shape[0], device=features_t.device)
        inertia = float(distances[row_indices, labels].sum().item())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = runtime.to_numpy(labels).astype(int)
            best_centroids = runtime.to_numpy(centroids)
    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, best_inertia


@dataclass
class DiagonalGMM:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihood: float


def _log_gaussian_diagonal(features: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(variances, 1e-6)
    log_det = np.sum(np.log(2.0 * np.pi * safe_var), axis=1)
    diff = features[:, None, :] - means[None, :, :]
    quadratic = np.sum((diff**2) / safe_var[None, :, :], axis=2)
    return -0.5 * (log_det[None, :] + quadratic)


def _fit_diagonal_gmm(features: np.ndarray, k: int, cfg: ClusterConfig) -> tuple[np.ndarray, DiagonalGMM]:
    rng = np.random.default_rng(cfg.random_seed)
    n_samples, n_features = features.shape
    best_model = None
    best_resp = None
    best_ll = -math.inf

    for _ in range(cfg.gmm_restarts):
        labels, means, _ = _kmeans(features, k, cfg)
        weights = np.bincount(labels, minlength=k).astype(float)
        weights /= weights.sum()
        variances = np.vstack(
            [
                features[labels == cluster_id].var(axis=0) + 1e-3
                if np.any(labels == cluster_id)
                else np.full(n_features, 1.0)
                for cluster_id in range(k)
            ]
        )

        prev_ll = -math.inf
        for _ in range(cfg.gmm_iterations):
            log_prob = _log_gaussian_diagonal(features, means, variances) + np.log(np.maximum(weights, 1e-12))[None, :]
            max_log = log_prob.max(axis=1, keepdims=True)
            log_sum = max_log + np.log(np.exp(log_prob - max_log).sum(axis=1, keepdims=True))
            responsibilities = np.exp(log_prob - log_sum)
            ll = float(log_sum.sum())

            nk = responsibilities.sum(axis=0) + 1e-12
            weights = nk / n_samples
            means = (responsibilities.T @ features) / nk[:, None]
            diff = features[:, None, :] - means[None, :, :]
            variances = (responsibilities[:, :, None] * diff**2).sum(axis=0) / nk[:, None]
            variances = np.maximum(variances, 1e-6)

            if abs(ll - prev_ll) < cfg.gmm_tolerance:
                prev_ll = ll
                break
            prev_ll = ll

        if prev_ll > best_ll:
            best_ll = prev_ll
            best_resp = responsibilities.copy()
            best_model = DiagonalGMM(weights=weights.copy(), means=means.copy(), variances=variances.copy(), log_likelihood=best_ll)

    assert best_resp is not None and best_model is not None
    labels = np.argmax(best_resp, axis=1)
    return labels, best_model


def _fit_diagonal_gmm_torch(features: np.ndarray, k: int, cfg: ClusterConfig, runtime: AccelerationRuntime) -> tuple[np.ndarray, DiagonalGMM]:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")
    features_t = runtime.tensor(features)
    n_samples, n_features = features.shape
    best_model = None
    best_resp = None
    best_ll = -math.inf

    for _ in range(cfg.gmm_restarts):
        labels, means_np, _ = _kmeans_torch(features, k, cfg, runtime)
        means = runtime.tensor(means_np)
        weights = torch.as_tensor(np.bincount(labels, minlength=k) / max(len(labels), 1), device=features_t.device, dtype=features_t.dtype)
        variances = torch.stack(
            [
                features_t[runtime.tensor((labels == cluster_id).astype(float)).bool()].var(dim=0, unbiased=False) + 1e-3
                if np.any(labels == cluster_id)
                else torch.full((n_features,), 1.0, device=features_t.device, dtype=features_t.dtype)
                for cluster_id in range(k)
            ]
        )

        prev_ll = -math.inf
        for _ in range(cfg.gmm_iterations):
            safe_var = torch.clamp(variances, min=1e-6)
            log_det = torch.sum(torch.log(2.0 * np.pi * safe_var), dim=1)
            diff = features_t[:, None, :] - means[None, :, :]
            quadratic = torch.sum((diff**2) / safe_var[None, :, :], dim=2)
            log_prob = -0.5 * (log_det[None, :] + quadratic) + torch.log(torch.clamp(weights, min=1e-12))[None, :]
            log_sum = torch.logsumexp(log_prob, dim=1, keepdim=True)
            responsibilities = torch.exp(log_prob - log_sum)
            ll = float(log_sum.sum().item())

            nk = responsibilities.sum(dim=0) + 1e-12
            weights = nk / n_samples
            means = (responsibilities.T @ features_t) / nk[:, None]
            diff = features_t[:, None, :] - means[None, :, :]
            variances = (responsibilities[:, :, None] * diff**2).sum(dim=0) / nk[:, None]
            variances = torch.clamp(variances, min=1e-6)

            if abs(ll - prev_ll) < cfg.gmm_tolerance:
                prev_ll = ll
                break
            prev_ll = ll

        if prev_ll > best_ll:
            best_ll = prev_ll
            best_resp = runtime.to_numpy(responsibilities)
            best_model = DiagonalGMM(
                weights=runtime.to_numpy(weights),
                means=runtime.to_numpy(means),
                variances=runtime.to_numpy(variances),
                log_likelihood=best_ll,
            )

    assert best_resp is not None and best_model is not None
    labels = np.argmax(best_resp, axis=1)
    return labels, best_model


def _sample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    if n_samples <= sample_size:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False))


def _silhouette_score(features: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0
    indices = _sample_indices(features.shape[0], sample_size, seed)
    sample = features[indices]
    sample_labels = labels[indices]
    distances = cdist(sample, sample, metric="euclidean")
    silhouettes = []
    for idx, label in enumerate(sample_labels):
        same_mask = sample_labels == label
        other_mask = sample_labels != label
        if same_mask.sum() <= 1 or not np.any(other_mask):
            silhouettes.append(0.0)
            continue
        intra = distances[idx, same_mask]
        a = intra[intra > 0].mean() if np.any(intra > 0) else 0.0
        b = math.inf
        for other_label in unique:
            if other_label == label:
                continue
            mask = sample_labels == other_label
            if np.any(mask):
                b = min(b, float(distances[idx, mask].mean()))
        silhouettes.append(0.0 if not np.isfinite(b) or max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(silhouettes))


def _davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0
    centroids = np.vstack([features[labels == label].mean(axis=0) for label in unique])
    scatters = np.array(
        [
            np.mean(np.linalg.norm(features[labels == label] - centroids[i], axis=1))
            for i, label in enumerate(unique)
        ]
    )
    centroid_distances = cdist(centroids, centroids)
    centroid_distances[centroid_distances == 0] = np.inf
    scores = []
    for i in range(unique.size):
        ratios = (scatters[i] + scatters) / centroid_distances[i]
        ratios[i] = -np.inf
        scores.append(float(np.max(ratios)))
    return float(np.mean(scores))


def _cluster_balance_entropy(labels: np.ndarray, k: int) -> float:
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _average_spectra(spectra: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    means = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        means.append(spectra[mask].mean(axis=0) if np.any(mask) else np.zeros(spectra.shape[1]))
    return np.vstack(means)


def cluster(
    features: FeatureResult,
    config: ClusterConfig | None = None,
    runtime: AccelerationRuntime | None = None,
) -> ClusterResult:
    cfg = config or ClusterConfig()
    matrix = features.scaled_table.to_numpy(dtype=float)
    k = cfg.n_clusters

    if runtime is not None and runtime.should_accelerate("clustering", matrix.shape[0]):
        try:
            kmeans_labels, kmeans_centroids, inertia = _kmeans_torch(matrix, k, cfg, runtime)
            gmm_labels, gmm_model = _fit_diagonal_gmm_torch(matrix, k, cfg, runtime)
            runtime.record_stage("clustering", True, "KMeans and diagonal GMM executed on GPU; hierarchical remains on CPU.")
        except Exception as exc:
            kmeans_labels, kmeans_centroids, inertia = _kmeans(matrix, k, cfg)
            gmm_labels, gmm_model = _fit_diagonal_gmm(matrix, k, cfg)
            runtime.record_stage("clustering", False, f"GPU clustering fallback: {exc}")
    else:
        kmeans_labels, kmeans_centroids, inertia = _kmeans(matrix, k, cfg)
        gmm_labels, gmm_model = _fit_diagonal_gmm(matrix, k, cfg)
        if runtime is not None:
            runtime.record_stage("clustering", False, "CPU clustering path.")

    hierarchy = linkage(matrix, method=cfg.hierarchical_linkage)
    hier_labels = fcluster(hierarchy, t=k, criterion="maxclust") - 1

    methods: dict[str, ClusterMethodResult] = {}
    for name, labels in {
        "kmeans": kmeans_labels,
        "gmm": gmm_labels,
        "hierarchical": hier_labels,
    }.items():
        metrics = {
            "silhouette": _silhouette_score(matrix, labels, cfg.metric_sample_size, cfg.random_seed),
            "davies_bouldin": _davies_bouldin_score(matrix, labels),
            "balance_entropy": _cluster_balance_entropy(labels, k),
        }
        extra: dict[str, float | np.ndarray] = {}
        if name == "kmeans":
            extra["inertia"] = inertia
            extra["centroids"] = kmeans_centroids
        if name == "gmm":
            n_params = (k - 1) + k * matrix.shape[1] * 2
            bic = -2.0 * gmm_model.log_likelihood + n_params * np.log(matrix.shape[0])
            aic = -2.0 * gmm_model.log_likelihood + 2.0 * n_params
            metrics["bic"] = float(bic)
            metrics["aic"] = float(aic)
            extra["weights"] = gmm_model.weights
            extra["means"] = gmm_model.means
            extra["variances"] = gmm_model.variances
        if name == "hierarchical":
            extra["linkage_matrix"] = hierarchy
        methods[name] = ClusterMethodResult(
            labels=labels.astype(int),
            metrics=metrics,
            average_spectra=_average_spectra(features.eeg_spectra, labels, k),
            extra=extra,
        )

    return ClusterResult(methods=methods)
