from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from .config import HMMConfig
from .types import HMMResult


def _run_lengths(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    change_points = np.flatnonzero(np.diff(labels, prepend=labels[0] - 1))
    starts = change_points
    ends = np.r_[change_points[1:], labels.size]
    run_labels = labels[starts]
    lengths = ends - starts
    return run_labels.astype(int), lengths.astype(int)


def _initialize_centroids(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    first_idx = int(rng.integers(features.shape[0]))
    centroids = [features[first_idx]]
    for _ in range(1, k):
        distances = cdist(features, np.vstack(centroids), metric="sqeuclidean").min(axis=1)
        if distances.sum() <= 0:
            next_idx = int(rng.integers(features.shape[0]))
        else:
            next_idx = int(rng.choice(features.shape[0], p=distances / distances.sum()))
        centroids.append(features[next_idx])
    return np.vstack(centroids)


def _simple_kmeans(features: np.ndarray, k: int, seed: int, max_iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = _initialize_centroids(features, k, rng)
    labels = np.zeros(features.shape[0], dtype=int)
    for _ in range(max_iterations):
        distances = cdist(features, centroids, metric="sqeuclidean")
        new_labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            mask = new_labels == cluster_id
            if np.any(mask):
                new_centroids[cluster_id] = features[mask].mean(axis=0)
            else:
                new_centroids[cluster_id] = features[int(rng.integers(features.shape[0]))]
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return labels, centroids


def _initialize_gaussian_hmm(features: np.ndarray, n_states: int, cfg: HMMConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, means = _simple_kmeans(features, n_states, seed=seed)
    initial = np.full(n_states, cfg.transition_pseudocount, dtype=float)
    initial[labels[0]] += 1.0
    initial /= initial.sum()

    transitions = np.full((n_states, n_states), cfg.transition_pseudocount, dtype=float)
    for current, nxt in zip(labels[:-1], labels[1:], strict=True):
        transitions[current, nxt] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)

    variances = np.vstack(
        [
            features[labels == state].var(axis=0) + cfg.regularization
            if np.any(labels == state)
            else np.full(features.shape[1], 1.0 + cfg.regularization)
            for state in range(n_states)
        ]
    )
    variances = np.maximum(variances, cfg.regularization)
    return initial, transitions, means, variances


def _log_gaussian_diag(features: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(variances, 1e-12)
    log_det = np.sum(np.log(2.0 * np.pi * safe_var), axis=1)
    diff = features[:, None, :] - means[None, :, :]
    quadratic = np.sum((diff**2) / safe_var[None, :, :], axis=2)
    return -0.5 * (log_det[None, :] + quadratic)


def _forward_backward(log_emission: np.ndarray, initial: np.ndarray, transition: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_states = log_emission.shape
    log_initial = np.log(np.maximum(initial, 1e-300))
    log_transition = np.log(np.maximum(transition, 1e-300))

    log_alpha = np.empty((n_samples, n_states), dtype=float)
    log_alpha[0] = log_initial + log_emission[0]
    for t in range(1, n_samples):
        log_alpha[t] = log_emission[t] + logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0)

    log_beta = np.zeros((n_samples, n_states), dtype=float)
    for t in range(n_samples - 2, -1, -1):
        log_beta[t] = logsumexp(log_transition + log_emission[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    log_likelihood = float(logsumexp(log_alpha[-1]))
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)

    xi = np.empty((n_samples - 1, n_states, n_states), dtype=float)
    for t in range(n_samples - 1):
        xi[t] = np.exp(
            log_alpha[t][:, None]
            + log_transition
            + log_emission[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_likelihood
        )
    return log_likelihood, gamma, xi, log_alpha


def _viterbi(log_emission: np.ndarray, initial: np.ndarray, transition: np.ndarray) -> np.ndarray:
    n_samples, n_states = log_emission.shape
    log_initial = np.log(np.maximum(initial, 1e-300))
    log_transition = np.log(np.maximum(transition, 1e-300))

    delta = np.empty((n_samples, n_states), dtype=float)
    psi = np.zeros((n_samples, n_states), dtype=int)
    delta[0] = log_initial + log_emission[0]
    for t in range(1, n_samples):
        scores = delta[t - 1][:, None] + log_transition
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_emission[t] + scores[psi[t], np.arange(n_states)]

    states = np.zeros(n_samples, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n_samples - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, stationary_index])
    stationary = np.maximum(stationary, 0.0)
    if stationary.sum() <= 0:
        stationary = np.full(transition.shape[0], 1.0 / transition.shape[0])
    else:
        stationary /= stationary.sum()
    return stationary


def _fit_single_gaussian_hmm(features: np.ndarray, n_states: int, cfg: HMMConfig, seed: int) -> HMMResult:
    initial, transition, means, variances = _initialize_gaussian_hmm(features, n_states, cfg, seed)
    best_log_likelihood = -math.inf

    for _ in range(cfg.max_iterations):
        log_emission = _log_gaussian_diag(features, means, variances)
        log_likelihood, gamma, xi, _ = _forward_backward(log_emission, initial, transition)

        initial = np.maximum(gamma[0], cfg.transition_pseudocount)
        initial /= initial.sum()

        gamma_sum = gamma.sum(axis=0) + 1e-12
        transition = xi.sum(axis=0) + cfg.transition_pseudocount
        transition /= transition.sum(axis=1, keepdims=True)

        means = (gamma.T @ features) / gamma_sum[:, None]
        diff = features[:, None, :] - means[None, :, :]
        variances = (gamma[:, :, None] * diff**2).sum(axis=0) / gamma_sum[:, None]
        variances = np.maximum(variances, cfg.regularization)

        if abs(log_likelihood - best_log_likelihood) < cfg.tolerance:
            best_log_likelihood = log_likelihood
            break
        best_log_likelihood = log_likelihood

    log_emission = _log_gaussian_diag(features, means, variances)
    log_likelihood, _, _, _ = _forward_backward(log_emission, initial, transition)
    hidden_states = _viterbi(log_emission, initial, transition)
    stationary = _stationary_distribution(transition)
    run_labels, run_lengths = _run_lengths(hidden_states)

    durations_sec: dict[int, np.ndarray] = {}
    run_length_map: dict[int, np.ndarray] = {}
    for state in range(n_states):
        lengths = run_lengths[run_labels == state]
        run_length_map[state] = lengths
        durations_sec[state] = lengths.astype(float)

    n_features = features.shape[1]
    n_parameters = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_features
    bic = -2.0 * log_likelihood + n_parameters * np.log(features.shape[0])

    return HMMResult(
        n_states=n_states,
        hidden_states=hidden_states,
        transition_matrix=transition,
        initial_distribution=initial,
        stationary_distribution=stationary,
        means=means,
        variances=variances,
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        durations_sec=durations_sec,
        run_lengths=run_length_map,
    )


def hmm_analysis(
    features: np.ndarray,
    window_sec: float,
    config: HMMConfig | None = None,
) -> dict[int, HMMResult]:
    cfg = config or HMMConfig()
    feature_matrix = np.asarray(features, dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError("Gaussian HMM expects a 2D feature time series matrix.")
    if feature_matrix.shape[0] < 2:
        raise ValueError("At least two time windows are required for HMM analysis.")
    if cfg.covariance_type != "diag":
        raise ValueError("Only diagonal-covariance Gaussian HMM is currently implemented.")

    results: dict[int, HMMResult] = {}
    for offset, n_states in enumerate(cfg.state_counts):
        if n_states <= 1:
            raise ValueError("Each HMM state count must be greater than 1.")
        best_result = None
        best_log_likelihood = -math.inf
        for restart in range(cfg.n_restarts):
            seed = cfg.random_seed + offset * 1000 + restart
            candidate = _fit_single_gaussian_hmm(feature_matrix, n_states, cfg, seed)
            if candidate.log_likelihood > best_log_likelihood:
                best_log_likelihood = candidate.log_likelihood
                best_result = candidate
        assert best_result is not None
        best_result.durations_sec = {
            state: durations.astype(float) * float(window_sec)
            for state, durations in best_result.run_lengths.items()
        }
        results[n_states] = best_result
    return results
