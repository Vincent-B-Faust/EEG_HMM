from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExplainConfig
from .types import ExplainResult, TreeNode
from .utils import inverse_threshold


def _gini(labels: np.ndarray, n_classes: int) -> float:
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs**2))


def _candidate_thresholds(values: np.ndarray, limit: int) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([], dtype=float)
    if unique.size <= limit:
        return (unique[:-1] + unique[1:]) / 2.0
    quantiles = np.linspace(0.05, 0.95, limit)
    thresholds = np.quantile(values, quantiles)
    return np.unique(thresholds)


@dataclass
class _Split:
    feature_index: int
    threshold: float
    gain: float
    left_mask: np.ndarray
    right_mask: np.ndarray


class SimpleDecisionTree:
    def __init__(self, max_depth: int, min_samples_leaf: int, threshold_candidates: int):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.threshold_candidates = threshold_candidates
        self.root: TreeNode | None = None
        self.n_classes: int = 0
        self.feature_importances_: np.ndarray | None = None
        self._node_counter = 0
        self._total_samples = 0
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray, means: np.ndarray, stds: np.ndarray) -> "SimpleDecisionTree":
        self.n_classes = int(np.max(labels)) + 1
        self.feature_importances_ = np.zeros(features.shape[1], dtype=float)
        self._node_counter = 0
        self._total_samples = features.shape[0]
        self._feature_means = means
        self._feature_stds = stds
        self.root = self._build(features, labels, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _build(self, features: np.ndarray, labels: np.ndarray, depth: int) -> TreeNode:
        counts = np.bincount(labels, minlength=self.n_classes)
        predicted = int(np.argmax(counts))
        impurity = _gini(labels, self.n_classes)
        node = TreeNode(
            node_id=self._node_counter,
            depth=depth,
            class_counts=counts,
            predicted_class=predicted,
            impurity=impurity,
            n_samples=int(labels.size),
        )
        self._node_counter += 1
        if (
            depth >= self.max_depth
            or labels.size < 2 * self.min_samples_leaf
            or np.unique(labels).size == 1
        ):
            return node

        split = self._find_best_split(features, labels)
        if split is None or split.gain <= 0:
            return node

        assert self._feature_means is not None and self._feature_stds is not None and self.feature_importances_ is not None
        node.feature_index = split.feature_index
        node.threshold_scaled = float(split.threshold)
        node.threshold_raw = float(
            inverse_threshold(split.threshold, self._feature_means[split.feature_index], self._feature_stds[split.feature_index])
        )
        self.feature_importances_[split.feature_index] += split.gain * labels.size / max(self._total_samples, 1)
        node.left = self._build(features[split.left_mask], labels[split.left_mask], depth + 1)
        node.right = self._build(features[split.right_mask], labels[split.right_mask], depth + 1)
        return node

    def _find_best_split(self, features: np.ndarray, labels: np.ndarray) -> _Split | None:
        parent_impurity = _gini(labels, self.n_classes)
        best: _Split | None = None
        for feature_index in range(features.shape[1]):
            values = features[:, feature_index]
            thresholds = _candidate_thresholds(values, self.threshold_candidates)
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                left_impurity = _gini(labels[left_mask], self.n_classes)
                right_impurity = _gini(labels[right_mask], self.n_classes)
                weighted = (left_mask.sum() * left_impurity + right_mask.sum() * right_impurity) / labels.size
                gain = parent_impurity - weighted
                if best is None or gain > best.gain:
                    best = _Split(
                        feature_index=feature_index,
                        threshold=float(threshold),
                        gain=float(gain),
                        left_mask=left_mask,
                        right_mask=right_mask,
                    )
        return best

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree must be fitted before prediction.")
        predictions = np.zeros(features.shape[0], dtype=int)
        for idx, row in enumerate(features):
            node = self.root
            while node.feature_index is not None and node.left is not None and node.right is not None:
                if row[node.feature_index] <= float(node.threshold_scaled):
                    node = node.left
                else:
                    node = node.right
            predictions[idx] = node.predicted_class
        return predictions


def _collect_nodes(node: TreeNode, feature_names: list[str]) -> list[dict[str, object]]:
    records = [
        {
            "node_id": node.node_id,
            "depth": node.depth,
            "predicted_class": node.predicted_class,
            "n_samples": node.n_samples,
            "impurity": node.impurity,
            "feature": None if node.feature_index is None else feature_names[node.feature_index],
            "threshold_scaled": node.threshold_scaled,
            "threshold_raw": node.threshold_raw,
            "class_counts": node.class_counts.tolist(),
        }
    ]
    if node.left is not None:
        records.extend(_collect_nodes(node.left, feature_names))
    if node.right is not None:
        records.extend(_collect_nodes(node.right, feature_names))
    return records


def _rules_text(node: TreeNode, feature_names: list[str], prefix: str = "") -> list[str]:
    if node.feature_index is None or node.left is None or node.right is None:
        counts_text = ", ".join(f"S{idx}={count}" for idx, count in enumerate(node.class_counts))
        return [f"{prefix}Leaf -> state {node.predicted_class} ({counts_text})"]

    feature_name = feature_names[node.feature_index]
    threshold = node.threshold_raw if node.threshold_raw is not None else node.threshold_scaled
    here = f"{prefix}if {feature_name} <= {threshold:.4f}"
    left_lines = _rules_text(node.left, feature_names, prefix + "  ")
    right_lines = _rules_text(node.right, feature_names, prefix + "  ")
    return [here] + left_lines + [f"{prefix}else"] + right_lines


def _best_thresholds_by_cluster(
    raw_table: pd.DataFrame,
    scaled_table: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    threshold_candidates: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    n_clusters = int(np.max(labels)) + 1
    for cluster_id in range(n_clusters):
        target = (labels == cluster_id).astype(int)
        for feature_index, feature_name in enumerate(feature_names):
            scaled_values = scaled_table.iloc[:, feature_index].to_numpy(dtype=float)
            raw_values = raw_table.iloc[:, feature_index].to_numpy(dtype=float)
            best_score = -np.inf
            best_threshold_scaled = None
            best_direction = "<="
            for threshold in _candidate_thresholds(scaled_values, threshold_candidates):
                for direction in ("<=", ">"):
                    if direction == "<=":
                        prediction = (scaled_values <= threshold).astype(int)
                    else:
                        prediction = (scaled_values > threshold).astype(int)
                    tp = np.sum((prediction == 1) & (target == 1))
                    fp = np.sum((prediction == 1) & (target == 0))
                    fn = np.sum((prediction == 0) & (target == 1))
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
                    if score > best_score:
                        best_score = score
                        best_threshold_scaled = float(threshold)
                        best_direction = direction
            if best_threshold_scaled is None:
                best_threshold_scaled = 0.0
                best_score = 0.0
            records.append(
                {
                    "cluster": cluster_id,
                    "feature": feature_name,
                    "direction": best_direction,
                    "threshold_scaled": best_threshold_scaled,
                    "threshold_raw": inverse_threshold(best_threshold_scaled, means[feature_index], stds[feature_index]),
                    "score_f1": best_score,
                    "feature_mean_raw": float(raw_values.mean()),
                }
            )
    return pd.DataFrame(records)


def explain(
    raw_table: pd.DataFrame,
    scaled_table: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    scale_mean: np.ndarray,
    scale_std: np.ndarray,
    config: ExplainConfig | None = None,
) -> ExplainResult:
    cfg = config or ExplainConfig()
    tree = SimpleDecisionTree(
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        threshold_candidates=cfg.threshold_candidates,
    ).fit(scaled_table.to_numpy(dtype=float), labels.astype(int), scale_mean, scale_std)
    predictions = tree.predict(scaled_table.to_numpy(dtype=float))
    fidelity = float(np.mean(predictions == labels))
    assert tree.root is not None and tree.feature_importances_ is not None
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": tree.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    thresholds = _best_thresholds_by_cluster(
        raw_table=raw_table,
        scaled_table=scaled_table,
        labels=labels.astype(int),
        feature_names=feature_names,
        means=scale_mean,
        stds=scale_std,
        threshold_candidates=cfg.threshold_candidates,
    )
    rules = "\n".join(_rules_text(tree.root, feature_names))
    return ExplainResult(
        tree_root=tree.root,
        feature_importance=importance,
        thresholds=thresholds,
        rules_text=rules,
        fidelity=fidelity,
        node_records=_collect_nodes(tree.root, feature_names),
    )
