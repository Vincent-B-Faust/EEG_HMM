from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ClusterConfig
from .types import AlignmentMethodResult, AlignmentResult, ClusterResult
from .utils import compute_confusion_matrix


def align_cluster_labels(cluster_result: ClusterResult, config: ClusterConfig | None = None) -> AlignmentResult:
    cfg = config or ClusterConfig()
    if cfg.alignment_reference not in cluster_result.methods:
        raise ValueError(f"Unknown alignment reference: {cfg.alignment_reference}")
    reference_labels = cluster_result.methods[cfg.alignment_reference].labels
    n_clusters = cfg.n_clusters

    aligned_methods: dict[str, AlignmentMethodResult] = {}
    for method_name, method_result in cluster_result.methods.items():
        labels = method_result.labels
        confusion_before = compute_confusion_matrix(reference_labels, labels, n_clusters)
        if method_name == cfg.alignment_reference:
            mapping = {cluster_id: cluster_id for cluster_id in range(n_clusters)}
            aligned = labels.copy()
            confusion_after = confusion_before.copy()
        else:
            row_ind, col_ind = linear_sum_assignment(confusion_before.max() - confusion_before)
            mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind, strict=True)}
            aligned = np.vectorize(lambda item: mapping.get(int(item), int(item)))(labels).astype(int)
            confusion_after = compute_confusion_matrix(reference_labels, aligned, n_clusters)
        aligned_methods[method_name] = AlignmentMethodResult(
            aligned_labels=aligned,
            mapping=mapping,
            confusion_before=confusion_before,
            confusion_after=confusion_after,
        )

    return AlignmentResult(reference_method=cfg.alignment_reference, methods=aligned_methods)
