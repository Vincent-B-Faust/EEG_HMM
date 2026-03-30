from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import AccelerationConfig


@dataclass
class AccelerationRuntime:
    config: AccelerationConfig
    backend_requested: str
    device_requested: str
    backend_used: str
    device_used: str
    active: bool
    reason: str
    torch_module: Any | None = None
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def should_accelerate(self, stage: str, workload_size: int | None = None) -> bool:
        if not self.active or self.backend_used != "torch":
            return False
        if workload_size is not None and workload_size < self.config.min_windows_for_gpu:
            return False
        if stage == "features":
            return self.config.accelerate_features
        if stage == "clustering":
            return self.config.accelerate_clustering
        if stage == "manifold":
            return self.config.accelerate_manifold
        return False

    def record_stage(self, stage: str, accelerated: bool, detail: str) -> None:
        self.stage_results[stage] = {
            "accelerated": accelerated,
            "backend": self.backend_used if accelerated else "numpy",
            "device": self.device_used if accelerated else "cpu",
            "detail": detail,
        }

    def tensor(self, array: np.ndarray) -> Any:
        if self.torch_module is None:
            raise RuntimeError("Torch runtime is not available.")
        dtype = self.torch_module.float32 if self.config.dtype == "float32" else self.torch_module.float64
        array_safe = np.asarray(array)
        if not array_safe.flags.writeable or not array_safe.flags.c_contiguous:
            array_safe = np.array(array_safe, copy=True, order="C")
        return self.torch_module.as_tensor(array_safe, dtype=dtype, device=self.device_used)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if self.torch_module is None:
            raise RuntimeError("Torch runtime is not available.")
        return tensor.detach().cpu().numpy()

    def info(self) -> dict[str, Any]:
        return {
            "backend_requested": self.backend_requested,
            "device_requested": self.device_requested,
            "backend_used": self.backend_used,
            "device_used": self.device_used,
            "active": self.active,
            "reason": self.reason,
            "stage_results": self.stage_results,
        }


def resolve_acceleration(config: AccelerationConfig | None = None) -> AccelerationRuntime:
    cfg = config or AccelerationConfig()
    backend_requested = cfg.backend
    device_requested = cfg.device

    if not cfg.enabled:
        return AccelerationRuntime(
            config=cfg,
            backend_requested=backend_requested,
            device_requested=device_requested,
            backend_used="numpy",
            device_used="cpu",
            active=False,
            reason="GPU acceleration disabled by configuration.",
        )

    if cfg.backend == "numpy":
        return AccelerationRuntime(
            config=cfg,
            backend_requested=backend_requested,
            device_requested=device_requested,
            backend_used="numpy",
            device_used="cpu",
            active=False,
            reason="NumPy backend explicitly requested.",
        )

    if not importlib.util.find_spec("torch"):
        return AccelerationRuntime(
            config=cfg,
            backend_requested=backend_requested,
            device_requested=device_requested,
            backend_used="numpy",
            device_used="cpu",
            active=False,
            reason="PyTorch is not installed; falling back to CPU.",
        )

    import torch  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if cfg.device == "auto":
        if cuda_available:
            device_used = "cuda"
        elif mps_available:
            device_used = "mps"
        else:
            device_used = "cpu"
    else:
        device_used = cfg.device

    if device_used == "cuda" and not cuda_available:
        reason = "CUDA device requested but unavailable; falling back to CPU."
        device_used = "cpu"
    elif device_used == "mps" and not mps_available:
        reason = "MPS device requested but unavailable; falling back to CPU."
        device_used = "cpu"
    elif device_used == "cpu":
        reason = "CPU device selected."
    else:
        reason = f"Using PyTorch on {device_used}."

    active = device_used in {"cuda", "mps"} and cfg.backend in {"auto", "torch"}
    backend_used = "torch" if active else "numpy"
    if cfg.backend == "torch" and not active and device_used == "cpu":
        reason = f"{reason} GPU path inactive."

    return AccelerationRuntime(
        config=cfg,
        backend_requested=backend_requested,
        device_requested=device_requested,
        backend_used=backend_used,
        device_used=device_used,
        active=active,
        reason=reason,
        torch_module=torch if active else None,
    )
