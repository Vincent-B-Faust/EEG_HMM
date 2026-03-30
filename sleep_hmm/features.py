from __future__ import annotations

import numpy as np
import pandas as pd

from .acceleration import AccelerationRuntime
from .config import FeatureConfig, WindowConfig
from .types import FeatureResult, SignalBundle, WindowResult
from .utils import dataframe_from_matrix, minmax_scale_matrix, standardize_matrix


def window_signals(bundle: SignalBundle, config: WindowConfig | None = None) -> WindowResult:
    cfg = config or WindowConfig()
    window_samples = int(round(cfg.window_sec * bundle.fs))
    overlap_samples = int(round(cfg.overlap_sec * bundle.fs))
    step = window_samples - overlap_samples
    if window_samples <= 0 or step <= 0:
        raise ValueError("Window length must be positive and overlap must be smaller than the window size.")
    if bundle.eeg.size < window_samples:
        raise ValueError("Signal is shorter than the requested window size.")

    start_indices = np.arange(0, bundle.eeg.size - window_samples + 1, step, dtype=int)
    eeg_windows = np.vstack([bundle.eeg[start : start + window_samples] for start in start_indices])
    emg_windows = None
    if bundle.emg is not None:
        emg_windows = np.vstack([bundle.emg[start : start + window_samples] for start in start_indices])
    start_times = start_indices / bundle.fs
    return WindowResult(
        eeg_windows=eeg_windows,
        emg_windows=emg_windows,
        start_indices=start_indices,
        start_times=start_times,
        window_sec=cfg.window_sec,
        overlap_sec=cfg.overlap_sec,
    )


def _energy_integral(window: np.ndarray, fs: float) -> float:
    return float(np.trapezoid(window**2, dx=1.0 / fs))


def _zero_crossing_rate(window: np.ndarray) -> float:
    signs = np.signbit(window)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return float(crossings / max(window.size - 1, 1))


def _peak_count(window: np.ndarray, scale: float) -> int:
    threshold = max(float(np.std(window) * scale), 1e-8)
    abs_window = np.abs(window)
    peaks = (abs_window[1:-1] > abs_window[:-2]) & (abs_window[1:-1] >= abs_window[2:]) & (abs_window[1:-1] >= threshold)
    return int(np.count_nonzero(peaks))


def _spectral_entropy(power: np.ndarray) -> float:
    total = power.sum()
    if total <= 0:
        return 0.0
    probs = power / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _bandpower(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def _uniform_trapezoid_numpy(values: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return np.trapezoid(values, dx=dx, axis=axis)


def _compute_fft_spectra_numpy(eeg_windows: np.ndarray, fs: float, max_hz: float) -> tuple[np.ndarray, np.ndarray]:
    sample_count = eeg_windows.shape[1]
    window_fn = np.hanning(sample_count)
    windowed = eeg_windows * window_fn[None, :]
    fft_values = np.fft.rfft(windowed, axis=1)
    normalization = max(float(np.sum(window_fn**2)), 1e-12)
    power = (np.abs(fft_values) ** 2) / normalization
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / fs)
    mask = freqs <= max_hz
    return freqs[mask], power[:, mask]


def _extract_base_feature_table_numpy(
    windows: WindowResult,
    fs: float,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    eeg_windows = windows.eeg_windows
    freqs, eeg_spectra = _compute_fft_spectra_numpy(eeg_windows, fs, cfg.max_spectrum_hz)
    freq_step = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    total_power = _uniform_trapezoid_numpy(eeg_spectra, dx=freq_step, axis=1) if eeg_spectra.size else np.zeros(eeg_windows.shape[0], dtype=float)
    dominant_frequency = freqs[np.argmax(eeg_spectra, axis=1)] if eeg_spectra.size else np.zeros(eeg_windows.shape[0], dtype=float)
    spectral_probs = np.divide(eeg_spectra, np.maximum(total_power[:, None], 1e-12))
    spectral_probs = np.where(spectral_probs > 0, spectral_probs, 1.0)
    spectral_entropy = -(np.divide(eeg_spectra, np.maximum(total_power[:, None], 1e-12)) * np.log2(spectral_probs)).sum(axis=1) if eeg_spectra.size else np.zeros(eeg_windows.shape[0], dtype=float)

    eeg_abs = np.abs(eeg_windows)
    eeg_threshold = np.maximum(np.std(eeg_windows, axis=1) * cfg.peak_height_std, 1e-8)
    eeg_peaks = (
        (eeg_abs[:, 1:-1] > eeg_abs[:, :-2])
        & (eeg_abs[:, 1:-1] >= eeg_abs[:, 2:])
        & (eeg_abs[:, 1:-1] >= eeg_threshold[:, None])
    )
    data: dict[str, np.ndarray] = {
        "eeg_energy": _uniform_trapezoid_numpy(eeg_windows**2, dx=1.0 / fs, axis=1),
        "eeg_peak_to_peak": np.ptp(eeg_windows, axis=1),
        "eeg_zcr": np.count_nonzero(np.signbit(eeg_windows[:, 1:]) != np.signbit(eeg_windows[:, :-1]), axis=1) / max(eeg_windows.shape[1] - 1, 1),
        "eeg_peak_count": eeg_peaks.sum(axis=1).astype(float),
        "eeg_dominant_frequency": dominant_frequency.astype(float),
        "eeg_spectral_entropy": spectral_entropy.astype(float),
        "eeg_total_power": total_power.astype(float),
        "eeg_iemg2_legacy": (np.trapezoid(eeg_windows**2, axis=1) / max(eeg_windows.shape[1], 1)).astype(float),
    }

    for band_name, (low, high) in cfg.band_definitions.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_power = _uniform_trapezoid_numpy(eeg_spectra[:, mask], dx=freq_step, axis=1)
        else:
            band_power = np.zeros(eeg_windows.shape[0], dtype=float)
        data[f"eeg_{band_name}_power"] = band_power
        data[f"eeg_{band_name}_ratio"] = np.divide(band_power, np.maximum(total_power, 1e-12))

    if windows.emg_windows is not None:
        emg_windows = windows.emg_windows
        emg_abs = np.abs(emg_windows)
        emg_threshold = np.maximum(np.std(emg_windows, axis=1) * cfg.peak_height_std, 1e-8)
        emg_peaks = (
            (emg_abs[:, 1:-1] > emg_abs[:, :-2])
            & (emg_abs[:, 1:-1] >= emg_abs[:, 2:])
            & (emg_abs[:, 1:-1] >= emg_threshold[:, None])
        )
        data.update(
            {
                "emg_energy": _uniform_trapezoid_numpy(emg_windows**2, dx=1.0 / fs, axis=1),
                "emg_peak_to_peak": np.ptp(emg_windows, axis=1),
                "emg_zcr": np.count_nonzero(np.signbit(emg_windows[:, 1:]) != np.signbit(emg_windows[:, :-1]), axis=1) / max(emg_windows.shape[1] - 1, 1),
                "emg_peak_count": emg_peaks.sum(axis=1).astype(float),
                "emg_iemg2_legacy": (np.trapezoid(emg_windows**2, axis=1) / max(emg_windows.shape[1], 1)).astype(float),
            }
        )

    return pd.DataFrame(data), freqs, eeg_spectra


def _uniform_trapezoid_torch(values: object, dx: float, dim: int, torch_module: object) -> object:
    return ((values.narrow(dim, 0, values.shape[dim] - 1) + values.narrow(dim, 1, values.shape[dim] - 1)) * 0.5).sum(dim=dim) * dx


def _extract_base_feature_table_torch(
    windows: WindowResult,
    fs: float,
    cfg: FeatureConfig,
    runtime: AccelerationRuntime,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    torch = runtime.torch_module
    if torch is None:
        raise RuntimeError("Torch runtime is not available.")

    eeg_windows = runtime.tensor(windows.eeg_windows)
    sample_count = eeg_windows.shape[1]
    window_fn = torch.hann_window(sample_count, periodic=False, device=runtime.device_used, dtype=eeg_windows.dtype)
    fft_values = torch.fft.rfft(eeg_windows * window_fn.unsqueeze(0), dim=1)
    normalization = torch.clamp(torch.sum(window_fn**2), min=1e-12)
    power = (torch.abs(fft_values) ** 2) / normalization
    freqs = torch.fft.rfftfreq(sample_count, d=1.0 / fs, device=runtime.device_used)
    mask = freqs <= cfg.max_spectrum_hz
    freqs_masked = freqs[mask]
    power_masked = power[:, mask]
    freq_step = float((freqs_masked[1] - freqs_masked[0]).item()) if freqs_masked.numel() > 1 else 1.0
    total_power = _uniform_trapezoid_torch(power_masked, dx=freq_step, dim=1, torch_module=torch) if power_masked.numel() else torch.zeros(eeg_windows.shape[0], device=runtime.device_used, dtype=eeg_windows.dtype)
    dominant_frequency = freqs_masked[torch.argmax(power_masked, dim=1)] if power_masked.numel() else torch.zeros(eeg_windows.shape[0], device=runtime.device_used, dtype=eeg_windows.dtype)
    prob = power_masked / torch.clamp(total_power.unsqueeze(1), min=1e-12)
    prob_safe = torch.where(prob > 0, prob, torch.ones_like(prob))
    spectral_entropy = -(prob * torch.log2(prob_safe)).sum(dim=1) if power_masked.numel() else torch.zeros(eeg_windows.shape[0], device=runtime.device_used, dtype=eeg_windows.dtype)

    eeg_abs = torch.abs(eeg_windows)
    eeg_threshold = torch.clamp(torch.std(eeg_windows, dim=1) * cfg.peak_height_std, min=1e-8)
    eeg_peaks = (
        (eeg_abs[:, 1:-1] > eeg_abs[:, :-2])
        & (eeg_abs[:, 1:-1] >= eeg_abs[:, 2:])
        & (eeg_abs[:, 1:-1] >= eeg_threshold.unsqueeze(1))
    )
    data: dict[str, np.ndarray] = {
        "eeg_energy": runtime.to_numpy(_uniform_trapezoid_torch(eeg_windows**2, dx=1.0 / fs, dim=1, torch_module=torch)),
        "eeg_peak_to_peak": runtime.to_numpy(torch.amax(eeg_windows, dim=1) - torch.amin(eeg_windows, dim=1)),
        "eeg_zcr": runtime.to_numpy(torch.sum((eeg_windows[:, 1:] < 0) != (eeg_windows[:, :-1] < 0), dim=1) / max(sample_count - 1, 1)),
        "eeg_peak_count": runtime.to_numpy(eeg_peaks.sum(dim=1).to(eeg_windows.dtype)),
        "eeg_dominant_frequency": runtime.to_numpy(dominant_frequency),
        "eeg_spectral_entropy": runtime.to_numpy(spectral_entropy),
        "eeg_total_power": runtime.to_numpy(total_power),
        "eeg_iemg2_legacy": runtime.to_numpy(_uniform_trapezoid_torch(eeg_windows**2, dx=1.0, dim=1, torch_module=torch) / max(sample_count, 1)),
    }

    for band_name, (low, high) in cfg.band_definitions.items():
        band_mask = (freqs_masked >= low) & (freqs_masked < high)
        if bool(torch.any(band_mask)):
            band_power = _uniform_trapezoid_torch(power_masked[:, band_mask], dx=freq_step, dim=1, torch_module=torch)
        else:
            band_power = torch.zeros(eeg_windows.shape[0], device=runtime.device_used, dtype=eeg_windows.dtype)
        data[f"eeg_{band_name}_power"] = runtime.to_numpy(band_power)
        data[f"eeg_{band_name}_ratio"] = runtime.to_numpy(band_power / torch.clamp(total_power, min=1e-12))

    if windows.emg_windows is not None:
        emg_windows = runtime.tensor(windows.emg_windows)
        emg_abs = torch.abs(emg_windows)
        emg_threshold = torch.clamp(torch.std(emg_windows, dim=1) * cfg.peak_height_std, min=1e-8)
        emg_peaks = (
            (emg_abs[:, 1:-1] > emg_abs[:, :-2])
            & (emg_abs[:, 1:-1] >= emg_abs[:, 2:])
            & (emg_abs[:, 1:-1] >= emg_threshold.unsqueeze(1))
        )
        data.update(
            {
                "emg_energy": runtime.to_numpy(_uniform_trapezoid_torch(emg_windows**2, dx=1.0 / fs, dim=1, torch_module=torch)),
                "emg_peak_to_peak": runtime.to_numpy(torch.amax(emg_windows, dim=1) - torch.amin(emg_windows, dim=1)),
                "emg_zcr": runtime.to_numpy(torch.sum((emg_windows[:, 1:] < 0) != (emg_windows[:, :-1] < 0), dim=1) / max(emg_windows.shape[1] - 1, 1)),
                "emg_peak_count": runtime.to_numpy(emg_peaks.sum(dim=1).to(emg_windows.dtype)),
                "emg_iemg2_legacy": runtime.to_numpy(_uniform_trapezoid_torch(emg_windows**2, dx=1.0, dim=1, torch_module=torch) / max(emg_windows.shape[1], 1)),
            }
        )

    return pd.DataFrame(data), runtime.to_numpy(freqs_masked), runtime.to_numpy(power_masked)


def extract_features(
    windows: WindowResult,
    fs: float,
    config: FeatureConfig | None = None,
    runtime: AccelerationRuntime | None = None,
) -> FeatureResult:
    cfg = config or FeatureConfig()
    n_windows, samples_per_window = windows.eeg_windows.shape
    if runtime is not None and runtime.should_accelerate("features", n_windows):
        try:
            full_table, freqs, eeg_spectra = _extract_base_feature_table_torch(windows, fs, cfg, runtime)
            runtime.record_stage("features", True, "Batched tensor features and FFT spectra on GPU.")
        except Exception as exc:
            full_table, freqs, eeg_spectra = _extract_base_feature_table_numpy(windows, fs, cfg)
            runtime.record_stage("features", False, f"GPU feature extraction fallback: {exc}")
    else:
        full_table, freqs, eeg_spectra = _extract_base_feature_table_numpy(windows, fs, cfg)
        if runtime is not None:
            runtime.record_stage("features", False, "CPU vectorized feature extraction.")

    if cfg.mode == "legacy":
        primary_prefix = "emg" if windows.emg_windows is not None else "eeg"
        raw_table = pd.DataFrame(
            {
                "peak_to_peak": full_table[f"{primary_prefix}_peak_to_peak"],
                "zero_crossing_rate": full_table[f"{primary_prefix}_zcr"],
                "peak_count": full_table[f"{primary_prefix}_peak_count"],
                "iemg2": full_table[f"{primary_prefix}_iemg2_legacy"],
            }
        )
    else:
        raw_table = full_table.drop(columns=[column for column in full_table.columns if column.endswith("_iemg2_legacy")], errors="ignore")

    feature_names = list(raw_table.columns)
    raw_matrix = raw_table.to_numpy(dtype=float)
    if cfg.scaling == "minmax":
        scaled_matrix, mean, std = minmax_scale_matrix(raw_matrix)
    else:
        scaled_matrix, mean, std = standardize_matrix(raw_matrix)
    scaled_table = dataframe_from_matrix(scaled_matrix, feature_names)
    metadata = pd.DataFrame(
        {
            "window_index": np.arange(n_windows, dtype=int),
            "start_time_sec": windows.start_times,
            "end_time_sec": windows.start_times + windows.window_sec,
        }
    )
    return FeatureResult(
        raw_table=raw_table,
        scaled_table=scaled_table,
        feature_names=feature_names,
        scale_mean=mean,
        scale_std=std,
        freqs=np.asarray(freqs if freqs is not None else []),
        eeg_spectra=np.asarray(eeg_spectra if eeg_spectra is not None else np.empty((0, 0))),
        metadata=metadata,
    )
