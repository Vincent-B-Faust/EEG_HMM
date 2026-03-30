from __future__ import annotations

import numpy as np
from scipy import signal

from .config import FilterConfig
from .types import PreprocessResult, SignalBundle


def _bandpass_filter(samples: np.ndarray, fs: float, band: tuple[float, float], order: int) -> np.ndarray:
    low, high = band
    nyquist = fs / 2.0
    if low <= 0 and high >= nyquist:
        return samples.copy()
    if high >= nyquist:
        high = nyquist - 1e-3
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, samples)


def _notch_filter(samples: np.ndarray, fs: float, freq: float, quality: float) -> np.ndarray:
    if freq <= 0 or freq >= fs / 2.0:
        return samples
    b, a = signal.iirnotch(freq, quality, fs=fs)
    return signal.filtfilt(b, a, samples)


def _zscore(samples: np.ndarray) -> np.ndarray:
    std = samples.std()
    if std < 1e-8:
        return samples - samples.mean()
    return (samples - samples.mean()) / std


def preprocess(
    eeg: np.ndarray,
    fs: float,
    emg: np.ndarray | None = None,
    config: FilterConfig | None = None,
) -> PreprocessResult:
    cfg = config or FilterConfig()
    eeg = np.asarray(eeg, dtype=float).squeeze()
    emg = np.asarray(emg, dtype=float).squeeze() if emg is not None else None
    if eeg.ndim != 1:
        raise ValueError("EEG must be a one-dimensional array.")
    if emg is not None and emg.shape != eeg.shape:
        raise ValueError("EEG and EMG must have the same length when both are provided.")

    raw_bundle = SignalBundle(
        eeg=eeg.copy(),
        emg=None if emg is None else emg.copy(),
        fs=float(fs),
        time=np.arange(eeg.size, dtype=float) / float(fs),
    )

    eeg_filtered = _bandpass_filter(eeg, fs, cfg.eeg_band, cfg.filter_order)
    if cfg.notch_freq is not None:
        eeg_filtered = _notch_filter(eeg_filtered, fs, cfg.notch_freq, cfg.notch_quality)

    emg_filtered = None
    if emg is not None:
        emg_filtered = _bandpass_filter(emg, fs, cfg.emg_band, cfg.filter_order)
        if cfg.notch_freq is not None:
            emg_filtered = _notch_filter(emg_filtered, fs, cfg.notch_freq, cfg.notch_quality)

    if cfg.standardize_signal:
        eeg_filtered = _zscore(eeg_filtered)
        if emg_filtered is not None:
            emg_filtered = _zscore(emg_filtered)

    filtered_bundle = SignalBundle(
        eeg=eeg_filtered,
        emg=emg_filtered,
        fs=float(fs),
        time=raw_bundle.time.copy(),
    )
    return PreprocessResult(raw=raw_bundle, filtered=filtered_bundle)
