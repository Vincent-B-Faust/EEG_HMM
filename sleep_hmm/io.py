from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from .config import InputConfig
from .types import SignalBundle


def _normalize_selector(selector: str | int | None) -> str | int | None:
    if isinstance(selector, str):
        stripped = selector.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return selector


def _select_array(name_to_array: dict[str, np.ndarray], preferred: str | None = None) -> np.ndarray:
    if preferred is not None:
        for key, value in name_to_array.items():
            if key == preferred:
                return value
        raise ValueError(f"MAT variable '{preferred}' was not found.")
    for value in name_to_array.values():
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            return value
    raise ValueError("No numeric array variable found in MAT file.")


def _load_mat_array(source: Path, variable: str | None) -> np.ndarray:
    try:
        payload = loadmat(source)
        candidates = {key: np.asarray(value) for key, value in payload.items() if not key.startswith("__")}
        return np.asarray(_select_array(candidates, variable)).squeeze()
    except NotImplementedError as exc:
        if not importlib.util.find_spec("h5py"):
            raise ValueError("MAT v7.3 files require 'h5py', which is not installed in this environment.") from exc
        import h5py  # type: ignore

        with h5py.File(source, "r") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError("MAT file does not contain readable datasets.")
            dataset_name = variable or keys[0]
            if dataset_name not in handle:
                raise ValueError(f"MAT variable '{dataset_name}' was not found.")
            return np.asarray(handle[dataset_name]).squeeze()


def _is_numeric_token(value: Any) -> bool:
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _detect_csv_separator(source: Path, separator: str | None) -> str:
    if separator is not None:
        return separator
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            candidates = [",", ";", "\t", " "]
            counts = {candidate: stripped.count(candidate) for candidate in candidates}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else ","
    return ","


def _detect_csv_header(source: Path, separator: str | None) -> bool:
    csv_separator = _detect_csv_separator(source, separator)
    preview = pd.read_csv(source, header=None, nrows=3, sep=csv_separator, engine="python")
    if preview.empty:
        return False
    first_row = preview.iloc[0].tolist()
    return not all(_is_numeric_token(item) for item in first_row)


def _resolve_column(frame: pd.DataFrame, selector: str | int | None, fallback_names: list[str], required: bool) -> pd.Series | None:
    if selector is not None:
        if selector in frame.columns:
            return frame[selector]
        if isinstance(selector, str):
            lower_map = {str(column).lower(): column for column in frame.columns}
            if selector.lower() in lower_map:
                return frame[lower_map[selector.lower()]]
        if required:
            raise ValueError(f"Column '{selector}' was not found in CSV input.")
        return None

    lower_map = {str(column).lower(): column for column in frame.columns}
    for fallback in fallback_names:
        if fallback.lower() in lower_map:
            return frame[lower_map[fallback.lower()]]
    numeric_columns = [column for column in frame.columns if np.issubdtype(frame[column].dtype, np.number)]
    if numeric_columns:
        return frame[numeric_columns[0]]
    if required:
        raise ValueError("No numeric EEG column could be identified in CSV input.")
    return None


def _resolve_channel_index(labels: list[str], selector: str | int | None, keywords: list[str], required: bool) -> int | None:
    selector = _normalize_selector(selector)
    if isinstance(selector, int):
        if selector < 0 or selector >= len(labels):
            raise ValueError(f"EDF channel index {selector} is out of range.")
        return selector
    if isinstance(selector, str):
        needle = selector.lower()
        for index, label in enumerate(labels):
            if label.lower() == needle:
                return index
        for index, label in enumerate(labels):
            if needle in label.lower():
                return index
        raise ValueError(f"EDF channel '{selector}' was not found.")
    for keyword in keywords:
        for index, label in enumerate(labels):
            if keyword in label.lower():
                return index
    if required and labels:
        return 0
    return None


def _read_ascii(handle: Any, width: int) -> str:
    return handle.read(width).decode("latin-1", errors="ignore").strip()


def _load_edf_channels(source: Path, eeg_selector: str | int | None, emg_selector: str | int | None) -> tuple[np.ndarray, np.ndarray | None, float]:
    with source.open("rb") as handle:
        _ = _read_ascii(handle, 8)
        _ = _read_ascii(handle, 80)
        _ = _read_ascii(handle, 80)
        _ = _read_ascii(handle, 8)
        _ = _read_ascii(handle, 8)
        header_bytes = int(float(_read_ascii(handle, 8) or 0))
        _ = _read_ascii(handle, 44)
        data_records = int(float(_read_ascii(handle, 8) or 0))
        duration_per_record = float(_read_ascii(handle, 8) or 1.0)
        n_signals = int(float(_read_ascii(handle, 4) or 0))

        labels = [_read_ascii(handle, 16) for _ in range(n_signals)]
        _ = [_read_ascii(handle, 80) for _ in range(n_signals)]
        _ = [_read_ascii(handle, 8) for _ in range(n_signals)]
        physical_min = np.array([float(_read_ascii(handle, 8) or 0) for _ in range(n_signals)], dtype=float)
        physical_max = np.array([float(_read_ascii(handle, 8) or 1) for _ in range(n_signals)], dtype=float)
        digital_min = np.array([float(_read_ascii(handle, 8) or -32768) for _ in range(n_signals)], dtype=float)
        digital_max = np.array([float(_read_ascii(handle, 8) or 32767) for _ in range(n_signals)], dtype=float)
        _ = [_read_ascii(handle, 80) for _ in range(n_signals)]
        samples_per_record = np.array([int(float(_read_ascii(handle, 8) or 0)) for _ in range(n_signals)], dtype=int)
        _ = [_read_ascii(handle, 32) for _ in range(n_signals)]

        handle.seek(header_bytes)
        if data_records <= 0:
            bytes_per_record = int(samples_per_record.sum() * 2)
            remaining = source.stat().st_size - header_bytes
            data_records = remaining // max(bytes_per_record, 1)

        raw_channels = [np.empty(data_records * samples_per_record[idx], dtype=np.int16) for idx in range(n_signals)]
        for record in range(data_records):
            for channel_idx in range(n_signals):
                sample_count = samples_per_record[channel_idx]
                block = handle.read(sample_count * 2)
                if len(block) != sample_count * 2:
                    raise ValueError("Unexpected end of EDF file while reading samples.")
                raw_channels[channel_idx][record * sample_count : (record + 1) * sample_count] = np.frombuffer(block, dtype="<i2")

    scale = np.where(digital_max != digital_min, (physical_max - physical_min) / (digital_max - digital_min), 1.0)
    offset = physical_min - digital_min * scale
    channels = [raw_channels[idx].astype(float) * scale[idx] + offset[idx] for idx in range(n_signals)]
    sample_rates = samples_per_record / max(duration_per_record, 1e-12)

    eeg_index = _resolve_channel_index(labels, eeg_selector, ["eeg"], required=True)
    emg_index = _resolve_channel_index(labels, emg_selector, ["emg"], required=False)
    assert eeg_index is not None
    eeg = channels[eeg_index]
    eeg_fs = float(sample_rates[eeg_index])

    emg = None
    if emg_index is not None:
        emg = channels[emg_index]
        emg_fs = float(sample_rates[emg_index])
        if abs(emg_fs - eeg_fs) > 1e-6:
            emg = resample_poly(emg, int(round(eeg_fs)), int(round(emg_fs)))
        target_length = min(len(eeg), len(emg))
        eeg = eeg[:target_length]
        emg = emg[:target_length]
    return np.asarray(eeg, dtype=float), None if emg is None else np.asarray(emg, dtype=float), eeg_fs


def load_signals(path: str | Path, fs: float | None = None, config: InputConfig | None = None) -> SignalBundle:
    source = Path(path)
    cfg = config or InputConfig()
    suffix = source.suffix.lower()
    if suffix == ".npz":
        payload = np.load(source, allow_pickle=False)
        eeg = np.asarray(payload["eeg"], dtype=float).squeeze()
        emg = np.asarray(payload["emg"], dtype=float).squeeze() if "emg" in payload else None
        sample_rate = float(payload["fs"]) if "fs" in payload else fs
    elif suffix == ".npy":
        eeg = np.asarray(np.load(source), dtype=float).squeeze()
        emg = None
        sample_rate = fs
    elif suffix == ".mat":
        eeg = _load_mat_array(source, cfg.mat_variable)
        emg = None
        sample_rate = fs
    elif suffix == ".edf":
        eeg, emg, sample_rate = _load_edf_channels(source, cfg.eeg_channel, cfg.emg_channel)
    elif suffix == ".csv":
        csv_separator = _detect_csv_separator(source, cfg.csv_separator)
        has_header = cfg.csv_has_header if cfg.csv_has_header is not None else _detect_csv_header(source, csv_separator)
        frame = pd.read_csv(source, header=0 if has_header else None, sep=csv_separator, engine="python")
        eeg_series = _resolve_column(frame, _normalize_selector(cfg.csv_eeg_column), ["eeg"], required=True)
        emg_series = _resolve_column(frame, _normalize_selector(cfg.csv_emg_column), ["emg"], required=False)
        if eeg_series is None:
            raise ValueError("CSV input does not contain a usable EEG column.")
        eeg = pd.to_numeric(eeg_series, errors="coerce").dropna().to_numpy(dtype=float)
        emg = None
        if emg_series is not None:
            emg_array = pd.to_numeric(emg_series, errors="coerce").dropna().to_numpy(dtype=float)
            target_length = min(len(eeg), len(emg_array))
            eeg = eeg[:target_length]
            emg = emg_array[:target_length]
        sample_rate = fs
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if sample_rate is None:
        raise ValueError("Sampling rate fs must be provided either in the file or as an argument.")
    time = np.arange(eeg.size, dtype=float) / float(sample_rate)
    return SignalBundle(eeg=eeg, emg=emg, fs=float(sample_rate), time=time)


def save_signal_bundle(bundle: SignalBundle, path: str | Path) -> None:
    payload = {"eeg": bundle.eeg, "fs": bundle.fs, "time": bundle.time}
    if bundle.emg is not None:
        payload["emg"] = bundle.emg
    np.savez(Path(path), **payload)
