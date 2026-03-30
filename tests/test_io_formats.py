from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from sleep_hmm.interactive import build_compatible_config, run_file_pipeline
from sleep_hmm.io import load_signals


def _edf_field(value: str | int | float, width: int) -> bytes:
    return str(value).ljust(width)[:width].encode("latin-1")


def _write_test_edf(path: Path, eeg: np.ndarray, emg: np.ndarray, fs: int) -> None:
    labels = ["EEG Fpz-Cz", "EMG Chin"]
    signals = [eeg.astype(float), emg.astype(float)]
    n_signals = len(signals)
    samples_per_record = [fs for _ in range(n_signals)]
    duration_per_record = 1
    n_records = len(eeg) // fs
    if n_records <= 0 or n_records * fs != len(eeg) or len(eeg) != len(emg):
        raise ValueError("Test EDF writer expects equal-length signals with integer-second duration.")

    physical_min = [-200.0, -200.0]
    physical_max = [200.0, 200.0]
    digital_min = [-32768, -32768]
    digital_max = [32767, 32767]
    header_bytes = 256 + n_signals * 256

    with path.open("wb") as handle:
        handle.write(_edf_field("0", 8))
        handle.write(_edf_field("Test Patient", 80))
        handle.write(_edf_field("Test Recording", 80))
        handle.write(_edf_field("01.01.26", 8))
        handle.write(_edf_field("00.00.00", 8))
        handle.write(_edf_field(header_bytes, 8))
        handle.write(_edf_field("", 44))
        handle.write(_edf_field(n_records, 8))
        handle.write(_edf_field(duration_per_record, 8))
        handle.write(_edf_field(n_signals, 4))

        for label in labels:
            handle.write(_edf_field(label, 16))
        for _ in labels:
            handle.write(_edf_field("", 80))
        for _ in labels:
            handle.write(_edf_field("uV", 8))
        for value in physical_min:
            handle.write(_edf_field(value, 8))
        for value in physical_max:
            handle.write(_edf_field(value, 8))
        for value in digital_min:
            handle.write(_edf_field(value, 8))
        for value in digital_max:
            handle.write(_edf_field(value, 8))
        for _ in labels:
            handle.write(_edf_field("", 80))
        for value in samples_per_record:
            handle.write(_edf_field(value, 8))
        for _ in labels:
            handle.write(_edf_field("", 32))

        for record in range(n_records):
            for signal_index, signal_data in enumerate(signals):
                start = record * fs
                stop = start + fs
                segment = signal_data[start:stop]
                scale = (digital_max[signal_index] - digital_min[signal_index]) / (
                    physical_max[signal_index] - physical_min[signal_index]
                )
                digital = np.round((segment - physical_min[signal_index]) * scale + digital_min[signal_index]).astype("<i2")
                handle.write(digital.tobytes())


class IOFormatTestCase(unittest.TestCase):
    def test_load_csv_mat_and_edf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fs = 128.0
            t = np.arange(0, 2.0, 1 / fs)
            eeg = np.sin(2 * np.pi * 5 * t)
            emg = 0.3 * np.cos(2 * np.pi * 12 * t)

            csv_path = tmp_path / "signals.csv"
            pd.DataFrame({"eeg": eeg, "emg": emg}).to_csv(csv_path, index=False)
            csv_bundle = load_signals(csv_path, fs=fs)
            self.assertEqual(csv_bundle.eeg.shape, eeg.shape)
            self.assertIsNotNone(csv_bundle.emg)

            plain_csv_path = tmp_path / "plain.csv"
            pd.DataFrame(eeg).to_csv(plain_csv_path, index=False, header=False)
            plain_bundle = load_signals(plain_csv_path, fs=fs)
            self.assertEqual(len(plain_bundle.eeg), len(eeg))

            mat_path = tmp_path / "signals.mat"
            savemat(mat_path, {"emg_signal": eeg})
            mat_bundle = load_signals(mat_path, fs=fs)
            self.assertEqual(len(mat_bundle.eeg), len(eeg))

            edf_path = tmp_path / "signals.edf"
            _write_test_edf(edf_path, eeg=eeg, emg=emg, fs=int(fs))
            edf_bundle = load_signals(edf_path)
            self.assertAlmostEqual(edf_bundle.fs, fs)
            self.assertIsNotNone(edf_bundle.emg)
            self.assertEqual(len(edf_bundle.eeg), len(eeg))

    def test_run_file_pipeline_with_notebook_compatible_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fs = 200.0
            t = np.arange(0, 30.0, 1 / fs)
            eeg = np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 8 * t)
            emg = 0.3 * np.random.default_rng(42).standard_normal(t.size)
            csv_path = tmp_path / "signals.csv"
            pd.DataFrame({"eeg": eeg, "emg": emg}).to_csv(csv_path, index=False)

            config = build_compatible_config(
                filename=str(csv_path),
                fs=fs,
                use_dask=True,
                k_user=3,
                window_size=40,
                overlap=10,
                output_dir=tmp_path / "outputs",
                feature_mode="legacy",
                feature_scaling="minmax",
                window_strategy="samples",
            )
            result = run_file_pipeline(csv_path, fs=fs, config=config)
            self.assertEqual(list(result.features.raw_table.columns), ["peak_to_peak", "zero_crossing_rate", "peak_count", "iemg2"])
            self.assertTrue((result.output_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
