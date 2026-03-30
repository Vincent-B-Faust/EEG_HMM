from pathlib import Path

from sleep_hmm.cli import _synthetic_demo
from sleep_hmm.config import PipelineConfig
from sleep_hmm.pipeline import run_pipeline


def main() -> None:
    config = PipelineConfig()
    config.output.output_dir = Path("demo_outputs")
    config.windowing.window_sec = 30.0
    config.clustering.n_clusters = 3
    eeg, emg = _synthetic_demo(fs=128.0, duration_sec=60.0 * 30, seed=42)
    run_pipeline(eeg=eeg, emg=emg, fs=128.0, config=config)


if __name__ == "__main__":
    main()
