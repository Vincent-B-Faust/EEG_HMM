from pathlib import Path
import tempfile
import unittest

from sleep_hmm.cli import _synthetic_demo
from sleep_hmm.config import PipelineConfig
from sleep_hmm.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            eeg, emg = _synthetic_demo(fs=128.0, duration_sec=60.0 * 20, seed=7)
            config = PipelineConfig()
            config.output.output_dir = Path(tmp_dir) / "outputs"
            config.manifold.method = "pca"
            result = run_pipeline(eeg=eeg, emg=emg, fs=128.0, config=config)

            self.assertGreaterEqual(result.features.raw_table.shape[0], 20)
            self.assertEqual(set(result.clustering.methods), {"kmeans", "gmm", "hierarchical"})
            self.assertEqual(set(result.hmm), {3, 4, 5})
            self.assertEqual(result.hmm[3].hidden_states.shape[0], result.features.raw_table.shape[0])
            self.assertEqual(result.manifold.embedding.shape[1], 2)
            self.assertTrue((result.output_dir / "report.md").exists())
            self.assertTrue((result.output_dir / "hmm_model_comparison.csv").exists())
            self.assertTrue((result.output_dir / "session_view.html").exists())
            self.assertGreater(len(result.artifact_paths["figures"]), 5)
            self.assertEqual(len(result.artifact_paths["interactive"]), 1)
            self.assertIn("backend_used", result.runtime_info)
            self.assertIn("hmm", result.runtime_info.get("stage_results", {}))
            html_text = (result.output_dir / "session_view.html").read_text(encoding="utf-8")
            self.assertIn("spectrum-canvas", html_text)
            self.assertIn("Selected epoch spectrum", html_text)


if __name__ == "__main__":
    unittest.main()
