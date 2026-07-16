from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_PATH = REPO_ROOT / "pipeline" / "4_backtest.py"

spec = importlib.util.spec_from_file_location("synthetic_backtest", BACKTEST_PATH)
backtest = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backtest)


class SyntheticPipelineTests(unittest.TestCase):
    def test_config_rejects_delete_inputs_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad.yaml"
            config_path.write_text("fixture: fixtures/synthetic_l2.csv\ndelete_inputs_after: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "delete_inputs_after"):
                backtest.load_config(config_path)

    def test_build_command_is_deterministic(self) -> None:
        config = backtest.load_config(REPO_ROOT / "pipeline" / "backtest_config.yaml")
        command = backtest.build_command(REPO_ROOT, config, "baseline")
        self.assertEqual(command[:6], ["cargo", "run", "--quiet", "--bin", "statmm-bin", "--"])
        self.assertIn(str(REPO_ROOT / "fixtures" / "synthetic_l2.csv"), command)
        self.assertIn("baseline", command)

    def test_supported_smoke_run_writes_both_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            config = backtest.load_config(REPO_ROOT / "pipeline" / "backtest_config.yaml")
            config["output_dir"] = str(output_dir.relative_to(Path(tmpdir)))
            config_path = Path(tmpdir) / "smoke.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            subprocess.run(
                [sys.executable, str(BACKTEST_PATH), "-c", str(config_path)],
                cwd=REPO_ROOT,
                check=True,
            )

            expected_dir = REPO_ROOT / config["output_dir"]
            try:
                self.assertTrue((expected_dir / "baseline_summary.json").exists())
                self.assertTrue((expected_dir / "obi_summary.json").exists())
            finally:
                shutil.rmtree(expected_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
