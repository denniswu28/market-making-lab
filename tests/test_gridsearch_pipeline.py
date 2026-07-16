from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDSEARCH_PATH = REPO_ROOT / "pipeline" / "5_gridsearch.py"

spec = importlib.util.spec_from_file_location("gridsearch", GRIDSEARCH_PATH)
gridsearch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gridsearch
spec.loader.exec_module(gridsearch)


class GridsearchPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(
            (REPO_ROOT / "pipeline" / "gridsearch_config.yaml").read_text(encoding="utf-8")
        )

    def test_partitions_select_only_their_configured_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_root = Path(tmpdir)
            self.config["base_root"] = str(base_root)
            self.config["out_path"] = str(base_root / "out")
            for phase, date in (("train", "20250101"), ("validation", "20250102"), ("test", "20250103")):
                data_path = base_root / "data" / self.config["exchange"] / "SYNTHUSDT"
                latency_path = base_root / "latency" / self.config["exchange"] / "SYNTHUSDT"
                data_path.mkdir(parents=True, exist_ok=True)
                latency_path.mkdir(parents=True, exist_ok=True)
                (data_path / f"SYNTHUSDT_{date}.npz").touch()
                (latency_path / f"latency_{date}.npz").touch()

                runs = [run for run in gridsearch._build_runs(self.config) if run.phase == phase]
                self.assertTrue(runs)
                self.assertEqual([Path(path).stem[-8:] for path in runs[0].data_files], [date])
                self.assertEqual([Path(path).stem[-8:] for path in runs[0].latency_files], [date])

    def test_overlapping_partitions_are_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["gridsearch"]["validation_dates"] = [20250101, 20250102]

        with self.assertRaisesRegex(ValueError, "non-overlapping and ordered"):
            gridsearch._build_runs(config)

    def test_python_command_runs_configured_rust_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            command = gridsearch._build_cmd(
                binary="cargo",
                name="compatibility",
                out_path=str(output_path),
                data_files=[str(REPO_ROOT / "fixtures" / "synthetic_l2.csv")],
                latency_files=[],
                tick_size=1.0,
                lot_size=1.0,
                maker_fee=-0.00005,
                taker_fee=0.0007,
                queue_power=3.0,
                grid={
                    "relative_half_spread": 0.0005,
                    "relative_grid_interval": 0.0005,
                    "grid_num": 5,
                    "order_qty": 1.0,
                    "max_position": 2.0,
                    "skew": 0.0,
                },
                time_ctrl={"elapse_ns": 1_000_000_000, "record_every": 1},
                algo_cfg={"name": "obi-static-alpha", "params": {"normalize": True}},
                xform={"kind": "zscore", "window": 60},
                initial_snapshot=None,
            )
            command[1:1] = ["run", "--quiet", "--example", "gridtrading_backtest_args", "--"]

            subprocess.run(command, cwd=REPO_ROOT, check=True)

            self.assertTrue((output_path / "compatibility.csv").exists())
            self.assertTrue((output_path / "compatibility_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
