from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDSEARCH_PATH = REPO_ROOT / "pipeline" / "5_gridsearch.py"

spec = importlib.util.spec_from_file_location("gridsearch", GRIDSEARCH_PATH)
gridsearch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gridsearch
spec.loader.exec_module(gridsearch)


EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
DEPTH_EVENT = 1
DEPTH_SNAPSHOT_EVENT = 4

EVENT_DTYPE = np.dtype(
    [
        ("ev", "<u8"),
        ("exch_ts", "<i8"),
        ("local_ts", "<i8"),
        ("px", "<f8"),
        ("qty", "<f8"),
        ("order_id", "<u8"),
        ("ival", "<i8"),
        ("fval", "<f8"),
    ],
    align=True,
)
LATENCY_DTYPE = np.dtype(
    [
        ("req_ts", "<i8"),
        ("exch_ts", "<i8"),
        ("resp_ts", "<i8"),
        ("_padding", "<i8"),
    ],
    align=True,
)


def _write_hftbacktest_fixture(directory: Path) -> tuple[Path, Path, Path]:
    events = np.zeros(100, dtype=EVENT_DTYPE)
    for i in range(1, 51):
        timestamp = i * 100
        events[2 * (i - 1)] = (
            EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
            timestamp,
            timestamp,
            99.0 + i % 3,
            10.0 + i,
            0,
            0,
            0.0,
        )
        events[2 * (i - 1) + 1] = (
            EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
            timestamp + 1,
            timestamp + 1,
            101.0 + i % 3,
            20.0 - i / 4,
            0,
            0,
            0.0,
        )
    data_path = directory / "data.npz"
    np.savez(data_path, data=events)

    latency = np.array(
        [(0, 1, 2, 0), (10_000, 10_001, 10_002, 0)],
        dtype=LATENCY_DTYPE,
    )
    latency_path = directory / "latency.npz"
    np.savez(latency_path, data=latency)

    snapshot = np.array(
        [
            (
                EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT,
                0,
                0,
                99.0,
                10.0,
                0,
                0,
                0.0,
            ),
            (
                EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT,
                0,
                0,
                101.0,
                10.0,
                0,
                0,
                0.0,
            ),
        ],
        dtype=EVENT_DTYPE,
    )
    snapshot_path = directory / "snapshot.npz"
    np.savez(snapshot_path, data=snapshot)
    return data_path, latency_path, snapshot_path


def _validate_hftbacktest_fixture(
    data_path: Path, latency_path: Path, snapshot_path: Path
) -> None:
    with np.load(data_path) as archive:
        events = archive["data"]
    assert events.dtype == EVENT_DTYPE
    assert events.dtype.itemsize == 64
    assert events.shape == (100,)
    expected_flags = {
        EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
        EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
    }
    assert set(events["ev"].tolist()) == expected_flags
    assert np.all(np.diff(events["exch_ts"]) >= 0)
    assert np.all(events["local_ts"] >= events["exch_ts"])
    assert np.all(events["px"] > 0)
    assert np.all(events["qty"] > 0)

    with np.load(latency_path) as archive:
        latency = archive["data"]
    assert latency.dtype == LATENCY_DTYPE
    assert latency.dtype.itemsize == 32
    assert latency.shape == (2,)
    assert np.all(latency["req_ts"] <= latency["exch_ts"])
    assert np.all(latency["exch_ts"] <= latency["resp_ts"])

    with np.load(snapshot_path) as archive:
        snapshot = archive["data"]
    assert snapshot.dtype == EVENT_DTYPE
    assert snapshot.shape == (2,)
    assert set(snapshot["ev"].tolist()) == {
        EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT,
        EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT,
    }


class GridsearchPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(
            (REPO_ROOT / "pipeline" / "gridsearch_config.yaml").read_text(encoding="utf-8")
        )

    def test_partitions_are_ordered_and_non_overlapping(self) -> None:
        self.assertEqual(
            [phase for phase, _ in gridsearch._date_partitions(self.config)],
            ["train", "validation", "test"],
        )
        config = copy.deepcopy(self.config)
        config["gridsearch"]["validation_dates"] = [20250101, 20250102]
        with self.assertRaisesRegex(ValueError, "non-overlapping and ordered"):
            gridsearch._date_partitions(config)

    def test_explore_then_locked_test_is_strictly_two_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            for date in ("20250101", "20250102"):
                directory = base_root / "data" / config["exchange"] / "SYNTHUSDT"
                latency_directory = base_root / "latency" / config["exchange"] / "SYNTHUSDT"
                directory.mkdir(parents=True, exist_ok=True)
                latency_directory.mkdir(parents=True, exist_ok=True)
                (directory / f"SYNTHUSDT_{date}.npz").touch()
                (latency_directory / f"latency_{date}.npz").touch()

            exploration_runs = gridsearch._build_runs(config, phase_mode="explore")
            self.assertTrue(exploration_runs)
            self.assertEqual({run.phase for run in exploration_runs}, {"train", "validation"})
            self.assertFalse(
                (base_root / "data" / config["exchange"] / "SYNTHUSDT" / "SYNTHUSDT_20250103.npz").exists()
            )
            locked = next(
                run.candidate_id for run in exploration_runs if run.phase == "validation"
            )
            config["gridsearch"]["locked_candidate"] = [locked]

            with self.assertRaisesRegex(ValueError, "requires gridsearch_validation_summary"):
                gridsearch._build_runs(config, phase_mode="test")

            output_dir = Path(config["out_path"])
            output_dir.mkdir(parents=True, exist_ok=True)
            for run in exploration_runs:
                (output_dir / f"{run.name}.csv").write_text(
                    "timestamp,balance,position,price,fee\n"
                    "100,0,0,100,0\n"
                    "200,0,0,100,0\n",
                    encoding="utf-8",
                )
            gridsearch._summarize(
                str(output_dir),
                exploration_runs,
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0 for run in exploration_runs},
            )
            validation_summary = output_dir / "gridsearch_validation_summary.csv"
            self.assertTrue(validation_summary.exists())
            self.assertFalse((output_dir / "gridsearch_test_summary.csv").exists())

            directory = base_root / "data" / config["exchange"] / "SYNTHUSDT"
            latency_directory = base_root / "latency" / config["exchange"] / "SYNTHUSDT"
            (directory / "SYNTHUSDT_20250103.npz").touch()
            (latency_directory / "latency_20250103.npz").touch()

            test_runs = gridsearch._build_runs(config, phase_mode="test")
            self.assertEqual(len(test_runs), 1)
            self.assertEqual(test_runs[0].phase, "test")
            self.assertEqual(test_runs[0].candidate_id, locked)

            validation_summary.write_text(
                f"candidate_id,phase,status\n{locked},validation,failed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no successful validation candidates"):
                gridsearch._build_runs(config, phase_mode="test")

    def test_test_phase_requires_exactly_one_locked_candidate(self) -> None:
        config = copy.deepcopy(self.config)
        config["gridsearch"].pop("locked_candidate")
        with self.assertRaisesRegex(ValueError, "exactly one validation candidate"):
            gridsearch._build_runs(config, phase_mode="test")

        config["gridsearch"]["locked_candidate"] = ["one", "two"]
        with self.assertRaisesRegex(ValueError, "exactly one validation candidate"):
            gridsearch._build_runs(config, phase_mode="test")

        config["gridsearch"]["locked_candidate"] = ["TODO(Dennis)"]
        with self.assertRaisesRegex(ValueError, "replace gridsearch.locked_candidate"):
            gridsearch._build_runs(config, phase_mode="test")

    def test_real_npz_command_uses_hftbacktest_and_distinct_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            data_path, latency_path, snapshot_path = _write_hftbacktest_fixture(directory)
            _validate_hftbacktest_fixture(data_path, latency_path, snapshot_path)

            executed = []
            for name, algo, transform in (
                ("obi", "obi-static-alpha", "zscore"),
                ("vamp", "vamp", "ema"),
                ("baseline", "baseline", "none"),
            ):
                command = gridsearch._build_cmd(
                    binary="cargo",
                    name=name,
                    out_path=str(directory),
                    data_files=[str(data_path)],
                    latency_files=[str(latency_path)],
                    tick_size=1.0,
                    lot_size=1.0,
                    maker_fee=-0.00005,
                    taker_fee=0.0007,
                    queue_power=3.0,
                    grid={
                        "relative_half_spread": 0.005,
                        "relative_grid_interval": 0.005,
                        "grid_num": 2,
                        "order_qty": 1.0,
                        "max_position": 2.0,
                        "skew": 0.0,
                    },
                    time_ctrl={"elapse_ns": 100, "record_every": 1},
                    algo_cfg={"name": algo, "params": {"alpha_scale": 2.0}},
                    xform={"kind": transform, "window": 2, "ema_alpha": 0.5},
                    initial_snapshot=str(snapshot_path),
                )
                command[1:1] = ["run", "--quiet", "--example", "gridtrading_backtest_args", "--"]
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                output = next(directory.glob(f"{name}*.csv"))
                self.assertGreater(len(output.read_text(encoding="utf-8").splitlines()), 1)
                self.assertIn("timestamp,balance,position", output.read_text(encoding="utf-8"))

                manifest = json.loads(
                    (directory / f"{name}_run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["algorithm"], algo)
                self.assertEqual(manifest["transform"], transform)
                self.assertEqual(manifest["elapse_ns"], 100)
                self.assertEqual(manifest["record_every"], 1)
                executed.append(manifest["executed_strategy"])

            self.assertEqual(executed, ["obi-static-alpha", "vamp", "baseline"])


    def test_missing_latency_is_rejected_with_migration_note(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --latency-files"):
            gridsearch._build_cmd(
                "binary", "name", "out", ["data.npz"], [], 1.0, 1.0, 0.0, 0.0, 3.0,
                {"relative_half_spread": 0.1, "relative_grid_interval": 0.1, "grid_num": 1,
                 "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
                {}, {"name": "obi-static-alpha", "params": {}}, {"kind": "none"}, None,
            )

    def test_unknown_algorithm_and_transform_are_rejected(self) -> None:
        command = gridsearch._build_cmd(
            "cargo", "name", "out", ["data.npz"], ["latency.npz"], 1.0, 1.0, 0.0, 0.0, 3.0,
            {"relative_half_spread": 0.1, "relative_grid_interval": 0.1, "grid_num": 1,
             "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
            {}, {"name": "obi-static-alpha", "params": {}}, {"kind": "none"}, None,
        )
        command[1:1] = ["run", "--quiet", "--example", "gridtrading_backtest_args", "--"]
        for flag, value in (("--algo", "unknown"), ("--transform", "unknown")):
            invalid = list(command)
            invalid[invalid.index(flag) + 1] = value
            result = subprocess.run(invalid, cwd=REPO_ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("possible values", result.stderr)


if __name__ == "__main__":
    unittest.main()
