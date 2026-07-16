from __future__ import annotations

import copy
import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDSEARCH_PATH = REPO_ROOT / "pipeline" / "5_gridsearch.py"

spec = importlib.util.spec_from_file_location("gridsearch", GRIDSEARCH_PATH)
gridsearch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gridsearch
spec.loader.exec_module(gridsearch)


def _write_npz(path: Path, rows: list[tuple], descr: str, row_format: str) -> None:
    header = f"{{'descr': {descr}, 'fortran_order': False, 'shape': ({len(rows)},), }}"
    header_bytes = header.encode("ascii")
    header_bytes += b" " * ((64 - ((10 + len(header_bytes) + 1) % 64)) % 64) + b"\n"
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes
    npy += b"".join(struct.pack(row_format, *row) for row in rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("data.npy", npy)


def _write_hftbacktest_fixture(directory: Path) -> tuple[Path, Path]:
    events = []
    for i in range(1, 51):
        timestamp = i * 100
        events.extend(
            [
                ((1 << 30) | (1 << 29) | 1, timestamp, timestamp, 99.0 + i % 3, 10.0 + i, 0, 0, 0.0),
                ((1 << 30) | (1 << 28) | 1, timestamp + 1, timestamp + 1, 101.0 + i % 3, 20.0 - i / 4, 0, 0, 0.0),
            ]
        )
    data_path = directory / "data.npz"
    _write_npz(
        data_path,
        events,
        "[('ev', '<u8'), ('exch_ts', '<i8'), ('local_ts', '<i8'), ('px', '<f8'), "
        "('qty', '<f8'), ('order_id', '<u8'), ('ival', '<i8'), ('fval', '<f8'), ]",
        "<QqqddQqd8x",
    )
    latency_path = directory / "latency.npz"
    _write_npz(
        latency_path,
        [(0, 1, 2, 0), (10_000, 10_001, 10_002, 0)],
        "[('req_ts', '<i8'), ('exch_ts', '<i8'), ('resp_ts', '<i8'), ('_padding', '<i8'), ]",
        "<qqqq",
    )
    return data_path, latency_path


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

    def test_test_phase_requires_one_locked_validation_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            for date in ("20250101", "20250102", "20250103"):
                directory = base_root / "data" / config["exchange"] / "SYNTHUSDT"
                latency_directory = base_root / "latency" / config["exchange"] / "SYNTHUSDT"
                directory.mkdir(parents=True, exist_ok=True)
                latency_directory.mkdir(parents=True, exist_ok=True)
                (directory / f"SYNTHUSDT_{date}.npz").touch()
                (latency_directory / f"latency_{date}.npz").touch()

            config["gridsearch"].pop("locked_candidate")
            with self.assertRaisesRegex(ValueError, "exactly one validation candidate"):
                gridsearch._build_runs(config)

            config["gridsearch"]["locked_candidate"] = ["one", "two"]
            with self.assertRaisesRegex(ValueError, "exactly one validation candidate"):
                gridsearch._build_runs(config)

            config["gridsearch"]["locked_candidate"] = ["not-a-validation-candidate"]
            with self.assertRaisesRegex(ValueError, "must match a candidate evaluated on validation"):
                gridsearch._build_runs(config)

            config["gridsearch"]["algo_params"] = {"alpha_scale": [25.0]}
            algo = {"name": config["algo"]["name"], "params": dict(config["algo"]["params"], alpha_scale=25.0)}
            grid = dict(config["grid"])
            grid["order_qty"] = 1.0
            grid["max_position"] = 5.0
            grid["skew"] = 0.0001
            locked = gridsearch._candidate_id("SYNTHUSDT", grid, algo, config["transform"])
            config["gridsearch"]["locked_candidate"] = [locked]
            runs = gridsearch._build_runs(config)
            self.assertEqual(len([run for run in runs if run.phase == "test"]), 1)
            self.assertEqual({run.candidate_id for run in runs if run.phase == "test"}, {locked})
            self.assertTrue(all(run.phase != "test" or run.candidate_id == locked for run in runs))

    def test_real_npz_command_uses_hftbacktest_and_distinct_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            data_path, latency_path = _write_hftbacktest_fixture(directory)
            outputs = []
            for name, algo, transform in (
                ("obi", "obi-static-alpha", "zscore"),
                ("vamp", "vamp", "ema"),
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
                    initial_snapshot=str(data_path),
                )
                command[1:1] = ["run", "--quiet", "--example", "gridtrading_backtest_args", "--"]
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                output = next(directory.glob(f"{name}*.csv"))
                self.assertGreater(len(output.read_text(encoding="utf-8").splitlines()), 1)
                outputs.append(output.read_text(encoding="utf-8"))
            self.assertTrue(all("timestamp,balance,position" in output for output in outputs))

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
