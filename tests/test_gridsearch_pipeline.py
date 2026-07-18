from __future__ import annotations

import copy
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDSEARCH_PATH = REPO_ROOT / "pipeline" / "5_gridsearch.py"
SNAPSHOT_MANIFEST_PATH = REPO_ROOT / "pipeline" / "snapshot_manifest.py"

spec = importlib.util.spec_from_file_location("gridsearch", GRIDSEARCH_PATH)
gridsearch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gridsearch
spec.loader.exec_module(gridsearch)

snapshot_spec = importlib.util.spec_from_file_location(
    "snapshot_manifest", SNAPSHOT_MANIFEST_PATH
)
snapshot_manifest = importlib.util.module_from_spec(snapshot_spec)
assert snapshot_spec.loader is not None
sys.modules[snapshot_spec.name] = snapshot_manifest
snapshot_spec.loader.exec_module(snapshot_manifest)


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
    events = np.zeros(200, dtype=EVENT_DTYPE)
    for i in range(1, 51):
        timestamp = i * 100
        bid_price = 98.0 + i % 2
        ask_price = 101.0 + i % 2
        bid_quantity = 0.0 if i == 10 else 10.0 + i
        offset = 4 * (i - 1)
        events[offset] = (
            EXCH_EVENT | BUY_EVENT | DEPTH_EVENT,
            timestamp,
            timestamp + 30,
            bid_price,
            bid_quantity,
            0,
            0,
            0.0,
        )
        events[offset + 1] = (
            EXCH_EVENT | SELL_EVENT | DEPTH_EVENT,
            timestamp + 10,
            timestamp + 20,
            ask_price,
            20.0 - i / 4,
            0,
            0,
            0.0,
        )
        events[offset + 2] = (
            LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
            timestamp + 10,
            timestamp + 20,
            ask_price,
            20.0 - i / 4,
            0,
            0,
            0.0,
        )
        events[offset + 3] = (
            LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
            timestamp,
            timestamp + 30,
            bid_price,
            bid_quantity,
            0,
            0,
            0.0,
        )
    data_path = directory / "data.npz"
    np.savez(data_path, data=events)

    latency = np.array(
        [(0, 1, 100, 0), (10, 11, 20, 0)],
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
    assert events.shape == (200,)
    expected_flags = {
        EXCH_EVENT | BUY_EVENT | DEPTH_EVENT,
        EXCH_EVENT | SELL_EVENT | DEPTH_EVENT,
        LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
        LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
    }
    assert set(events["ev"].tolist()) == expected_flags
    exch_rows = events["ev"] & EXCH_EVENT != 0
    local_rows = events["ev"] & LOCAL_EVENT != 0
    assert np.all(np.diff(events["exch_ts"][exch_rows]) >= 0)
    assert np.all(np.diff(events["local_ts"][local_rows]) >= 0)
    assert np.any(np.diff(events["exch_ts"]) < 0)
    assert np.any(np.diff(events["local_ts"]) < 0)
    assert np.all(events["local_ts"] >= events["exch_ts"])
    assert np.all(events["px"] > 0)
    assert np.all(events["qty"] >= 0)
    assert np.any(events["qty"] == 0)

    with np.load(latency_path) as archive:
        latency = archive["data"]
    assert latency.dtype == LATENCY_DTYPE
    assert latency.dtype.itemsize == 32
    assert latency.shape == (2,)
    assert np.all(latency["req_ts"] <= latency["exch_ts"])
    assert np.all(latency["exch_ts"] <= latency["resp_ts"])
    assert np.all(np.diff(latency["req_ts"]) >= 0)
    assert np.all(np.diff(latency["exch_ts"]) >= 0)
    assert np.any(np.diff(latency["resp_ts"]) < 0)

    with np.load(snapshot_path) as archive:
        snapshot = archive["data"]
    assert snapshot.dtype == EVENT_DTYPE
    assert snapshot.shape == (2,)
    assert set(snapshot["ev"].tolist()) == {
        EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT,
        EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT,
    }
    gridsearch._validate_event_file(str(data_path))
    gridsearch._validate_latency_file(str(latency_path))


def _write_partition_inputs(config: dict, base_root: Path, dates: tuple[str, ...]) -> None:
    data_directory = base_root / "data" / config["exchange"] / "SYNTHUSDT"
    latency_directory = base_root / "latency" / config["exchange"] / "SYNTHUSDT"
    data_directory.mkdir(parents=True, exist_ok=True)
    latency_directory.mkdir(parents=True, exist_ok=True)
    for date in dates:
        timestamp = int(date) * 1_000
        events = np.array(
            [
                (
                    EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
                    timestamp,
                    timestamp + 1,
                    99.0,
                    10.0,
                    0,
                    0,
                    0.0,
                ),
                (
                    EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
                    timestamp + 10,
                    timestamp + 11,
                    101.0,
                    10.0,
                    0,
                    0,
                    0.0,
                ),
            ],
            dtype=EVENT_DTYPE,
        )
        latency = np.array(
            [
                (timestamp, timestamp + 1, timestamp + 2, 0),
                (timestamp + 10, timestamp + 11, timestamp + 12, 0),
            ],
            dtype=LATENCY_DTYPE,
        )
        np.savez(data_directory / f"SYNTHUSDT_{date}.npz", data=events)
        np.savez(latency_directory / f"latency_{date}.npz", data=latency)


def _write_snapshot_sidecar(snapshot: Path, as_of_ns: int) -> Path:
    sidecar = Path(f"{snapshot}.manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "as_of_ns": as_of_ns,
                "snapshot_sha256": gridsearch._sha256_file(str(snapshot)),
                "source": "test-fixture",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def _write_snapshot_fixture(snapshot: Path, event_timestamp: int, as_of_ns: int) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    rows = np.array(
        [
            (
                EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT,
                event_timestamp,
                event_timestamp,
                99.0,
                10.0,
                0,
                0,
                0.0,
            ),
            (
                EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT,
                event_timestamp,
                event_timestamp,
                101.0,
                10.0,
                0,
                0,
                0.0,
            ),
        ],
        dtype=EVENT_DTYPE,
    )
    np.savez(snapshot, data=rows)
    _write_snapshot_sidecar(snapshot, as_of_ns)


def _write_completed_run(run: object, end_balance: float = 0.0) -> None:
    output = Path(gridsearch._expected_result_csv(run))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "timestamp,balance,position,price,fee\n"
        "100,0,0,100,0\n"
        f"200,{end_balance},0,100,0\n",
        encoding="utf-8",
    )
    Path(gridsearch._rust_manifest_path(run)).write_text(
        json.dumps(gridsearch._expected_rust_manifest(run), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gridsearch._write_artifact_manifest(run)


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
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))

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
            _write_partition_inputs(config, base_root, ("20250103",))

            with self.assertRaisesRegex(ValueError, "requires gridsearch_validation_summary"):
                gridsearch._build_runs(config, phase_mode="test")

            output_dir = Path(config["out_path"])
            for run in exploration_runs:
                _write_completed_run(run)
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

            test_runs = gridsearch._build_runs(config, phase_mode="test")
            self.assertEqual(len(test_runs), 1)
            self.assertEqual(test_runs[0].phase, "test")
            self.assertEqual(test_runs[0].candidate_id, locked)

            validation_summary.write_text(
                f"candidate_id,phase,status\n{locked},validation,failed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required"):
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

    def test_lock_rejects_changed_execution_assumptions_and_partition_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            _write_partition_inputs(
                config, base_root, ("20250101", "20250102", "20250103")
            )
            exploration_runs = gridsearch._build_runs(config, phase_mode="explore")
            validation_run = next(run for run in exploration_runs if run.phase == "validation")
            config["gridsearch"]["locked_candidate"] = [validation_run.candidate_id]
            for run in exploration_runs:
                _write_completed_run(run)
            gridsearch._summarize(
                config["out_path"],
                exploration_runs,
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0 for run in exploration_runs},
            )

            changed_configs = []
            changed_fee = copy.deepcopy(config)
            changed_fee["fees"]["maker"] = 0.123
            changed_configs.append(changed_fee)
            changed_queue = copy.deepcopy(config)
            changed_queue["queue_power"] = 99.0
            changed_configs.append(changed_queue)
            changed_timing = copy.deepcopy(config)
            changed_timing["elapse_ns"] = 777
            changed_configs.append(changed_timing)
            for changed in changed_configs:
                with self.assertRaisesRegex(ValueError, "current execution assumptions"):
                    gridsearch._build_runs(changed, phase_mode="test")

            changed_partition = copy.deepcopy(config)
            changed_partition["gridsearch"]["test_dates"] = [20250104, 20250104]
            _write_partition_inputs(changed_partition, base_root, ("20250104",))
            with self.assertRaisesRegex(ValueError, "partition_manifest does not match"):
                gridsearch._build_runs(changed_partition, phase_mode="test")

    def test_lock_rejects_forged_incomplete_duplicate_and_changed_input_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            _write_partition_inputs(
                config, base_root, ("20250101", "20250102", "20250103")
            )
            exploration_runs = gridsearch._build_runs(config, phase_mode="explore")
            validation_run = next(run for run in exploration_runs if run.phase == "validation")
            config["gridsearch"]["locked_candidate"] = [validation_run.candidate_id]
            for run in exploration_runs:
                _write_completed_run(run)
            gridsearch._summarize(
                config["out_path"],
                exploration_runs,
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0 for run in exploration_runs},
            )
            summary = Path(config["out_path"]) / "gridsearch_validation_summary.csv"
            valid_summary = summary.read_text(encoding="utf-8")

            summary.write_text(
                f"candidate_id,phase,status\n{validation_run.candidate_id},validation,ok\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required"):
                gridsearch._build_runs(config, phase_mode="test")

            locked_line = next(
                line
                for line in valid_summary.splitlines()[1:]
                if validation_run.candidate_id in line
            )
            summary.write_text(valid_summary + locked_line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one unique row"):
                gridsearch._build_runs(config, phase_mode="test")

            summary.write_text(valid_summary, encoding="utf-8")
            validation_latency = (
                base_root
                / "latency"
                / config["exchange"]
                / "SYNTHUSDT"
                / "latency_20250102.npz"
            )
            with np.load(validation_latency) as archive:
                changed_latency = archive["data"].copy()
            changed_latency[0]["resp_ts"] += 1
            np.savez(validation_latency, data=changed_latency)
            with self.assertRaisesRegex(ValueError, "input_manifest does not match"):
                gridsearch._build_runs(config, phase_mode="test")

    def test_lock_survives_checkout_data_and_output_path_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            original_root = directory / "original-data"
            moved_root = directory / "moved-data"
            original_out = directory / "original-output"
            moved_out = directory / "moved-output"

            config = copy.deepcopy(self.config)
            config["base_root"] = str(original_root)
            config["out_path"] = str(original_out)
            _write_partition_inputs(
                config,
                original_root,
                ("20250101", "20250102", "20250103"),
            )
            exploration_runs = gridsearch._build_runs(config, phase_mode="explore")
            for run in exploration_runs:
                _write_completed_run(run)
            summary = gridsearch._summarize(
                str(original_out),
                exploration_runs,
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0 for run in exploration_runs},
            )
            locked = summary[
                (summary["phase"] == "validation") & (summary["status"] == "ok")
            ].iloc[0]["candidate_id"]

            shutil.copytree(original_root, moved_root)
            shutil.copytree(original_out, moved_out)
            moved_config = copy.deepcopy(config)
            moved_config["base_root"] = str(moved_root)
            moved_config["out_path"] = str(moved_out)
            moved_config["gridsearch"]["locked_candidate"] = [locked]

            test_runs = gridsearch._build_runs(moved_config, phase_mode="test")
            self.assertEqual(len(test_runs), 1)
            self.assertEqual(test_runs[0].candidate_id, locked)
            original_path = str(original_root).replace("\\", "/")
            for manifest_json in (
                test_runs[0].candidate_manifest_json,
                test_runs[0].partition_manifest_json,
                test_runs[0].input_manifest_json,
            ):
                self.assertNotIn(original_path, manifest_json)

    def test_real_npz_command_uses_hftbacktest_and_distinct_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            data_path, latency_path, snapshot_path = _write_hftbacktest_fixture(directory)
            _validate_hftbacktest_fixture(data_path, latency_path, snapshot_path)

            executed = []
            for name, algo, transform, params in (
                ("obi", "obi-static-alpha", "zscore", {"alpha_scale": 2.0}),
                ("vamp", "vamp", "zscore", {"alpha_scale": 2.0}),
                ("vamp-effective", "vamp-effective", "zscore", {"alpha_scale": 3.0}),
                (
                    "weighted-depth",
                    "weighted-depth",
                    "zscore",
                    {"alpha_scale": 4.0, "target_qty_per_side": 10.0},
                ),
                ("baseline", "baseline", "none", {}),
            ):
                xform = {"kind": transform}
                if transform in ("sma", "zscore"):
                    xform["window"] = 2
                elif transform == "ema":
                    xform["ema_alpha"] = 0.5
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
                    algo_cfg={"name": algo, "params": params},
                    xform=xform,
                    initial_snapshot=str(snapshot_path),
                )
                if algo in ("vamp", "vamp-effective", "weighted-depth"):
                    self.assertIn("--alpha-scale", command)
                    self.assertEqual(
                        float(command[command.index("--alpha-scale") + 1]), params["alpha_scale"]
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
                for key, value in params.items():
                    self.assertEqual(manifest["executed_parameters"][key], value)
                executed.append(manifest["executed_strategy"])

            self.assertEqual(
                executed,
                ["obi-static-alpha", "vamp", "vamp-effective", "weighted-depth", "baseline"],
            )

    def test_quantized_zero_order_quantity_cannot_emit_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            data_path, latency_path, snapshot_path = _write_hftbacktest_fixture(directory)
            command = gridsearch._build_cmd(
                binary="cargo",
                name="invalid-quantity",
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
                    "order_qty": 0.1,
                    "max_position": 2.0,
                    "skew": 0.0,
                },
                time_ctrl={"elapse_ns": 100, "record_every": 1},
                algo_cfg={"name": "baseline", "params": {}},
                xform={"kind": "none"},
                initial_snapshot=str(snapshot_path),
            )
            command[1:1] = ["run", "--quiet", "--example", "gridtrading_backtest_args", "--"]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rounds to zero", result.stderr)
            self.assertFalse((directory / "invalid-quantity0.csv").exists())
            self.assertFalse((directory / "invalid-quantity_run_manifest.json").exists())


    def test_missing_latency_is_rejected_with_migration_note(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --latency-files"):
            gridsearch._build_cmd(
                "binary", "name", "out", ["data.npz"], [], 1.0, 1.0, 0.0, 0.0, 3.0,
                {"relative_half_spread": 0.1, "relative_grid_interval": 0.1, "grid_num": 1,
                 "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
                {}, {"name": "obi-static-alpha", "params": {}}, {"kind": "none"}, None,
            )

    def test_date_pairs_and_configured_snapshots_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            missing_latency = (
                base_root
                / "latency"
                / config["exchange"]
                / "SYNTHUSDT"
                / "latency_20250102.npz"
            )
            missing_latency.unlink()
            with self.assertRaisesRegex(ValueError, r"latency dates \['20250102'\]"):
                gridsearch._build_runs(config, phase_mode="explore")

            _write_partition_inputs(config, base_root, ("20250102",))
            config["initial_snapshot"] = "{base_root}/snapshots/{symbol}_{date}.npz"
            with self.assertRaisesRegex(ValueError, "configured initial snapshot does not exist"):
                gridsearch._build_runs(config, phase_mode="explore")

            invalid_snapshot = base_root / "snapshots" / "SYNTHUSDT_20250101.npz"
            invalid_snapshot.parent.mkdir(parents=True, exist_ok=True)
            np.savez(invalid_snapshot, data=np.array([1, 2, 3]))
            with self.assertRaisesRegex(ValueError, "expected pinned dtype"):
                gridsearch._build_runs(config, phase_mode="explore")

            first_train_event = int("20250101") * 1_000 + 1
            _write_snapshot_fixture(
                invalid_snapshot,
                event_timestamp=first_train_event - 100,
                as_of_ns=first_train_event,
            )
            with self.assertRaisesRegex(ValueError, "earlier than the first replay event"):
                gridsearch._build_runs(config, phase_mode="explore")

            _write_snapshot_fixture(
                invalid_snapshot,
                event_timestamp=first_train_event - 100,
                as_of_ns=first_train_event - 1,
            )
            validation_snapshot = base_root / "snapshots" / "SYNTHUSDT_20250102.npz"
            first_validation_event = int("20250102") * 1_000 + 1
            _write_snapshot_fixture(
                validation_snapshot,
                event_timestamp=first_validation_event - 100,
                as_of_ns=first_validation_event - 1,
            )
            runs = gridsearch._build_runs(config, phase_mode="explore")
            self.assertTrue(runs)

    def test_snapshot_semantics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "snapshot.npz"
            cases = (
                ("trade_rows", "only DEPTH_SNAPSHOT_EVENT"),
                ("missing_bid", "at least one bid and one ask"),
                ("crossed_book", "best_bid must be strictly less"),
                ("zero_quantity", "quantities must be finite and positive"),
                ("off_tick", "prices must align to tick_size"),
                ("off_lot", "quantities must align to lot_size"),
            )
            for name, expected in cases:
                with self.subTest(case=name):
                    _write_snapshot_fixture(snapshot, event_timestamp=100, as_of_ns=200)
                    with np.load(snapshot) as archive:
                        rows = archive["data"].copy()
                    if name == "trade_rows":
                        rows["ev"] = (
                            rows["ev"] & np.uint64(~0xFF & ((1 << 64) - 1))
                        ) | np.uint64(2)
                    elif name == "missing_bid":
                        rows[0]["ev"] = (
                            EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT
                        )
                    elif name == "crossed_book":
                        rows[0]["px"] = 101.0
                        rows[1]["px"] = 100.0
                    elif name == "zero_quantity":
                        rows[0]["qty"] = 0.0
                    elif name == "off_tick":
                        rows[0]["px"] = 99.5
                    elif name == "off_lot":
                        rows[0]["qty"] = 10.5
                    np.savez(snapshot, data=rows)
                    _write_snapshot_sidecar(snapshot, as_of_ns=200)
                    with self.assertRaisesRegex(ValueError, expected):
                        gridsearch._validate_snapshot_file(
                            str(snapshot),
                            first_replay_ts=300,
                            tick_size=1.0,
                            lot_size=1.0,
                        )

    def test_external_snapshot_utility_writes_verified_as_of_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            snapshot = directory / "snapshot.npz"
            snapshot_events = np.array(
                [
                    (
                        EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT,
                        200,
                        200,
                        99.0,
                        10.0,
                        0,
                        0,
                        0.0,
                    ),
                    (
                        EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT,
                        200,
                        200,
                        101.0,
                        10.0,
                        0,
                        0,
                        0.0,
                    ),
                ],
                dtype=EVENT_DTYPE,
            )
            np.savez(snapshot, data=snapshot_events)

            with mock.patch.object(
                sys,
                "argv",
                [
                    "snapshot_manifest.py",
                    "--snapshot",
                    str(snapshot),
                    "--as-of-ns",
                    "200",
                    "--source",
                    "external-test",
                ],
            ):
                self.assertEqual(snapshot_manifest.main(), 0)
            sidecar = Path(f"{snapshot}.manifest.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["as_of_ns"], 200)
            self.assertEqual(manifest["snapshot_sha256"], gridsearch._sha256_file(str(snapshot)))
            self.assertEqual(manifest["source"], "external-test")
            gridsearch._validate_snapshot_file(
                str(snapshot), first_replay_ts=201, tick_size=1.0, lot_size=1.0
            )

            with self.assertRaisesRegex(ValueError, "logical label, not a path"):
                snapshot_manifest.write_snapshot_manifest(snapshot, 200, str(directory))
            manifest["source"] = str(directory)
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "logical label, not a path"):
                gridsearch._validate_snapshot_file(
                    str(snapshot), first_replay_ts=201, tick_size=1.0, lot_size=1.0
                )
            manifest["source"] = "external-test"
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")

            tampered_snapshot = snapshot_events.copy()
            tampered_snapshot[0]["qty"] = 11.0
            np.savez(snapshot, data=tampered_snapshot)
            with self.assertRaisesRegex(ValueError, "snapshot_sha256 does not match"):
                gridsearch._validate_snapshot_file(
                    str(snapshot), first_replay_ts=201, tick_size=1.0, lot_size=1.0
                )

    def test_upstream_split_events_deletions_and_overlapping_latency_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))

            first_timestamp = int("20250101") * 1_000
            split_events = np.array(
                [
                    (
                        EXCH_EVENT | BUY_EVENT | DEPTH_EVENT,
                        first_timestamp,
                        first_timestamp + 30,
                        99.0,
                        0.0,
                        0,
                        0,
                        0.0,
                    ),
                    (
                        EXCH_EVENT | SELL_EVENT | DEPTH_EVENT,
                        first_timestamp + 10,
                        first_timestamp + 20,
                        101.0,
                        5.0,
                        0,
                        0,
                        0.0,
                    ),
                    (
                        LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
                        first_timestamp + 10,
                        first_timestamp + 20,
                        101.0,
                        5.0,
                        0,
                        0,
                        0.0,
                    ),
                    (
                        LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
                        first_timestamp,
                        first_timestamp + 30,
                        99.0,
                        0.0,
                        0,
                        0,
                        0.0,
                    ),
                ],
                dtype=EVENT_DTYPE,
            )
            data_path = (
                base_root
                / "data"
                / config["exchange"]
                / "SYNTHUSDT"
                / "SYNTHUSDT_20250101.npz"
            )
            np.savez(data_path, data=split_events)

            second_request = int("20250102") * 1_000
            latency = np.array(
                [
                    (first_timestamp, first_timestamp + 1, second_request + 100, 0),
                    (first_timestamp + 10, first_timestamp + 11, first_timestamp + 20, 0),
                ],
                dtype=LATENCY_DTYPE,
            )
            latency_path = (
                base_root
                / "latency"
                / config["exchange"]
                / "SYNTHUSDT"
                / "latency_20250101.npz"
            )
            np.savez(latency_path, data=latency)

            runs = gridsearch._build_runs(config, phase_mode="explore")
            self.assertTrue(runs)
            self.assertLess(split_events["exch_ts"][-1], split_events["exch_ts"][1])
            self.assertLess(latency["resp_ts"][1], latency["resp_ts"][0])

    def test_runtime_npz_and_empty_results_fail_closed_through_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            config_path = base_root / "gridsearch.yaml"
            data_path = (
                base_root
                / "data"
                / config["exchange"]
                / "SYNTHUSDT"
                / "SYNTHUSDT_20250101.npz"
            )
            latency_path = (
                base_root
                / "latency"
                / config["exchange"]
                / "SYNTHUSDT"
                / "latency_20250101.npz"
            )

            def invoke() -> int:
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "5_gridsearch.py",
                        "--phase",
                        "explore",
                        "--no-plots",
                        "-c",
                        str(config_path),
                    ],
                ):
                    return gridsearch.main()

            for case, expected in (
                ("dtype", "expected pinned dtype"),
                ("flags", "unsupported event flags"),
                ("l3_kind", "unsupported event flags"),
                ("trade_without_side", "depth and trade events require one side"),
                ("price", "market prices must be finite and positive"),
                ("quantity", "depth quantities must be finite and non-negative"),
                ("nonfinite_price", "market prices must be finite and positive"),
                ("event_time", "local_ts precedes exch_ts"),
                ("exchange_order", "exchange events are out of order"),
                ("local_order", "local events are out of order"),
                ("latency_time", "req_ts <= exch_ts <= resp_ts"),
                ("latency_axis", "interpolation axes must be monotonic"),
            ):
                with self.subTest(case=case):
                    _write_partition_inputs(config, base_root, ("20250101", "20250102"))
                    if case == "dtype":
                        np.savez(data_path, data=np.array([1, 2, 3]))
                    elif case in {"latency_time", "latency_axis"}:
                        with np.load(latency_path) as archive:
                            latency = archive["data"].copy()
                        if case == "latency_time":
                            latency[0]["resp_ts"] = latency[0]["req_ts"] - 1
                        else:
                            latency[1]["req_ts"] = latency[0]["req_ts"] - 1
                        np.savez(latency_path, data=latency)
                    else:
                        with np.load(data_path) as archive:
                            events = archive["data"].copy()
                        if case == "flags":
                            events[0]["ev"] = 0
                        elif case == "l3_kind":
                            events[0]["ev"] = EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | 10
                        elif case == "trade_without_side":
                            events[0]["ev"] = EXCH_EVENT | LOCAL_EVENT | 2
                        elif case == "price":
                            events[0]["px"] = -1.0
                        elif case == "quantity":
                            events[0]["qty"] = -1.0
                        elif case == "nonfinite_price":
                            events[0]["px"] = np.nan
                        elif case == "exchange_order":
                            events[1]["exch_ts"] = events[0]["exch_ts"] - 1
                        elif case == "local_order":
                            events[0]["local_ts"] = events[1]["local_ts"] + 100
                        else:
                            events[0]["local_ts"] = events[0]["exch_ts"] - 1
                        np.savez(data_path, data=events)
                    with self.assertRaisesRegex(ValueError, expected):
                        invoke()

            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            with np.load(data_path) as archive:
                valid_trade_events = archive["data"].copy()
            valid_trade_events[0]["ev"] = EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | 2
            np.savez(data_path, data=valid_trade_events)
            self.assertTrue(gridsearch._build_runs(config, phase_mode="explore"))

            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            config["gridsearch"]["skip_existing"] = True
            runs = gridsearch._build_runs(config, phase_mode="explore")
            run = runs[0]
            result = Path(gridsearch._expected_result_csv(run))
            result.parent.mkdir(parents=True, exist_ok=True)
            Path(gridsearch._rust_manifest_path(run)).write_text(
                json.dumps(gridsearch._expected_rust_manifest(run), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for case, contents, expected in (
                (
                    "empty",
                    "timestamp,balance,position,price,fee\n",
                    "contains no observations",
                ),
                (
                    "reversed_timestamp",
                    "timestamp,balance,position,price,fee\n200,0,0,100,0\n100,0,0,100,0\n",
                    "timestamps must be non-negative and monotonic",
                ),
                (
                    "nonfinite_equity",
                    "timestamp,balance,position,price,fee\n100,nan,0,100,0\n",
                    "non-finite value",
                ),
            ):
                with self.subTest(case=case):
                    result.write_text(contents, encoding="utf-8")
                    Path(gridsearch._artifact_manifest_path(run)).write_text(
                        "{}\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        invoke()

            result.write_text("timestamp,balance,position,price,fee\n", encoding="utf-8")
            summary = gridsearch._summarize(
                config["out_path"],
                [run],
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0},
            )
            self.assertEqual(summary.iloc[0]["status"], "invalid_artifact")
            self.assertIn("contains no observations", summary.iloc[0]["artifact_error"])

    def test_algorithm_parameter_schema_and_output_names_are_strict(self) -> None:
        normalized = gridsearch._normalize_algorithm(
            {"name": "vamp", "params": {"alpha_scale": "25.0"}},
            transform_kind="zscore",
        )
        self.assertEqual(normalized["params"]["alpha_scale"], 25.0)
        self.assertIsInstance(normalized["params"]["alpha_scale"], float)

        for section, field in (
            ("grid", "random_seed"),
            ("fees", "fee_currency"),
            ("defaults", "price_source"),
        ):
            with self.subTest(section=section):
                nested = copy.deepcopy(self.config)
                nested[section][field] = "unused"
                with self.assertRaisesRegex(ValueError, f"Unsupported {section} fields"):
                    gridsearch._validate_config_schema(nested)
        invalid_transform = copy.deepcopy(self.config)
        invalid_transform["transform"]["ema_alpha"] = 0.5
        with self.assertRaisesRegex(ValueError, "ema_alpha requires an ema transform"):
            gridsearch._validate_config_schema(invalid_transform)

        unsupported_mode = copy.deepcopy(self.config)
        unsupported_mode["gridsearch"]["max_position_mode"] = "unsupported-mode"
        with self.assertRaisesRegex(ValueError, "max_position_mode"):
            gridsearch._build_runs(unsupported_mode, phase_mode="explore")
        decorative_field = copy.deepcopy(self.config)
        decorative_field["gridsearch"]["random_seed"] = 7
        with self.assertRaisesRegex(ValueError, "Unsupported gridsearch configuration fields"):
            gridsearch._build_runs(decorative_field, phase_mode="explore")
        legacy_field = copy.deepcopy(self.config)
        legacy_field["date_from"] = 20250101
        with self.assertRaisesRegex(ValueError, "Unsupported top-level configuration fields"):
            gridsearch._build_runs(legacy_field, phase_mode="explore")

        with self.assertRaisesRegex(ValueError, "baseline requires transform=none"):
            gridsearch._build_cmd(
                "binary", "name", "out", ["data.npz"], ["latency.npz"],
                1.0, 1.0, 0.0, 0.0, 3.0,
                {"relative_half_spread": 0.1, "relative_grid_interval": 0.1,
                 "grid_num": 1, "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
                {}, {"name": "baseline", "params": {}}, {"kind": "zscore", "window": 3}, None,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported parameters for vamp"):
            gridsearch._build_cmd(
                "binary", "name", "out", ["data.npz"], ["latency.npz"],
                1.0, 1.0, 0.0, 0.0, 3.0,
                {"relative_half_spread": 0.1, "relative_grid_interval": 0.1,
                 "grid_num": 1, "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
                {}, {"name": "vamp", "params": {"look_depth_pct": 0.1}},
                {"kind": "none"}, None,
            )
        with self.assertRaisesRegex(ValueError, "alpha_scale is only applicable"):
            gridsearch._build_cmd(
                "binary", "name", "out", ["data.npz"], ["latency.npz"],
                1.0, 1.0, 0.0, 0.0, 3.0,
                {"relative_half_spread": 0.1, "relative_grid_interval": 0.1,
                 "grid_num": 1, "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
                {}, {"name": "vamp", "params": {"alpha_scale": 2.0}},
                {"kind": "ema", "ema_alpha": 0.5}, None,
            )
        inert_free_command = gridsearch._build_cmd(
            "binary", "name", "out", ["data.npz"], ["latency.npz"],
            1.0, 1.0, 0.0, 0.0, 3.0,
            {"relative_half_spread": 0.1, "relative_grid_interval": 0.1,
             "grid_num": 1, "order_qty": 1.0, "max_position": 1.0, "skew": 0.0},
            {}, {"name": "vamp", "params": {}},
            {"kind": "ema", "ema_alpha": 0.5}, None,
        )
        self.assertNotIn("--alpha-scale", inert_free_command)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            config["algo"] = {
                "name": "vamp",
                "params": {"vamp_depth_pct": 0.02, "alpha_scale": 1.0},
            }
            config["gridsearch"]["algo_params"] = {"alpha_scale": [1.0, 2.0]}
            config["transform"] = {"kind": "zscore", "window": 3}
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            runs = gridsearch._build_runs(config, phase_mode="explore")
            validation_runs = [run for run in runs if run.phase == "validation"]
            self.assertEqual(len({run.candidate_id for run in validation_runs}), 2)
            self.assertEqual(len({run.name for run in validation_runs}), 2)
            for run in validation_runs:
                command = gridsearch._command_for_spec(run)
                self.assertIn("--alpha-scale", command)
                self.assertEqual(
                    float(command[command.index("--alpha-scale") + 1]),
                    run.algo_cfg["params"]["alpha_scale"],
                )

    def test_skip_existing_verifies_and_summarizes_all_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            config["gridsearch"]["skip_existing"] = True
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            runs = gridsearch._build_runs(config, phase_mode="explore")
            for run in runs:
                _write_completed_run(run)
            config_path = base_root / "gridsearch.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                ["5_gridsearch.py", "--phase", "explore", "--no-plots", "-c", str(config_path)],
            ):
                self.assertEqual(gridsearch.main(), 0)
            for phase in ("train", "validation"):
                summary = base_root / "out" / f"gridsearch_{phase}_summary.csv"
                self.assertTrue(summary.exists())
                self.assertEqual(len(summary.read_text(encoding="utf-8").splitlines()), 3)

            sidecar = Path(gridsearch._artifact_manifest_path(runs[0]))
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["candidate_id"] = "stale"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                sys,
                "argv",
                ["5_gridsearch.py", "--phase", "explore", "--no-plots", "-c", str(config_path)],
            ):
                with self.assertRaisesRegex(ValueError, "refused stale or mismatched"):
                    gridsearch.main()

    def test_candidate_order_is_neutral_and_stale_test_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = copy.deepcopy(self.config)
            base_root = Path(tmpdir)
            config["base_root"] = str(base_root)
            config["out_path"] = str(base_root / "out")
            _write_partition_inputs(config, base_root, ("20250101", "20250102"))
            runs = gridsearch._build_runs(config, phase_mode="explore")
            validation_runs = sorted(
                (run for run in runs if run.phase == "validation"),
                key=lambda run: run.candidate_id,
            )
            self.assertEqual(len(validation_runs), 2)
            for run in runs:
                end_balance = 100.0 if run is validation_runs[1] else 0.0
                _write_completed_run(run, end_balance=end_balance)
            gridsearch._summarize(
                config["out_path"],
                runs,
                make_plots=False,
                usd_per_order=100.0,
                return_codes={run.name: 0 for run in runs},
            )
            validation_summary = Path(config["out_path"]) / "gridsearch_validation_summary.csv"
            with validation_summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["candidate_id"] for row in rows],
                [run.candidate_id for run in validation_runs],
            )
            self.assertLess(float(rows[0]["ret_abs"]), float(rows[1]["ret_abs"]))

            stale_test = Path(config["out_path"]) / "gridsearch_test_summary.csv"
            stale_test.write_text("candidate_id,status\nstale,ok\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refused an existing"):
                gridsearch._summarize(
                    config["out_path"],
                    runs,
                    make_plots=False,
                    usd_per_order=100.0,
                    return_codes={run.name: 0 for run in runs},
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

        transformed_baseline = list(command)
        transformed_baseline[transformed_baseline.index("--algo") + 1] = "baseline"
        transformed_baseline[transformed_baseline.index("--transform") + 1] = "zscore"
        result = subprocess.run(
            transformed_baseline, cwd=REPO_ROOT, text=True, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baseline requires --transform none", result.stderr)


if __name__ == "__main__":
    unittest.main()
