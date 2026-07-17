from __future__ import annotations

import copy
import csv
import importlib.util
import json
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
SNAPSHOT_PATH = REPO_ROOT / "pipeline" / "3_snapshot.py"

spec = importlib.util.spec_from_file_location("gridsearch", GRIDSEARCH_PATH)
gridsearch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gridsearch
spec.loader.exec_module(gridsearch)

snapshot_spec = importlib.util.spec_from_file_location("snapshot_pipeline", SNAPSHOT_PATH)
snapshot_pipeline = importlib.util.module_from_spec(snapshot_spec)
assert snapshot_spec.loader is not None
sys.modules[snapshot_spec.name] = snapshot_pipeline
snapshot_spec.loader.exec_module(snapshot_pipeline)


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

    def test_real_npz_command_uses_hftbacktest_and_distinct_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            data_path, latency_path, snapshot_path = _write_hftbacktest_fixture(directory)
            _validate_hftbacktest_fixture(data_path, latency_path, snapshot_path)

            executed = []
            for name, algo, transform, params in (
                ("obi", "obi-static-alpha", "zscore", {"alpha_scale": 2.0}),
                ("vamp", "vamp", "ema", {"alpha_scale": 2.0}),
                ("vamp-effective", "vamp-effective", "zscore", {"alpha_scale": 3.0}),
                (
                    "weighted-depth",
                    "weighted-depth",
                    "sma",
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

    def test_snapshot_generator_writes_verified_as_of_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "source.npz"
            snapshot = directory / "snapshot.npz"
            source_events = np.array(
                [
                    (
                        EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
                        100,
                        101,
                        99.0,
                        10.0,
                        0,
                        0,
                        0.0,
                    ),
                    (
                        EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT,
                        199,
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
            np.savez(source, data=source_events)
            np.savez(snapshot, data=snapshot_events)

            snapshot_pipeline._write_snapshot_manifest(str(snapshot), [str(source)])
            sidecar = Path(f"{snapshot}.manifest.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["as_of_ns"], 200)
            self.assertEqual(manifest["snapshot_sha256"], gridsearch._sha256_file(str(snapshot)))
            self.assertEqual(manifest["source"], "generated-eod")
            gridsearch._validate_snapshot_file(str(snapshot), first_replay_ts=201)

            np.savez(snapshot, data=snapshot_events[:1])
            with self.assertRaisesRegex(ValueError, "snapshot_sha256 does not match"):
                gridsearch._validate_snapshot_file(str(snapshot), first_replay_ts=201)

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
                ("price", "depth prices must be finite and positive"),
                ("quantity", "depth quantities must be finite and positive"),
                ("nonfinite_price", "depth prices must be finite and positive"),
                ("event_time", "local_ts precedes exch_ts"),
                ("latency_time", "req_ts <= exch_ts <= resp_ts"),
            ):
                with self.subTest(case=case):
                    _write_partition_inputs(config, base_root, ("20250101", "20250102"))
                    if case == "dtype":
                        np.savez(data_path, data=np.array([1, 2, 3]))
                    elif case == "latency_time":
                        with np.load(latency_path) as archive:
                            latency = archive["data"].copy()
                        latency[0]["resp_ts"] = latency[0]["req_ts"] - 1
                        np.savez(latency_path, data=latency)
                    else:
                        with np.load(data_path) as archive:
                            events = archive["data"].copy()
                        if case == "flags":
                            events[0]["ev"] = 0
                        elif case == "price":
                            events[0]["px"] = -1.0
                        elif case == "quantity":
                            events[0]["qty"] = 0.0
                        elif case == "nonfinite_price":
                            events[0]["px"] = np.nan
                        else:
                            events[0]["local_ts"] = events[0]["exch_ts"] - 1
                        np.savez(data_path, data=events)
                    with self.assertRaisesRegex(ValueError, expected):
                        invoke()

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
            {"name": "vamp", "params": {"alpha_scale": "25.0"}}
        )
        self.assertEqual(normalized["params"]["alpha_scale"], 25.0)
        self.assertIsInstance(normalized["params"]["alpha_scale"], float)

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
