from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_HFTBACKTEST_COMMIT = "6557e564ac984c46405a0ddfd08272f5009abc2e"
ENGINE_SOURCE_PATHS = (
    "Cargo.lock",
    "Cargo.toml",
    "examples/gridtrading_backtest_args.rs",
    "src/algo.rs",
)
ALGORITHM_PARAMETER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "obi-static-alpha": {
        "look_depth_pct": 0.02,
        "normalize": True,
        "alpha_scale": 50.0,
    },
    "vamp": {"vamp_depth_pct": 0.02, "alpha_scale": 50.0},
    "vamp-effective": {"vamp_depth_pct": 0.02, "alpha_scale": 50.0},
    "weighted-depth": {"target_qty_per_side": 500.0, "alpha_scale": 50.0},
}
TRANSFORM_PARAMETER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "none": {},
    "sma": {"window": 300},
    "ema": {"ema_alpha": 0.1},
    "zscore": {"window": 300},
}
EVENT_DTYPE_FIELDS = (
    "ev",
    "exch_ts",
    "local_ts",
    "px",
    "qty",
    "order_id",
    "ival",
    "fval",
)

# --------------------------- helpers: dates, files, params ---------------------------

def _date_range(d0: int, d1: int) -> List[str]:
    s = datetime.strptime(str(d0), "%Y%m%d")
    e = datetime.strptime(str(d1), "%Y%m%d")
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out

def _date_partitions(cfg: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    gridsearch = cfg.get("gridsearch", {}) or {}
    partitions: List[Tuple[str, List[str]]] = []
    previous_end: Optional[datetime] = None
    for phase in ("train", "validation", "test"):
        dates = gridsearch.get(f"{phase}_dates")
        if not isinstance(dates, (list, tuple)) or len(dates) != 2:
            raise ValueError(f"gridsearch.{phase}_dates must contain a start and end date")
        try:
            start = datetime.strptime(str(dates[0]), "%Y%m%d")
            end = datetime.strptime(str(dates[1]), "%Y%m%d")
        except ValueError as error:
            raise ValueError(f"gridsearch.{phase}_dates must use YYYYMMDD dates") from error
        if start > end:
            raise ValueError(f"gridsearch.{phase}_dates must be ordered")
        if previous_end is not None and start <= previous_end:
            raise ValueError("gridsearch train, validation, and test dates must be non-overlapping and ordered")
        partitions.append((phase, _date_range(int(dates[0]), int(dates[1]))))
        previous_end = end
    return partitions

def _files_for(base_root: str, exchange: str, symbol: str, yyyymmdd: str) -> Dict[str, str]:
    data = os.path.join(base_root, "data", exchange, symbol, f"{symbol}_{yyyymmdd}.npz")
    lat  = os.path.join(base_root, "latency", exchange, symbol, f"latency_{yyyymmdd}.npz")
    return {"date": yyyymmdd, "data": data, "lat": lat}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str) -> Dict[str, Any]:
    resolved = os.path.abspath(path)
    stat = os.stat(resolved)
    return {
        "path": resolved.replace("\\", "/"),
        "size": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _engine_source_identity() -> Dict[str, Any]:
    digest = hashlib.sha256()
    for relative_path in ENGINE_SOURCE_PATHS:
        path = REPO_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "source_paths": list(ENGINE_SOURCE_PATHS),
        "source_sha256": digest.hexdigest(),
        "upstream_repository": "nkaz001/hftbacktest",
        "upstream_commit": UPSTREAM_HFTBACKTEST_COMMIT,
    }


def _binary_identity(binary: str) -> Dict[str, Any]:
    configured = binary.replace("\\", "/")
    if os.path.isabs(binary) or os.path.dirname(binary):
        resolved = os.path.abspath(binary)
    else:
        resolved = shutil.which(binary) or binary
    identity: Dict[str, Any] = {
        "configured": configured,
        "resolved": os.path.abspath(resolved).replace("\\", "/"),
    }
    if os.path.isfile(resolved):
        identity["sha256"] = _sha256_file(resolved)
    else:
        identity["sha256"] = "MISSING"
    return identity


def _manifest(value: Dict[str, Any]) -> Tuple[str, str]:
    encoded = _canonical_json(value)
    return encoded, _sha256_text(encoded)


def _normalize_algorithm(algo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    unknown_top_level = set(algo_cfg) - {"name", "params"}
    if unknown_top_level:
        raise ValueError(f"Unsupported algorithm fields: {sorted(unknown_top_level)}")
    name = str(algo_cfg.get("name") or "").lower()
    if name == "glft-simple":
        raise ValueError(
            "glft-simple is not supported by the common grid-search CLI because its "
            "notional-sizing semantics differ; migrate it to a dedicated research command"
        )
    if name not in ALGORITHM_PARAMETER_DEFAULTS:
        raise ValueError(f"Unsupported algo: {name}")
    supplied = dict(algo_cfg.get("params") or {})
    allowed = set(ALGORITHM_PARAMETER_DEFAULTS[name])
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unsupported parameters for {name}: {sorted(unknown)}")
    params = {**ALGORITHM_PARAMETER_DEFAULTS[name], **supplied}
    if "normalize" in params and not isinstance(params["normalize"], bool):
        raise ValueError(f"{name}.normalize must be a boolean")
    for key in ("look_depth_pct", "vamp_depth_pct", "target_qty_per_side"):
        if key in params and (not math.isfinite(float(params[key])) or float(params[key]) <= 0):
            raise ValueError(f"{name}.{key} must be positive and finite")
    if "alpha_scale" in params and not math.isfinite(float(params["alpha_scale"])):
        raise ValueError(f"{name}.alpha_scale must be finite")
    return {"name": name, "params": params}


def _normalize_transform(xform: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    kind = str(xform.get("kind") or "").lower()
    if kind not in TRANSFORM_PARAMETER_DEFAULTS:
        raise ValueError(f"Unsupported transform: {kind}")
    allowed = {"kind", *TRANSFORM_PARAMETER_DEFAULTS[kind]}
    unknown = set(xform) - allowed
    if unknown:
        raise ValueError(f"Unsupported parameters for {kind} transform: {sorted(unknown)}")
    if algorithm == "baseline" and kind != "none":
        raise ValueError("baseline requires transform=none; transforms are not executed by baseline")
    normalized = {"kind": kind, **TRANSFORM_PARAMETER_DEFAULTS[kind]}
    normalized.update({key: value for key, value in xform.items() if key != "kind"})
    if "window" in normalized:
        normalized["window"] = int(normalized["window"])
        if normalized["window"] <= 0:
            raise ValueError(f"{kind}.window must be positive")
    if "ema_alpha" in normalized:
        normalized["ema_alpha"] = float(normalized["ema_alpha"])
        if not 0 < normalized["ema_alpha"] <= 1:
            raise ValueError("ema.ema_alpha must be in (0, 1]")
    return normalized

def _fmt_float_for_name(x: float) -> str:
    if x == 0 or not math.isfinite(x):
        return "0"
    return f"{x:.8g}"  # compact but stable

def _transform_tag(xform: Dict[str, Any]) -> str:
    k = xform["kind"].lower()
    if k == "ema":
        return f"emaA{_fmt_float_for_name(float(xform.get('ema_alpha', 0.1)))}"
    if k in ("sma", "zscore"):
        return f"{k[:1]}W{int(xform.get('window', 300))}"
    return k  # "none"

def _algo_tag(algo_cfg: Dict[str, Any]) -> str:
    nm = (algo_cfg.get("name") or "").lower()
    p = (algo_cfg.get("params") or {})
    f = lambda x: _fmt_float_for_name(float(x))
    if nm == "obi-static-alpha":
        nflag = "n1" if p.get("normalize", True) else "n0"
        ld = f(p.get("look_depth_pct", 0.02))
        c  = f(p.get("alpha_scale", 50.0))
        return f"obi-{nflag}-ld{ld}-c{c}"
    if nm == "vamp":
        dp = f(p.get("vamp_depth_pct", 0.02))
        c = f(p.get("alpha_scale", 50.0))
        return f"vamp-dp{dp}-c{c}"
    if nm == "vamp-effective":
        dp = f(p.get("vamp_depth_pct", 0.02))
        c = f(p.get("alpha_scale", 50.0))
        return f"vampe-dp{dp}-c{c}"
    if nm == "weighted-depth":
        t = f(p.get("target_qty_per_side", 500.0))
        c = f(p.get("alpha_scale", 50.0))
        return f"wdepth-t{t}-c{c}"
    if nm == "glft-simple":
        w = int(p.get("glft_vol_window", 6000))
        s = f(p.get("glft_vol_scale", 0.5))
        return f"glft-w{w}-s{s}"
    return nm

def _name_for_run(sym: str, rhs: float, rgi: float, n: int, skew: float,
                  xform: Dict[str, Any], algo_cfg: Dict[str, Any]) -> str:
    ttag = _transform_tag(xform)
    atag = _algo_tag(algo_cfg)
    return (f"{sym}__{atag}__x-{ttag}"
            f"__rhs{_fmt_float_for_name(rhs)}"
            f"_rgi{_fmt_float_for_name(rgi)}"
            f"_n{n}_sk{_fmt_float_for_name(skew)}")

def _candidate_manifest(
    cfg: Dict[str, Any],
    sym: str,
    grid: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    xform: Dict[str, Any],
    tick_size: float,
    lot_size: float,
    time_ctrl: Dict[str, Any],
    engine_identity: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "symbol": sym,
        "strategy": {"grid": grid, "algorithm": algo_cfg, "transform": xform},
        "market_model": {
            "tick_size": tick_size,
            "lot_size": lot_size,
            "maker_fee": cfg["fees"]["maker"],
            "taker_fee": cfg["fees"]["taker"],
            "queue_model": {
                "kind": "PowerProbQueueFunc3",
                "power": cfg.get("queue_power", 3.0),
            },
            "latency_model": {"kind": "IntpOrderLatency", "offset": 0},
            "exchange_kind": "NoPartialFillExchange",
            "asset_type": {"kind": "LinearAsset", "contract_size": 1.0},
        },
        "timing": time_ctrl,
        "engine": engine_identity,
    }


def _candidate_id(sym: str, candidate_manifest_hash: str) -> str:
    return f"{sym}-{candidate_manifest_hash[:16]}"


def _snapshot_path(cfg: Dict[str, Any], symbol: str, first_date_yyyymmdd: str) -> Optional[str]:
    ini = cfg.get("initial_snapshot")
    if not ini:
        return None

    def _fmt(value: str) -> str:
        return value.format(
            base_root=cfg["base_root"],
            exchange=cfg["exchange"],
            symbol=symbol,
            date=first_date_yyyymmdd,
        )

    if isinstance(ini, str):
        return _fmt(ini)
    if isinstance(ini, dict):
        value = ini.get(symbol) or ini.get("*")
        if not value:
            raise ValueError(f"initial_snapshot is configured but has no entry for {symbol}")
        return _fmt(value)
    raise ValueError("initial_snapshot must be null, a path template, or a symbol-to-template map")


def _validate_snapshot_file(path: str) -> None:
    if not os.path.isfile(path):
        raise ValueError(f"configured initial snapshot does not exist: {path}")
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError("missing data array")
            snapshot = archive["data"]
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid HftBacktest initial snapshot {path}: {error}") from error
    if snapshot.ndim != 1 or snapshot.size == 0:
        raise ValueError(f"invalid HftBacktest initial snapshot {path}: expected a non-empty 1D array")
    if snapshot.dtype.names != EVENT_DTYPE_FIELDS or snapshot.dtype.itemsize != 64:
        raise ValueError(
            f"invalid HftBacktest initial snapshot {path}: expected the pinned 64-byte event schema"
        )


def _resolve_initial_snapshot(cfg: Dict[str, Any], symbol: str, first_date_yyyymmdd: str) -> Optional[str]:
    path = _snapshot_path(cfg, symbol, first_date_yyyymmdd)
    if path is not None:
        _validate_snapshot_file(path)
    return path


def _partition_manifest(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    partitions: Dict[str, Any] = {}
    for phase, dates in _date_partitions(cfg):
        partitions[phase] = {
            "dates": dates,
            "data_files": [
                os.path.abspath(_files_for(cfg["base_root"], cfg["exchange"], symbol, date)["data"])
                .replace("\\", "/")
                for date in dates
            ],
            "latency_files": [
                os.path.abspath(_files_for(cfg["base_root"], cfg["exchange"], symbol, date)["lat"])
                .replace("\\", "/")
                for date in dates
            ],
            "initial_snapshot": (
                os.path.abspath(_snapshot_path(cfg, symbol, dates[0])).replace("\\", "/")
                if _snapshot_path(cfg, symbol, dates[0]) is not None
                else None
            ),
        }
    return {
        "schema_version": 1,
        "exchange": cfg["exchange"],
        "symbol": symbol,
        "partitions": partitions,
    }


def _input_manifest(
    phase: str,
    symbol: str,
    data_files: List[str],
    latency_files: List[str],
    initial_snapshot: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "symbol": symbol,
        "data_files": [_file_identity(path) for path in data_files],
        "latency_files": [_file_identity(path) for path in latency_files],
        "initial_snapshot": _file_identity(initial_snapshot) if initial_snapshot else None,
    }

def _read_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # timestamp → datetime index (try ns first; if that fails, infer)
    ts = pd.to_datetime(df["timestamp"], unit="ns", errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.index = ts

    # Normalize column names / dtypes
    if "price" not in df.columns and "mid_price" in df.columns:
        df = df.rename(columns={"mid_price": "price"})
    for c in ("balance", "position", "price", "fee"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _approx_daily_trades(df: pd.DataFrame, usd_per_order: float) -> float:
    pos = df["position"]
    mid = df["price"]
    mid_1d_last = mid.resample("1D").last()
    notional_qty = pos.diff().abs().rolling("1D").sum().resample("1D").last()
    notional_turnover = notional_qty * mid_1d_last
    approx_trades = (notional_turnover / max(1e-9, usd_per_order)).dropna()
    return float(approx_trades.mean()) if len(approx_trades) else 0.0

# --------------------------- command construction ---------------------------

def _build_cmd(
    binary: str,
    name: str,
    out_path: str,
    data_files: List[str],
    latency_files: List[str],
    tick_size: float,
    lot_size: float,
    maker_fee: float,
    taker_fee: float,
    queue_power: float,
    grid: Dict[str, Any],
    time_ctrl: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    xform: Dict[str, Any],
    initial_snapshot: Optional[str],
) -> List[str]:
    if not latency_files:
        raise ValueError(
            "HftBacktest grid search requires --latency-files; generate user-supplied latency "
            "data before running this research path"
        )
    algo_cfg = _normalize_algorithm(algo_cfg)
    xform = _normalize_transform(xform, algo_cfg["name"])
    args: List[str] = [
        binary,
        "--name", name,
        "--output-path", out_path,
        "--tick-size", str(tick_size),
        "--lot-size", str(lot_size),
        "--maker-fee", str(maker_fee),
        "--taker-fee", str(taker_fee),
        "--queue-power", str(queue_power),
        "--relative-half-spread", str(grid["relative_half_spread"]),
        "--relative-grid-interval", str(grid["relative_grid_interval"]),
        "--grid-num", str(grid["grid_num"]),
        "--order-qty", str(grid["order_qty"]),
        "--max-position", str(grid["max_position"]),
        "--skew", str(grid["skew"]),
        "--elapse-ns", str(time_ctrl.get("elapse_ns", 1_000_000_000)),
        "--record-every", str(time_ctrl.get("record_every", 1)),
        "--algo", algo_cfg["name"],
        "--transform", xform["kind"],
    ]
    mgs = grid.get("min_grid_step_override")
    if mgs is not None:
        args += ["--min-grid-step", str(mgs)]
    if initial_snapshot:
        args += ["--initial-snapshot", initial_snapshot]

    # transform extras
    kind = xform["kind"].lower()
    if kind in ("sma", "zscore"):
        args += ["--window", str(xform.get("window", 300))]
    if kind == "ema":
        args += ["--ema-alpha", str(xform.get("ema_alpha", 0.1))]

    # algo extras
    name_algo = algo_cfg["name"].lower()
    p = (algo_cfg.get("params") or {})
    if name_algo == "baseline":
        pass
    elif name_algo == "obi-static-alpha":
        args += [
            "--look-depth-pct", str(p.get("look_depth_pct", 0.02)),
            "--alpha-scale",    str(p.get("alpha_scale", 50.0)),
        ]
        if p.get("normalize", True):
            args += ["--normalize"]
    elif name_algo in ("vamp", "vamp-effective"):
        args += [
            "--vamp-depth-pct", str(p["vamp_depth_pct"]),
            "--alpha-scale", str(p["alpha_scale"]),
        ]
    elif name_algo == "weighted-depth":
        args += [
            "--target-qty-per-side", str(p["target_qty_per_side"]),
            "--alpha-scale", str(p["alpha_scale"]),
        ]

    # variable-length files last
    args += ["--data-files", *data_files]
    args += ["--latency-files", *latency_files]
    return args

# --------------------------- mini ParameterGrid ---------------------------

def _param_grid_iter(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    if not keys:
        yield {}
        return
    def rec(i: int, cur: Dict[str, Any]):
        if i == len(keys):
            yield dict(cur); return
        k = keys[i]
        vals = grid[k]
        if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
            vals = [vals]
        for v in vals:
            cur[k] = v
            yield from rec(i + 1, cur)
    yield from rec(0, {})

# --------------------------- grid baseline per symbol ---------------------------

def _symbol_base_grid(symbol: str, tickers: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    dflt = cfg["defaults"]
    info = tickers.get(symbol, {})
    tick_size = float(info.get("tick_size", dflt["tick_size"]))
    lot_size  = float(info.get("lot_size",  dflt["lot_size"]))
    wap = float(info.get("weighted_avg_price", 100.0))
    min_qty = float(info.get("min_qty", lot_size))

    g = cfg["grid"].copy()
    px = wap
    order_qty100 = round((g["order_value_usd"] / px) / lot_size) * lot_size
    g["order_qty"] = max(min_qty, order_qty100)
    g["max_position"] = g["max_position_in_grids"] * g["order_qty"]
    return dict(tick_size=tick_size, lot_size=lot_size, grid=g)

# --------------------------- transform variants ---------------------------

def _transform_variants(cfg_transform: Dict[str, Any], gs_transform: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base = dict(cfg_transform)
    gs_transform = gs_transform or {}

    def _as_list(v, default):
        if v is None:
            return [default]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    kinds = _as_list(gs_transform.get("kind", base.get("kind", "none")), base.get("kind", "none"))
    win_list = _as_list(gs_transform.get("window", base.get("window")), base.get("window", 300))
    ema_list = _as_list(gs_transform.get("ema_alpha", base.get("ema_alpha")), base.get("ema_alpha", 0.1))

    out: List[Dict[str, Any]] = []
    for k in kinds:
        k_low = str(k).lower()
        if k_low in ("sma", "zscore"):
            for w in win_list:
                out.append(dict(kind=k, window=int(w)))
        elif k_low == "ema":
            for a in ema_list:
                out.append(dict(kind=k, ema_alpha=float(a)))
        else:
            out.append(dict(kind=k))  # "none"
    return out or [base]

# --------------------------- run spec ---------------------------

@dataclass
class RunSpec:
    name: str
    candidate_id: str
    candidate_manifest_json: str
    candidate_manifest_sha256: str
    partition_manifest_json: str
    partition_manifest_sha256: str
    input_manifest_json: str
    input_manifest_sha256: str
    phase: str
    symbol: str
    data_files: List[str]
    latency_files: List[str]
    tick_size: float
    lot_size: float
    maker_fee: float
    taker_fee: float
    queue_power: float
    grid: Dict[str, Any]
    time_ctrl: Dict[str, Any]
    algo_cfg: Dict[str, Any]
    xform: Dict[str, Any]
    initial_snapshot: Optional[str]
    rust_log: Optional[str]
    rust_backtrace: Optional[int]
    out_path: str
    binary: str

# --------------------------- build runs ---------------------------

def _as_list(v) -> List[Any]:
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]

def _required_partition_files(
    files: List[Dict[str, str]], phase: str, symbol: str
) -> Tuple[List[str], List[str]]:
    missing_data = [item["date"] for item in files if not os.path.isfile(item["data"])]
    missing_latency = [item["date"] for item in files if not os.path.isfile(item["lat"])]
    if missing_data or missing_latency:
        details = []
        if missing_data:
            details.append(f"data dates {missing_data}")
        if missing_latency:
            details.append(f"latency dates {missing_latency}")
        raise ValueError(
            f"{phase} inputs for {symbol} are incomplete: {', '.join(details)}; "
            "provide one data and one latency file for every configured date"
        )
    return [item["data"] for item in files], [item["lat"] for item in files]


def _validation_record(out_path: str, locked_candidate: str) -> Dict[str, str]:
    summary_path = os.path.join(out_path, "gridsearch_validation_summary.csv")
    if not os.path.exists(summary_path):
        raise ValueError(
            "held-out test mode requires gridsearch_validation_summary.csv from a completed "
            "explore run"
        )
    with open(summary_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "name",
            "candidate_id",
            "phase",
            "status",
            "candidate_manifest_json",
            "candidate_manifest_sha256",
            "partition_manifest_json",
            "partition_manifest_sha256",
            "input_manifest_json",
            "input_manifest_sha256",
            "artifact_manifest_sha256",
            "csv",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "validation summary is missing required lock, manifest, or artifact columns"
            )
        matches = [row for row in reader if row["candidate_id"] == locked_candidate]
    if len(matches) != 1:
        raise ValueError(
            "gridsearch.locked_candidate must have exactly one unique row in the validation summary"
        )
    record = matches[0]
    if record["phase"] != "validation" or record["status"] != "ok":
        raise ValueError("gridsearch.locked_candidate must reference one successful validation row")
    return record


def _validate_manifest_cell(
    record: Dict[str, str], prefix: str, expected_json: str, expected_sha256: str
) -> None:
    encoded = record[f"{prefix}_json"]
    recorded_hash = record[f"{prefix}_sha256"]
    try:
        canonical = _canonical_json(json.loads(encoded))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"validation summary contains invalid {prefix} JSON") from error
    if canonical != encoded or _sha256_text(encoded) != recorded_hash:
        raise ValueError(f"validation summary contains a corrupt {prefix} fingerprint")
    if encoded != expected_json or recorded_hash != expected_sha256:
        raise ValueError(f"validation summary {prefix} does not match the current configuration")


def _validate_locked_validation(out_path: str, expected: RunSpec) -> None:
    record = _validation_record(out_path, expected.candidate_id)
    if record["name"] != expected.name:
        raise ValueError("validation summary run name does not match the locked candidate")
    _validate_manifest_cell(
        record,
        "candidate_manifest",
        expected.candidate_manifest_json,
        expected.candidate_manifest_sha256,
    )
    _validate_manifest_cell(
        record,
        "partition_manifest",
        expected.partition_manifest_json,
        expected.partition_manifest_sha256,
    )
    _validate_manifest_cell(
        record,
        "input_manifest",
        expected.input_manifest_json,
        expected.input_manifest_sha256,
    )
    _, artifact_manifest_sha256 = _validate_existing_artifacts(expected)
    if record["artifact_manifest_sha256"] != artifact_manifest_sha256:
        raise ValueError("validation summary artifact fingerprint does not match the current files")
    if os.path.abspath(record["csv"]) != os.path.abspath(_expected_result_csv(expected)):
        raise ValueError("validation summary CSV path does not match the locked validation artifact")


def _build_runs(cfg: Dict[str, Any], phase_mode: str = "explore") -> List[RunSpec]:
    if phase_mode not in ("explore", "test"):
        raise ValueError("phase_mode must be 'explore' or 'test'")
    base = cfg["base_root"]
    exch = cfg["exchange"]
    requested_phases = (
        {"train", "validation"} if phase_mode == "explore" else {"validation", "test"}
    )
    partitions = [
        partition
        for partition in _date_partitions(cfg)
        if partition[0] in requested_phases
    ]

    with open(cfg["tickers_json"], "r", encoding="utf-8") as f:
        tickers = json.load(f)

    gs = cfg.get("gridsearch", {}) or {}
    symbols: List[str] = gs.get("symbols") or cfg["symbols"]
    locked_candidate: Optional[str] = None
    if phase_mode == "test":
        locked = gs.get("locked_candidate")
        if (
            not isinstance(locked, (list, tuple))
            or len(locked) != 1
            or not isinstance(locked[0], str)
        ):
            raise ValueError(
                "gridsearch.locked_candidate must contain exactly one validation candidate ID; "
                "select it after reviewing the validation summary (TODO(Dennis))"
            )
        locked_candidate = locked[0]
        if locked_candidate == "TODO(Dennis)":
            raise ValueError(
                "replace gridsearch.locked_candidate after reviewing the validation summary "
                "(TODO(Dennis))"
            )

    # Grid sweeps
    rel_half_list: Sequence[float] = gs.get("rel_half_spread", [cfg["grid"]["relative_half_spread"]])
    grid_num_list: Sequence[int]   = gs.get("grid_num", [cfg["grid"]["grid_num"]])

    # "same" means rgi == rhs for each trial; do NOT create a cross-product
    rgi_vals = gs.get("relative_grid_interval", "same")
    rgi_same = (isinstance(rgi_vals, str) and rgi_vals.lower() == "same")

    skew_mode = gs.get("skew_mode", "rel_over_n")  # "rel_over_n" | "fixed"
    mp_mode = gs.get("max_position_mode", "as_in_config")  # "as_in_config" | "equal_to_grid_num"

    # Optional algo param sweeps (e.g., alpha_scale list)
    algo_overrides = gs.get("algo_params", {}) or {}

    # Transform sweeps
    xform_variants = _transform_variants(cfg["transform"], gs.get("transform"))

    # Build the cartesian only for independent dims
    param_grid = {
        "rel_half_spread": rel_half_list,
        "grid_num": grid_num_list,
    }
    if not rgi_same:
        if isinstance(rgi_vals, (list, tuple)):
            param_grid["relative_grid_interval"] = list(rgi_vals)
        else:
            param_grid["relative_grid_interval"] = [rgi_vals]

    # Allow skew as a *list* when skew_mode == "fixed"
    if skew_mode == "fixed":
        skew_vals = gs.get("skew_fixed_value", [cfg["grid"].get("skew_override", 0.0)])
        if not isinstance(skew_vals, (list, tuple)):
            skew_vals = [skew_vals]
        param_grid["skew_fixed_value"] = list(skew_vals)

    # Algo overrides become independent axes (e.g., glft_vol_scale sweep)
    for k, vals in algo_overrides.items():
        param_grid[f"algo::{k}"] = vals if isinstance(vals, (list, tuple)) else [vals]

    time_ctrl = dict(
        elapse_ns    = cfg.get("elapse_ns", 1_000_000_000),
        record_every = cfg.get("record_every", 1),
    )
    engine_identity = {
        "adapter": "gridtrading_backtest_args",
        "binary": _binary_identity(cfg["binary"]),
        **_engine_source_identity(),
    }

    runs: List[RunSpec] = []
    for phase, dates in partitions:
        for sym in symbols:
            base_params = _symbol_base_grid(sym, tickers, cfg)
            files = [_files_for(base, exch, sym, d) for d in dates]
            data_files, latency_files = _required_partition_files(files, phase, sym)

            init_snap = _resolve_initial_snapshot(cfg, sym, dates[0])
            partition_manifest_json, partition_manifest_sha256 = _manifest(
                _partition_manifest(cfg, sym)
            )
            input_manifest_json, input_manifest_sha256 = _manifest(
                _input_manifest(phase, sym, data_files, latency_files, init_snap)
            )

            for combo in _param_grid_iter(param_grid):
                g = dict(base_params["grid"])
                rhs = float(combo["rel_half_spread"])
                n   = int(combo["grid_num"])
                rgi = rhs if rgi_same else float(combo.get("relative_grid_interval", rhs))

                g["relative_half_spread"]   = rhs
                g["relative_grid_interval"] = rgi
                g["grid_num"]               = n

                if skew_mode == "rel_over_n":
                    g["skew"] = (rhs / n) if n != 0 else 0.0
                else:
                    g["skew"] = float(combo["skew_fixed_value"])

                if mp_mode == "equal_to_grid_num":
                    g["max_position"] = g["order_qty"] * n

                algo_cfg = dict(cfg["algo"])
                algo_cfg["params"] = dict(algo_cfg.get("params") or {})
                for k, v in combo.items():
                    if k.startswith("algo::"):
                        algo_cfg["params"][k.split("::", 1)[1]] = v
                algo_cfg = _normalize_algorithm(algo_cfg)

                for xf in xform_variants:
                    xform = _normalize_transform(dict(xf), algo_cfg["name"])
                    candidate_manifest_json, candidate_manifest_sha256 = _manifest(
                        _candidate_manifest(
                            cfg,
                            sym,
                            g,
                            algo_cfg,
                            xform,
                            base_params["tick_size"],
                            base_params["lot_size"],
                            time_ctrl,
                            engine_identity,
                        )
                    )
                    candidate_id = _candidate_id(sym, candidate_manifest_sha256)
                    name = (
                        f"{phase}__{_name_for_run(sym, rhs, rgi, n, g['skew'], xform, algo_cfg)}"
                        f"__cid{candidate_manifest_sha256[:16]}"
                    )
                    runs.append(
                        RunSpec(
                            name=name,
                            candidate_id=candidate_id,
                            candidate_manifest_json=candidate_manifest_json,
                            candidate_manifest_sha256=candidate_manifest_sha256,
                            partition_manifest_json=partition_manifest_json,
                            partition_manifest_sha256=partition_manifest_sha256,
                            input_manifest_json=input_manifest_json,
                            input_manifest_sha256=input_manifest_sha256,
                            phase=phase,
                            symbol=sym,
                            data_files=data_files,
                            latency_files=latency_files,
                            tick_size=base_params["tick_size"],
                            lot_size=base_params["lot_size"],
                            maker_fee=cfg["fees"]["maker"],
                            taker_fee=cfg["fees"]["taker"],
                            queue_power=cfg.get("queue_power", 3.0),
                            grid=g,
                            time_ctrl=time_ctrl,
                            algo_cfg=algo_cfg,
                            xform=xform,
                            initial_snapshot=init_snap,
                            rust_log=cfg.get("rust_log"),
                            rust_backtrace=cfg.get("rust_backtrace"),
                            out_path=cfg["out_path"],
                            binary=cfg["binary"],
                        )
                    )
    identities = [(run.phase, run.candidate_id) for run in runs]
    names = [run.name for run in runs]
    if len(identities) != len(set(identities)) or len(names) != len(set(names)):
        raise ValueError(
            "grid search produced duplicate candidate identities or output names; "
            "remove duplicate sweep values"
        )
    if phase_mode == "explore":
        return runs

    assert locked_candidate is not None
    validation_runs = [
        run for run in runs if run.phase == "validation" and run.candidate_id == locked_candidate
    ]
    test_runs = [run for run in runs if run.phase == "test" and run.candidate_id == locked_candidate]
    if len(validation_runs) != 1:
        raise ValueError(
            "gridsearch.locked_candidate does not match exactly one candidate under the current "
            "execution assumptions"
        )
    if len(test_runs) != 1:
        raise ValueError("gridsearch.locked_candidate must produce exactly one held-out test run")
    _validate_locked_validation(cfg["out_path"], validation_runs[0])
    return test_runs

# --------------------------- worker ---------------------------


def _command_for_spec(spec: RunSpec) -> List[str]:
    return _build_cmd(
        binary=spec.binary,
        name=spec.name,
        out_path=spec.out_path,
        data_files=spec.data_files,
        latency_files=spec.latency_files,
        tick_size=spec.tick_size,
        lot_size=spec.lot_size,
        maker_fee=spec.maker_fee,
        taker_fee=spec.taker_fee,
        queue_power=spec.queue_power,
        grid=spec.grid,
        time_ctrl=spec.time_ctrl,
        algo_cfg=spec.algo_cfg,
        xform=spec.xform,
        initial_snapshot=spec.initial_snapshot,
    )


def _expected_result_csv(spec: RunSpec) -> str:
    return os.path.join(spec.out_path, f"{spec.name}0.csv")


def _rust_manifest_path(spec: RunSpec) -> str:
    return os.path.join(spec.out_path, f"{spec.name}_run_manifest.json")


def _artifact_manifest_path(spec: RunSpec) -> str:
    return os.path.join(spec.out_path, f"{spec.name}_gridsearch_manifest.json")


def _expected_rust_manifest(spec: RunSpec) -> Dict[str, Any]:
    name = spec.algo_cfg["name"]
    params = spec.algo_cfg["params"]
    if name == "baseline":
        executed_parameters: Dict[str, Any] = {}
    elif name == "obi-static-alpha":
        executed_parameters = {
            "look_depth_pct": params["look_depth_pct"],
            "normalize": params["normalize"],
            "alpha_scale": params["alpha_scale"],
        }
    elif name in ("vamp", "vamp-effective"):
        executed_parameters = {
            "vamp_depth_pct": params["vamp_depth_pct"],
            "alpha_scale": params["alpha_scale"],
        }
    elif name == "weighted-depth":
        executed_parameters = {
            "target_qty_per_side": params["target_qty_per_side"],
            "alpha_scale": params["alpha_scale"],
        }
    else:
        raise ValueError(f"Unsupported algo: {name}")

    kind = spec.xform["kind"]
    if kind == "none":
        transform_parameters: Dict[str, Any] = {}
    elif kind in ("sma", "zscore"):
        transform_parameters = {"window": spec.xform["window"]}
    elif kind == "ema":
        transform_parameters = {"ema_alpha": spec.xform["ema_alpha"]}
    else:
        raise ValueError(f"Unsupported transform: {kind}")
    return {
        "algorithm": name,
        "transform": kind,
        "executed_strategy": name,
        "executed_parameters": executed_parameters,
        "transform_parameters": transform_parameters,
        "elapse_ns": int(spec.time_ctrl.get("elapse_ns", 1_000_000_000)),
        "record_every": int(spec.time_ctrl.get("record_every", 1)),
    }


def _load_rust_manifest(spec: RunSpec) -> Dict[str, Any]:
    path = _rust_manifest_path(spec)
    if not os.path.isfile(path):
        raise ValueError(f"missing Rust execution manifest: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Rust execution manifest: {path}") from error
    if manifest != _expected_rust_manifest(spec):
        raise ValueError(f"Rust execution manifest does not match run specification: {path}")
    return manifest


def _artifact_manifest_payload(spec: RunSpec) -> Dict[str, Any]:
    result_csv = _expected_result_csv(spec)
    matching_csvs = sorted(glob.glob(os.path.join(spec.out_path, f"{spec.name}*.csv")))
    if matching_csvs != [result_csv] or not os.path.isfile(result_csv):
        raise ValueError(
            f"expected exactly one HftBacktest result CSV at {result_csv}; found {matching_csvs}"
        )
    rust_manifest = _load_rust_manifest(spec)
    return {
        "schema_version": 1,
        "name": spec.name,
        "candidate_id": spec.candidate_id,
        "phase": spec.phase,
        "candidate_manifest_json": spec.candidate_manifest_json,
        "candidate_manifest_sha256": spec.candidate_manifest_sha256,
        "partition_manifest_json": spec.partition_manifest_json,
        "partition_manifest_sha256": spec.partition_manifest_sha256,
        "input_manifest_json": spec.input_manifest_json,
        "input_manifest_sha256": spec.input_manifest_sha256,
        "command": _command_for_spec(spec),
        "result_csv": _file_identity(result_csv),
        "rust_manifest": rust_manifest,
        "rust_manifest_file": _file_identity(_rust_manifest_path(spec)),
    }


def _write_artifact_manifest(spec: RunSpec) -> None:
    payload = _artifact_manifest_payload(spec)
    path = _artifact_manifest_path(spec)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(payload))
        handle.write("\n")
    os.replace(temporary, path)


def _validate_existing_artifacts(spec: RunSpec) -> Tuple[Dict[str, Any], str]:
    path = _artifact_manifest_path(spec)
    if not os.path.isfile(path):
        raise ValueError(f"missing grid-search artifact manifest: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid grid-search artifact manifest: {path}") from error
    expected = _artifact_manifest_payload(spec)
    if actual != expected:
        raise ValueError(f"grid-search artifact manifest does not match current run: {path}")
    return actual, _sha256_file(path)


def _run_one(spec: RunSpec) -> Tuple[RunSpec, int]:
    cmd = _command_for_spec(spec)
    env = os.environ.copy()
    if spec.rust_log:
        env["RUST_LOG"] = spec.rust_log
    if spec.rust_backtrace is not None:
        env["RUST_BACKTRACE"] = str(spec.rust_backtrace)
    print(f"[run] {spec.name}  RUST_LOG={env.get('RUST_LOG')}  data={len(spec.data_files)}  lat={len(spec.latency_files)}")
    sidecar = _artifact_manifest_path(spec)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except OSError as error:
        print(f"[run] failed to launch {spec.name}: {error}")
        return spec, 1
    if rc == 0:
        try:
            _write_artifact_manifest(spec)
        except (OSError, ValueError) as error:
            print(f"[run] invalid artifacts for {spec.name}: {error}")
            rc = 1
    return spec, rc

# --------------------------- summary / plots ---------------------------

def _summarize(
    out_dir: str,
    specs: List[RunSpec],
    make_plots: bool,
    usd_per_order: float,
    return_codes: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    global mdates, np, pd, plt
    import numpy as np
    import pandas as pd
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "gridsearch_plots")
    if make_plots:
        os.makedirs(plot_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for s in specs:
        return_code = (return_codes or {}).get(s.name, 0)
        artifact_error: Optional[str] = None
        artifact_manifest_sha256: Optional[str] = None
        csv_path: Optional[str] = None
        if return_code == 0:
            try:
                _, artifact_manifest_sha256 = _validate_existing_artifacts(s)
                csv_path = _expected_result_csv(s)
            except ValueError as error:
                artifact_error = str(error)
        if return_code != 0 or artifact_error is not None:
            rows.append(dict(
                name=s.name, candidate_id=s.candidate_id, phase=s.phase, symbol=s.symbol,
                status="failed" if return_code != 0 else "invalid_artifact",
                ret_abs=np.nan, ret_pct=np.nan, start_eq=np.nan, end_eq=np.nan,
                rel_half_spread=s.grid["relative_half_spread"],
                relative_grid_interval=s.grid["relative_grid_interval"],
                grid_num=s.grid["grid_num"],
                skew=s.grid["skew"],
                order_qty=s.grid["order_qty"],
                max_position=s.grid["max_position"],
                transform_kind=s.xform["kind"],
                transform_window=s.xform.get("window"),
                transform_ema_alpha=s.xform.get("ema_alpha"),
                algo=_algo_tag(s.algo_cfg),
                candidate_manifest_json=s.candidate_manifest_json,
                candidate_manifest_sha256=s.candidate_manifest_sha256,
                partition_manifest_json=s.partition_manifest_json,
                partition_manifest_sha256=s.partition_manifest_sha256,
                input_manifest_json=s.input_manifest_json,
                input_manifest_sha256=s.input_manifest_sha256,
                artifact_manifest_sha256=artifact_manifest_sha256,
                artifact_error=artifact_error,
                csv=None, plot=None,
            ))
            continue

        assert csv_path is not None
        df = _read_result_csv(csv_path)

        # Safe components (fill/carry)
        idx = df.index
        b   = (df["balance"] if "balance" in df else pd.Series(0.0, index=idx)).astype(float).ffill().fillna(0.0)
        p   = df["price"].astype(float).ffill()
        pos = (df["position"] if "position" in df else pd.Series(0.0, index=idx)).astype(float).fillna(0.0)
        fee = (df["fee"] if "fee" in df else pd.Series(0.0, index=idx)).astype(float).fillna(0.0)

        equity = (b + pos * p - fee).dropna()
        if len(equity):
            start_eq = float(equity.iloc[0])
            end_eq   = float(equity.iloc[-1])
            ret_abs  = end_eq - start_eq
            ret_pct  = (ret_abs / start_eq * 100.0) if start_eq != 0 else np.nan
        else:
            start_eq = end_eq = ret_abs = ret_pct = np.nan

        # Approximate daily trades (robust)
        try:
            mid_1d_last = p.resample("1D").last()
            notional_qty = pos.diff().abs().rolling("1D").sum().resample("1D").last()
            notional_turnover = notional_qty * mid_1d_last
            approx_trades = (notional_turnover / max(1e-9, usd_per_order)).dropna()
            avg_daily_trades = float(approx_trades.mean()) if len(approx_trades) else 0.0
        except Exception:
            avg_daily_trades = float("nan")

        plot_path = None
        if make_plots and len(equity):
            eq5 = equity.resample("5min").last()
            pos5 = df["position"].resample("5min").last()
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.set_title(f"{s.name} | avg daily trades ~ {avg_daily_trades:.0f}")
            ax.set_ylabel("Equity $")
            ax.plot(eq5.index, eq5.values)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax2 = ax.twinx()
            ax2.set_ylabel("Position")
            ax2.plot(pos5.index, pos5.values, alpha=0.4)
            fig.tight_layout()
            plot_path = os.path.join(plot_dir, f"{s.name}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        rows.append(dict(
            name=s.name, candidate_id=s.candidate_id, phase=s.phase, symbol=s.symbol, status="ok",
            ret_abs=ret_abs, ret_pct=ret_pct, start_eq=start_eq, end_eq=end_eq,
            rel_half_spread=s.grid["relative_half_spread"],
            relative_grid_interval=s.grid["relative_grid_interval"],
            grid_num=s.grid["grid_num"],
            skew=s.grid["skew"],
            order_qty=s.grid["order_qty"],
            max_position=s.grid["max_position"],
            approx_avg_daily_trades=avg_daily_trades,
            transform_kind=s.xform["kind"],
            transform_window=s.xform.get("window"),
            transform_ema_alpha=s.xform.get("ema_alpha"),
            algo=_algo_tag(s.algo_cfg),
            candidate_manifest_json=s.candidate_manifest_json,
            candidate_manifest_sha256=s.candidate_manifest_sha256,
            partition_manifest_json=s.partition_manifest_json,
            partition_manifest_sha256=s.partition_manifest_sha256,
            input_manifest_json=s.input_manifest_json,
            input_manifest_sha256=s.input_manifest_sha256,
            artifact_manifest_sha256=artifact_manifest_sha256,
            artifact_error=None,
            csv=csv_path, plot=plot_path,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("[summary] no runs to summarize")
        return df
    for phase in ("train", "validation", "test"):
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue
        if phase == "test":
            phase_df = phase_df.sort_values(["symbol", "candidate_id"])
        else:
            phase_df = phase_df.sort_values(["symbol", "ret_abs"], ascending=[True, False])
        out_csv = os.path.join(out_dir, f"gridsearch_{phase}_summary.csv")
        phase_df.to_csv(out_csv, index=False)
        print(f"[summary] wrote {out_csv}")
    return df

# --------------------------- CLI ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="pipeline/gridsearch_config.yaml",
                    help="Path to YAML backtest config with 'gridsearch' section.")
    ap.add_argument("--processes", type=int, default=None,
                    help="Override parallelism; default uses cfg.num_proc or 4.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a run if an output CSV already exists.")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    ap.add_argument(
        "--phase",
        choices=("explore", "test"),
        default="explore",
        help=(
            "explore runs train and validation only; test requires one successful candidate "
            "locked from the persisted validation summary"
        ),
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["out_path"], exist_ok=True)
    runs = _build_runs(cfg, phase_mode=args.phase)
    if not runs:
        print("[gridsearch] nothing to run.")
        return 0

    skip_existing = args.skip_existing or bool((cfg.get("gridsearch") or {}).get("skip_existing", False))
    all_runs = list(runs)
    existing_results: List[Tuple[RunSpec, int]] = []
    if skip_existing:
        pending = []
        for s in runs:
            related = [
                *glob.glob(os.path.join(s.out_path, f"{s.name}*.csv")),
                _rust_manifest_path(s),
                _artifact_manifest_path(s),
            ]
            if any(os.path.exists(path) for path in related):
                try:
                    _validate_existing_artifacts(s)
                except ValueError as error:
                    raise ValueError(
                        f"--skip-existing refused stale or mismatched artifacts for {s.name}: {error}"
                    ) from error
                print(f"[skip] {s.name} (verified manifest and exact result CSV)")
                existing_results.append((s, 0))
            else:
                pending.append(s)
        runs = pending

    nproc = args.processes or int(cfg.get("num_proc", 4))
    print(f"[gridsearch] launching {len(runs)} runs with processes={nproc}")

    if runs:
        with Pool(processes=nproc) as pool:
            new_results = pool.map(_run_one, runs)
    else:
        new_results = []
    results = existing_results + new_results

    bad = sum(1 for _, rc in results if rc != 0)
    print(f"[gridsearch] Done. {len(results)-bad} OK / {bad} FAIL")

    usd_per_order = float(cfg["grid"].get("order_value_usd", 100.0))
    summary = _summarize(
        cfg["out_path"],
        all_runs,
        make_plots=(not args.no_plots),
        usd_per_order=usd_per_order,
        return_codes={spec.name: return_code for spec, return_code in results},
    )
    invalid_artifacts = int((summary["status"] != "ok").sum()) if not summary.empty else 0
    return 1 if bad or invalid_artifacts else 0

if __name__ == "__main__":
    raise SystemExit(main())
