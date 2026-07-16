from __future__ import annotations
import argparse
import glob
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

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
    return {"data": data, "lat": lat}

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
        return f"vamp-dp{dp}"
    if nm == "vamp-effective":
        dp = f(p.get("vamp_depth_pct", 0.02))
        return f"vampe-dp{dp}"
    if nm == "weighted-depth":
        t = f(p.get("target_qty_per_side", 500.0))
        return f"wdepth-t{t}"
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

def _resolve_initial_snapshot(cfg: dict, symbol: str, first_date_yyyymmdd: str) -> Optional[str]:
    ini = cfg.get("initial_snapshot")
    if not ini:
        return None

    def _fmt(s: str) -> str:
        return s.format(
            base_root=cfg["base_root"],
            exchange=cfg["exchange"],
            symbol=symbol,
            date=first_date_yyyymmdd,
        )

    if isinstance(ini, str):
        path = _fmt(ini)
        return path if os.path.exists(path) else None

    if isinstance(ini, dict):
        val = ini.get(symbol) or ini.get("*")
        if not val:
            return None
        path = _fmt(val)
        return path if os.path.exists(path) else None

    return None

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

def _first_result_csv(out_path: str, name: str) -> Optional[str]:
    patt = os.path.join(out_path, f"{name}*.csv")
    matches = sorted(glob.glob(patt))
    return matches[0] if matches else None

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
    if name_algo == "obi-static-alpha":
        args += [
            "--look-depth-pct", str(p.get("look_depth_pct", 0.02)),
            "--alpha-scale",    str(p.get("alpha_scale", 50.0)),
        ]
        if p.get("normalize", True):
            args += ["--normalize"]
    elif name_algo in ("vamp", "vamp-effective"):
        args += ["--vamp-depth-pct", str(p.get("vamp_depth_pct", 0.02))]
    elif name_algo == "weighted-depth":
        args += ["--target-qty-per-side", str(p.get("target_qty_per_side", 500.0))]
    elif name_algo == "glft-simple":
        args += [
            "--glft-vol-window", str(int(p.get("glft_vol_window", 6000))),
            "--glft-vol-scale",  str(p.get("glft_vol_scale", 0.5)),
        ]
        # Do NOT pass any other GLFT flags if your binary doesn't expose them.
    else:
        raise ValueError(f"Unsupported algo: {name_algo}")

    # variable-length files last
    args += ["--data-files", *data_files]
    if latency_files:
        args += ["--latency-files", *latency_files]
    else:
        args += ["--latency-files"]
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
    if not gs_transform:
        return [base]

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

def _build_runs(cfg: Dict[str, Any]) -> List[RunSpec]:
    base = cfg["base_root"]
    exch = cfg["exchange"]
    partitions = _date_partitions(cfg)

    with open(cfg["tickers_json"], "r", encoding="utf-8") as f:
        tickers = json.load(f)

    gs = cfg.get("gridsearch", {}) or {}
    symbols: List[str] = gs.get("symbols") or cfg["symbols"]

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

    runs: List[RunSpec] = []
    for phase, dates in partitions:
        for sym in symbols:
            base_params = _symbol_base_grid(sym, tickers, cfg)
            # files
            files = [_files_for(base, exch, sym, d) for d in dates]
            data_files = [f["data"] for f in files if os.path.exists(f["data"])]
            latency_files = [f["lat"] for f in files if os.path.exists(f["lat"])]
            if not data_files:
                print(f"[gridsearch] WARN: no {phase} data files for {sym}; skipping.")
                continue

            # resolve per-symbol initial snapshot (if configured/templates provided)
            init_snap = _resolve_initial_snapshot(cfg, sym, dates[0])

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
                if "params" not in algo_cfg or algo_cfg["params"] is None:
                    algo_cfg["params"] = {}
                for k, v in combo.items():
                    if k.startswith("algo::"):
                        algo_cfg["params"][k.split("::", 1)[1]] = v

                for xf in xform_variants:
                    xform = dict(xf)
                    name = f"{phase}__{_name_for_run(sym, rhs, rgi, n, g['skew'], xform, algo_cfg)}"
                    runs.append(
                        RunSpec(
                            name=name,
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
    return runs

# --------------------------- worker ---------------------------

def _run_one(spec: RunSpec) -> Tuple[RunSpec, int]:
    cmd = _build_cmd(
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
    env = os.environ.copy()
    if spec.rust_log:
        env["RUST_LOG"] = spec.rust_log
    if spec.rust_backtrace is not None:
        env["RUST_BACKTRACE"] = str(spec.rust_backtrace)
    print(f"[run] {spec.name}  RUST_LOG={env.get('RUST_LOG')}  data={len(spec.data_files)}  lat={len(spec.latency_files)}")
    rc = subprocess.run(cmd, env=env).returncode
    return spec, rc

# --------------------------- summary / plots ---------------------------

def _summarize(out_dir: str, specs: List[RunSpec], make_plots: bool, usd_per_order: float) -> pd.DataFrame:
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
        csv_path = _first_result_csv(out_dir, s.name)
        if not csv_path or not os.path.exists(csv_path):
            rows.append(dict(
                name=s.name, symbol=s.symbol, status="missing_csv",
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
                csv=None, plot=None,
            ))
            continue

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
            name=s.name, symbol=s.symbol, status="ok",
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
            csv=csv_path, plot=plot_path,
        ))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "gridsearch_summary.csv")
    df.sort_values(["symbol", "ret_abs"], ascending=[True, False]).to_csv(out_csv, index=False)
    print(f"[summary] wrote {out_csv}")
    return df

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="pipeline/backtest_config.yaml",
                    help="Path to YAML backtest config with 'gridsearch' section.")
    ap.add_argument("--processes", type=int, default=None,
                    help="Override parallelism; default uses cfg.num_proc or 4.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a run if an output CSV already exists.")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["out_path"], exist_ok=True)
    runs = _build_runs(cfg)
    if not runs:
        print("[gridsearch] nothing to run.")
        return

    skip_existing = args.skip_existing or bool((cfg.get("gridsearch") or {}).get("skip_existing", False))
    if skip_existing:
        kept = []
        for s in runs:
            csv = _first_result_csv(cfg["out_path"], s.name)
            if csv and os.path.exists(csv):
                print(f"[skip] {s.name} (found {csv})")
            else:
                kept.append(s)
        runs = kept

    nproc = args.processes or int(cfg.get("num_proc", 4))
    print(f"[gridsearch] launching {len(runs)} runs with processes={nproc}")

    with Pool(processes=nproc) as pool:
        results = pool.map(_run_one, runs)

    bad = sum(1 for _, rc in results if rc != 0)
    print(f"[gridsearch] Done. {len(results)-bad} OK / {bad} FAIL")

    usd_per_order = float(cfg["grid"].get("order_value_usd", 100.0))
    _summarize(cfg["out_path"], [s for s, _ in results], make_plots=(not args.no_plots), usd_per_order=usd_per_order)

if __name__ == "__main__":
    main()
