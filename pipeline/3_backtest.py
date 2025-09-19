from __future__ import annotations
import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import Dict, List, Any

import yaml

def _date_range(d0:int, d1:int) -> List[str]:
    s = datetime.strptime(str(d0), "%Y%m%d")
    e = datetime.strptime(str(d1), "%Y%m%d")
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out

def _files_for(base_root:str, exchange:str, symbol:str, yyyymmdd:str) -> Dict[str, str]:
    data = os.path.join(base_root, "data", exchange, symbol, f"{symbol}_{yyyymmdd}.npz")
    lat  = os.path.join(base_root, "latency", exchange, symbol, f"latency_{yyyymmdd}.npz")
    return {"data": data, "lat": lat}

def _build_cmd(
    binary:str,
    name:str,
    out_path:str,
    data_files:List[str],
    latency_files:List[str],
    tick_size:float,
    lot_size:float,
    maker_fee:float,
    taker_fee:float,
    queue_power:float,
    grid:Dict[str, Any],
    time_ctrl:Dict[str, Any],
    algo_cfg:Dict[str, Any],
    xform:Dict[str, Any],
    initial_snapshot:str|None,
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
    if xform["kind"].lower() in ("sma", "zscore"):
        args += ["--window", str(xform.get("window", 300))]
    if xform["kind"].lower() == "ema":
        args += ["--ema-alpha", str(xform.get("ema_alpha", 0.1))]

    # algo extras
    name = algo_cfg["name"].lower()
    p = algo_cfg.get("params", {})
    if name == "obi-static-alpha":
        args += [
            "--look-depth-pct", str(p.get("look_depth_pct", 0.02)),
            "--alpha-scale", str(p.get("alpha_scale", 50.0)),
        ]
        if p.get("normalize", True):
            args += ["--normalize", "true"]
        else:
            args += ["--normalize", "false"]
    elif name in ("vamp", "vamp-effective"):
        args += ["--vamp-depth-pct", str(p.get("vamp_depth_pct", 0.02))]
    elif name == "weighted-depth":
        args += ["--target-qty-per-side", str(p.get("target_qty_per_side", 500.0))]

    # variable-length files last (to keep parsing simple)
    args += ["--data-files", *data_files]
    if latency_files:
        args += ["--latency-files", *latency_files]
    else:
        args += ["--latency-files"]  # empty is allowed

    return args

def _symbol_params(symbol:str, tickers:Dict[str, Any], cfg:Dict[str, Any]) -> Dict[str, Any]:
    # tick/lot from tickers.json if present
    dflt = cfg["defaults"]
    info = tickers.get(symbol, {})
    tick_size = float(info.get("tick_size", dflt["tick_size"]))
    lot_size  = float(info.get("lot_size",  dflt["lot_size"]))
    wap = float(info.get("weighted_avg_price", 100.0))
    min_qty = float(info.get("min_qty", lot_size))

    grid = cfg["grid"].copy()
    # order qty ~ fixed USD notion
    if symbol.startswith("1000"):
        px = 1000.0 * wap
    else:
        px = wap
    order_qty100 = round((grid["order_value_usd"] / px) / lot_size) * lot_size
    grid["order_qty"] = max(min_qty, order_qty100)
    grid["max_position"] = grid["max_position_in_grids"] * grid["order_qty"]

    if grid.get("skew_override") is None:
        grid["skew"] = grid["relative_half_spread"] / grid["grid_num"]
    else:
        grid["skew"] = float(grid["skew_override"])

    return dict(tick_size=tick_size, lot_size=lot_size, grid=grid)

def _run_one(args: Dict[str, Any]) -> int:
    cmd = _build_cmd(**args)
    # Use list invocation to avoid shell quoting issues
    proc = subprocess.run(cmd)
    print(f"{args['name']}: return={proc.returncode}")
    return proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="data_pipeline/backtest_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg["base_root"]
    exch = cfg["exchange"]
    symbols = cfg["symbols"]
    date_from = int(cfg["date_from"])
    date_to   = int(cfg["date_to"])
    dates = _date_range(date_from, date_to)

    with open(cfg["tickers_json"], "r", encoding="utf-8") as f:
        tickers = json.load(f)

    # time control hooks for future search
    time_ctrl = dict(
        elapse_ns = cfg.get("elapse_ns", 1_000_000_000),
        record_every = cfg.get("record_every", 1)
    )

    os.makedirs(cfg["out_path"], exist_ok=True)

    jobs = []
    for sym in symbols:
        p = _symbol_params(sym, tickers, cfg)
        files = [_files_for(base, exch, sym, d) for d in dates]
        data_files = [f["data"] for f in files]
        latency_files = [f["lat"] for f in files if os.path.exists(f["lat"])]  # allow empty

        jobs.append(dict(
            binary = cfg["binary"],
            name = sym,
            out_path = cfg["out_path"],
            data_files = data_files,
            latency_files = latency_files,
            tick_size = p["tick_size"],
            lot_size  = p["lot_size"],
            maker_fee = cfg["fees"]["maker"],
            taker_fee = cfg["fees"]["taker"],
            queue_power = cfg.get("queue_power", 3.0),
            grid = p["grid"],
            time_ctrl = time_ctrl,
            algo_cfg = cfg["algo"],
            xform = cfg["transform"],
            initial_snapshot = cfg.get("initial_snapshot"),
        ))

    # -------- hooks for grid/Optuna (placeholder) --------
    # You can wrap `jobs` expansion here by generating parameter grids or Optuna trials.
    # Example:
    # for sym in symbols:
    #     for rel_spread in [0.0003, 0.0005, 0.0008]:
    #         ...

    with Pool(processes=int(cfg.get("num_proc", 4))) as pool:
        ret = pool.map(_run_one, jobs)

    bad = sum(1 for r in ret if r != 0)
    print(f"Done. {len(ret)-bad} OK / {bad} FAIL")

if __name__ == "__main__":
    main()
