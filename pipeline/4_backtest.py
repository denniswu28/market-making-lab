from __future__ import annotations
import argparse
import json
import yaml
import os
import glob
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import Dict, List, Any
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter


def _date_range(d0: int, d1: int) -> List[str]:
    s = datetime.strptime(str(d0), "%Y%m%d")
    e = datetime.strptime(str(d1), "%Y%m%d")
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out


def _files_for(
    base_root: str, exchange: str, symbol: str, yyyymmdd: str
) -> Dict[str, str]:
    data = os.path.join(base_root, "data", exchange, symbol, f"{symbol}_{yyyymmdd}.npz")
    lat = os.path.join(
        base_root, "latency", exchange, symbol, f"latency_{yyyymmdd}.npz"
    )
    return {"data": data, "lat": lat}


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
    initial_snapshot: str | None,
) -> List[str]:
    args: List[str] = [
        binary,
        "--name",
        name,
        "--output-path",
        out_path,
        "--tick-size",
        str(tick_size),
        "--lot-size",
        str(lot_size),
        "--maker-fee",
        str(maker_fee),
        "--taker-fee",
        str(taker_fee),
        "--queue-power",
        str(queue_power),
        "--relative-half-spread",
        str(grid["relative_half_spread"]),
        "--relative-grid-interval",
        str(grid["relative_grid_interval"]),
        "--grid-num",
        str(grid["grid_num"]),
        "--order-qty",
        str(grid["order_qty"]),
        "--max-position",
        str(grid["max_position"]),
        "--skew",
        str(grid["skew"]),
        "--elapse-ns",
        str(time_ctrl.get("elapse_ns", 1_000_000_000)),
        "--record-every",
        str(time_ctrl.get("record_every", 1)),
        "--algo",
        algo_cfg["name"],
        "--transform",
        xform["kind"],
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
            "--look-depth-pct",
            str(p.get("look_depth_pct", 0.02)),
            "--alpha-scale",
            str(p.get("alpha_scale", 50.0)),
        ]
        if p.get("normalize", True):
            args += ["--normalize"]  # <— no "true"/"false" here
    elif name in ("vamp", "vamp-effective"):
        args += ["--vamp-depth-pct", str(p.get("vamp_depth_pct", 0.02))]
    elif name == "weighted-depth":
        args += ["--target-qty-per-side", str(p.get("target_qty_per_side", 500.0))]
    else:
        raise ValueError(f"Unsupported algo: {name}")

    # variable-length files last (to keep parsing simple)
    args += ["--data-files", *data_files]
    if latency_files:
        args += ["--latency-files", *latency_files]
    else:
        args += ["--latency-files"]  # empty is allowed
    # print(args)
    return args


def _symbol_params(
    symbol: str, tickers: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    # tick/lot from tickers.json if present
    dflt = cfg["defaults"]
    info = tickers.get(symbol, {})
    # print(info)
    tick_size = float(info.get("tick_size", dflt["tick_size"]))
    lot_size = float(info.get("lot_size", dflt["lot_size"]))
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


def _run_one(args: dict) -> int:
    cmd = _build_cmd(
        **{k: v for k, v in args.items() if k not in ("rust_log", "rust_backtrace")}
    )

    env = os.environ.copy()
    if args.get("rust_log"):
        env["RUST_LOG"] = args["rust_log"]
    if args.get("rust_backtrace") is not None:
        env["RUST_BACKTRACE"] = str(args["rust_backtrace"])

    # Optional: print once for visibility
    print(
        f"Launching with RUST_LOG={env.get('RUST_LOG')} RUST_BACKTRACE={env.get('RUST_BACKTRACE')}"
    )

    proc = subprocess.run(cmd, env=env)
    print(f"{args['name']}: return={proc.returncode}")
    return proc.returncode


def _first_result_csv(out_path: str, symbol: str) -> str | None:
    """
    The Rust backtester writes {out_path}/{name}{asset_index}.csv
    We grab the first match, e.g., SOLUSDT0.csv.
    """
    patt = os.path.join(out_path, f"{symbol}*.csv")
    matches = sorted(glob.glob(patt))
    return matches[0] if matches else None


def _read_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # make sure we keep a DateTimeIndex
    df.index = pd.to_datetime(df["timestamp"])
    # normalize expected column names
    if "price" not in df.columns and "mid_price" in df.columns:
        df = df.rename(columns={"mid_price": "price"})
    return df


def _approx_daily_trades(df: pd.DataFrame, usd_per_order: float) -> float:
    """
    Approximate a daily trade count using:
      - notional_turnover ≈ (|Δposition| rolling 1d sum) * (last mid of the day)
      - trades ≈ notional_turnover / usd_per_order
    """
    pos = df["position"]
    mid = df["price"]
    # daily last mid price (to scale turnover)
    mid_1d_last = mid.resample("1D").last()
    # 1D rolling sum of abs position changes (in qty), sampled at day end
    notional_qty = pos.diff().abs().rolling("1D").sum().resample("1D").last()
    notional_turnover = notional_qty * mid_1d_last
    approx_trades = (notional_turnover / max(1e-9, usd_per_order)).dropna()
    return float(approx_trades.mean()) if len(approx_trades) else 0.0


def _summarize_and_plot(
    out_path: str, symbols: list[str], usd_per_order: float
) -> None:
    os.makedirs(out_path, exist_ok=True)
    plot_dir = os.path.join(out_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    sel_pairs: list[str] = []
    total_equity_5m: pd.Series | None = None
    rows = []

    for i, sym in enumerate(symbols):
        result_csv = _first_result_csv(out_path, sym)
        if not result_csv or not os.path.exists(result_csv):
            print(f"[summary] WARN: no result CSV for {sym} in {out_path}")
            continue

        df = _read_result_csv(result_csv)

        # Equity = cash + position * price - fee
        px_col = (
            "price"
            if "price" in df.columns
            else ("mid_price" if "mid_price" in df.columns else None)
        )
        if px_col is None:
            print(f"[summary] WARN: no price/mid_price in {result_csv}; skipping {sym}")
            continue

        # equity = cash + position * mid - fee
        equity = df["balance"] + df["position"] * df["price"] - df["fee"]
        equity_5m = equity.resample("5min").last()

        avg_daily_trades = _approx_daily_trades(df, usd_per_order)

        fig = plt.figure(i, figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.set_title(f"{sym}, approx avg daily trades: {avg_daily_trades:.0f}")
        ax.set_ylabel("Equity $")

        ax.plot(equity_5m.index, equity_5m, label="Equity")

        # date ticks: concise, pretty
        locator = AutoDateLocator()
        formatter = ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.legend(loc="upper left")

        ax_pos = ax.twinx()
        ax_pos.set_ylabel("Position Qty")
        pos_5m = df["position"].resample("5min").last()
        ax_pos.plot(pos_5m.index, pos_5m, alpha=0.5, label="Position Qty")
        ax_pos.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{sym}.png"))
        plt.close(fig)

        # Winner selection
        if len(equity) and equity.iloc[-1] > equity.iloc[0]:
            sel_pairs.append(sym)
            total_equity_5m = (
                equity_5m
                if total_equity_5m is None
                else (total_equity_5m.add(equity_5m, fill_value=0.0))
            )

        # Basic stats row
        start_eq = float(equity.iloc[0]) if len(equity) else 0.0
        end_eq = float(equity.iloc[-1]) if len(equity) else 0.0
        ret_abs = end_eq - start_eq
        ret_pct = (ret_abs / start_eq * 100.0) if start_eq != 0 else float("nan")
        rows.append(
            dict(
                symbol=sym,
                start_equity=start_eq,
                end_equity=end_eq,
                ret_abs=ret_abs,
                ret_pct=ret_pct,
                approx_avg_daily_trades=avg_daily_trades,
                csv=result_csv,
                plot=os.path.join(plot_dir, f"{sym}.png"),
            )
        )

    # Combined equity (winners)
    if total_equity_5m is not None and len(total_equity_5m.dropna()):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

        ax.set_title(f"Combined Equity of {len(sel_pairs)} winning pairs")
        ax.set_ylabel("Equity $")
        ax.plot(total_equity_5m, label="Combined Equity")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.autofmt_xdate()
        fig.savefig(os.path.join(plot_dir, "combined_equity.png"))
        plt.close(fig)
        print(f"[summary] Winners: {sel_pairs}")
    else:
        print("[summary] No winners or no equity series to combine.")

    # Summary CSV
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_path, "summary.csv"), index=False)
        print(f"[summary] Wrote {os.path.join(out_path, 'summary.csv')}")
    else:
        print("[summary] Nothing to summarize.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="pipeline/backtest_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg["base_root"]
    exch = cfg["exchange"]
    symbols = cfg["symbols"]
    date_from = int(cfg["date_from"])
    date_to = int(cfg["date_to"])
    dates = _date_range(date_from, date_to)

    with open(cfg["tickers_json"], "r", encoding="utf-8") as f:
        tickers = json.load(f)

    # time control hooks for future search
    time_ctrl = dict(
        elapse_ns=cfg.get("elapse_ns", 1_000_000_000),
        record_every=cfg.get("record_every", 1),
    )

    os.makedirs(cfg["out_path"], exist_ok=True)

    jobs = []
    for sym in symbols:
        p = _symbol_params(sym, tickers, cfg)
        files = [_files_for(base, exch, sym, d) for d in dates]
        data_files = [f["data"] for f in files]
        latency_files = [f["lat"] for f in files if os.path.exists(f["lat"])]

        jobs.append(
            dict(
                binary=cfg["binary"],
                name=sym,
                out_path=cfg["out_path"],
                data_files=data_files,
                latency_files=latency_files,
                tick_size=p["tick_size"],
                lot_size=p["lot_size"],
                maker_fee=cfg["fees"]["maker"],
                taker_fee=cfg["fees"]["taker"],
                queue_power=cfg.get("queue_power", 3.0),
                grid=p["grid"],
                time_ctrl=time_ctrl,
                algo_cfg=cfg["algo"],
                xform=cfg["transform"],
                initial_snapshot=cfg.get("initial_snapshot"),
                rust_log=cfg.get("rust_log"),
                rust_backtrace=cfg.get("rust_backtrace"),
            )
        )

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

    usd_per_order = float(cfg["grid"].get("order_value_usd", 100.0))
    _summarize_and_plot(cfg["out_path"], symbols, usd_per_order)


if __name__ == "__main__":
    main()
