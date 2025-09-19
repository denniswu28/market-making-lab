from __future__ import annotations
import os
import sys
import argparse
import yaml
import shutil
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl
from tqdm.auto import tqdm
from hftbacktest.data.utils import tardis  # tardis.convert / tardis.convert_fuse


# ---------------------------- schema helpers ----------------------------

TRADES_DTYPES = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "id": pl.String,       
    "side": pl.String,
    "price": pl.Float64,
    "amount": pl.Float64,
}

DEPTH_DTYPES = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "is_snapshot": pl.Boolean,
    "side": pl.String,
    "price": pl.Float64,
    "amount": pl.Float64,
}

QUOTES_DTYPES = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "ask_amount": pl.Float64,
    "ask_price": pl.Float64,
    "bid_price": pl.Float64,
    "bid_amount": pl.Float64,
}

TRADES_COLS = [
    "exchange", "symbol",
    "timestamp", "local_timestamp",
    "id", "side", "price", "amount",
]

DEPTH_COLS = [
    "exchange", "symbol",
    "timestamp", "local_timestamp",
    "is_snapshot", "side", "price", "amount",
]

QUOTES_COLS = [
    "exchange", "symbol",
    "timestamp", "local_timestamp",
    "ask_amount", "ask_price", "bid_price", "bid_amount",
]

def _cast_then_select(df: pl.DataFrame, order: list[str], dtypes: dict[str, pl.PolarsDataType]) -> pl.DataFrame:
    # cast with strict=False so ints->strings, 0/1->bool, etc. won’t error
    casts = []
    for c in order:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
        casts.append(pl.col(c).cast(dtypes[c], strict=False).alias(c))
    return df.select(casts).rechunk()

def _norm_date(d: str) -> Tuple[str, str]:
    s = d.strip().replace("/", "-")
    if "-" in s:
        yyyy, mm, dd = s.split("-")
        return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}", f"{yyyy}{mm.zfill(2)}{dd.zfill(2)}"
    else:
        s = s.strip()
        if len(s) != 8 or not s.isdigit():
            raise ValueError(f"Bad date: {d}")
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}", s


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _pick_snapshot_mode(global_mode: str, this_date_ymd: str, sod_ymd: Optional[str]) -> str:
    if global_mode != "auto":
        return global_mode
    if sod_ymd and (this_date_ymd == sod_ymd):
        return "process"
    return "ignore_sod"


@dataclass
class Job:
    exchange: str
    symbol: str
    date_dash: str
    date_ymd: str
    data_root: str
    output_root: str
    tmp_root: str
    keep_temps: bool
    use_quotes: bool
    tick_size: Optional[float]
    lot_size: Optional[float]
    snapshot_mode: str
    buffer_size: int
    ss_buffer_size: int
    base_latency: float
    output_format: str  # 'npz' or 'npy'
    delete_inputs_after: bool


def _build_src_paths(j: Job) -> Dict[str, str]:
    base = os.path.join(j.data_root, j.exchange)
    return {
        "trades": os.path.join(base, "trades", j.date_dash, f"{j.symbol}.parquet"),
        "depth": os.path.join(base, "incremental_book_L2", j.date_dash, f"{j.symbol}.parquet"),
        "quotes": os.path.join(base, "quotes", j.date_dash, f"{j.symbol}.parquet"),
    }


def _output_path(j: Job) -> str:
    out_dir = os.path.join(j.output_root, j.exchange, j.symbol)
    _ensure_dir(out_dir)
    ext = ".npz" if j.output_format == "npz" else ".npy"
    return os.path.join(out_dir, f"{j.symbol}_{j.date_ymd}{ext}")


def _tmp_path(j: Job, kind: str) -> str:
    # unique per (exchange, symbol, date, kind)
    d = os.path.join(j.tmp_root, j.exchange, j.symbol, j.date_ymd)
    _ensure_dir(d)
    return os.path.join(d, f"{kind}.parquet")


def _standardize_quotes_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Tardis sometimes uses 'ask_size'/'bid_size'
    rename_map = {}
    if "ask_size" in df.columns and "ask_amount" not in df.columns:
        rename_map["ask_size"] = "ask_amount"
    if "bid_size" in df.columns and "bid_amount" not in df.columns:
        rename_map["bid_size"] = "bid_amount"
    if rename_map:
        df = df.rename(rename_map)
    return df


def _coerce_and_select(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    # Ensure order and dtypes that play nice with hftbacktest's Polars -> NumPy conversions
    # (We avoid over-casting; Polars will cast as needed inside tardis.convert.)
    return df.select(cols)


def _prepare_parquet(src_path: str, dst_path: str, kind: str, exchange: str, symbol: str) -> str:
    df = pl.read_parquet(src_path)

    # Ensure dummy ‘exchange’/‘symbol’
    if "exchange" not in df.columns:
        df = df.with_columns(pl.lit(exchange).alias("exchange"))
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    if kind == "trades":
        # Some drops have qty instead of amount
        if "amount" not in df.columns and "qty" in df.columns:
            df = df.rename({"qty": "amount"})
        df = _cast_then_select(df, TRADES_COLS, TRADES_DTYPES)

    elif kind == "depth":
        df = _cast_then_select(df, DEPTH_COLS, DEPTH_DTYPES)

    elif kind == "quotes":
        # Normalize ask_size/bid_size -> ask_amount/bid_amount
        if "ask_size" in df.columns and "ask_amount" not in df.columns:
            df = df.rename({"ask_size": "ask_amount"})
        if "bid_size" in df.columns and "bid_amount" not in df.columns:
            df = df.rename({"bid_size": "bid_amount"})
        df = _cast_then_select(df, QUOTES_COLS, QUOTES_DTYPES)

    else:
        raise ValueError(kind)

    df.write_parquet(dst_path)
    return dst_path

def _convert_one(j: Job) -> Tuple[str, str, Optional[str], List[str]]:
    logs: List[str] = []
    try:
        logs.append(f"[START] exch={j.exchange} sym={j.symbol} date={j.date_ymd} "
                    f"mode={j.snapshot_mode} quotes={j.use_quotes}")
        src = _build_src_paths(j)
        tmp_trades = _tmp_path(j, "trades")
        tmp_depth  = _tmp_path(j, "depth")
        tmp_quotes = _tmp_path(j, "quotes")

        # Always prepare the standardized inputs with dummy columns
        trades_file = _prepare_parquet(src["trades"], tmp_trades, "trades", j.exchange, j.symbol)
        depth_file  = _prepare_parquet(src["depth"],  tmp_depth,  "depth",  j.exchange, j.symbol)
        logs.append(f"[READ] trades={trades_file}")
        logs.append(f"[READ] depth={depth_file}")

        out_file = _output_path(j)

        if j.use_quotes:
            quotes_file = _prepare_parquet(src["quotes"], tmp_quotes, "quotes", j.exchange, j.symbol)
            logs.append(f"[READ] quotes={quotes_file}")
            if j.tick_size is None or j.lot_size is None:
                logs.append("[ERROR] tick_size/lot_size missing for quotes fuse")
                return (j.symbol, j.date_ymd, "tick_size and lot_size required when use_quotes=true", logs)

            tardis.convert_fuse(
                trades_filename=trades_file,
                depth_filename=depth_file,
                book_ticker_filename=quotes_file,
                tick_size=j.tick_size,
                lot_size=j.lot_size,
                output_filename=(out_file if j.output_format == "npz" else None),
                ss_buffer_size=j.ss_buffer_size,
                base_latency=j.base_latency,
                snapshot_mode=j.snapshot_mode,
            )
        else:
            tardis.convert(
                input_files=[trades_file, depth_file],  # trades first
                output_filename=(out_file if j.output_format == "npz" else None),
                buffer_size=j.buffer_size,
                ss_buffer_size=j.ss_buffer_size,
                base_latency=j.base_latency,
                snapshot_mode=j.snapshot_mode,
            )

        if j.output_format == "npy":
            import numpy as np
            if j.use_quotes:
                arr = tardis.convert_fuse(
                    trades_filename=trades_file,
                    depth_filename=depth_file,
                    book_ticker_filename=quotes_file,
                    tick_size=j.tick_size,
                    lot_size=j.lot_size,
                    output_filename=None,
                    ss_buffer_size=j.ss_buffer_size,
                    base_latency=j.base_latency,
                    snapshot_mode=j.snapshot_mode,
                )
            else:
                arr = tardis.convert(
                    input_files=[trades_file, depth_file],
                    output_filename=None,
                    buffer_size=j.buffer_size,
                    ss_buffer_size=j.ss_buffer_size,
                    base_latency=j.base_latency,
                    snapshot_mode=j.snapshot_mode,
                )
            np.save(_output_path(j), arr)

        logs.append(f"[OUT] {out_file}")

        if not j.keep_temps:
            base_tmp = os.path.join(j.tmp_root, j.exchange, j.symbol, j.date_ymd)
            shutil.rmtree(base_tmp, ignore_errors=True)

        if j.delete_inputs_after:
            for p in src.values():
                try:
                    os.remove(p)
                    logs.append(f"[CLEAN] deleted {p}")
                except Exception:
                    pass

        return (j.symbol, j.date_ymd, None, logs)

    except Exception as e:
        logs.append(f"[EXCEPTION] {e}")
        return (j.symbol, j.date_ymd, f"{e}\n{traceback.format_exc(limit=3)}", logs)


def main():
    parser = argparse.ArgumentParser(description="Convert Tardis parquet to hftbacktest npz/npy (adds dummy exchange/symbol).")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root: str = cfg["data_root"]
    output_root: str = cfg["output_root"]
    tmp_root: str = cfg.get("tmp_root", os.path.join(output_root, ".tmp"))
    keep_temps: bool = bool(cfg.get("keep_temps", False))

    exchange: str = cfg["exchange"]
    symbols: List[str] = cfg["symbols"]

    dates_in: List[str] = cfg["dates"]
    sod_date: Optional[str] = cfg.get("sod_date")
    snapshot_mode_global: str = cfg.get("snapshot_mode", "auto")

    use_quotes: bool = bool(cfg.get("use_quotes", False))
    tick_size_map: Dict[str, float] = cfg.get("tick_size", {}) or {}
    lot_size_map: Dict[str, float] = cfg.get("lot_size", {}) or {}

    num_proc: int = int(cfg.get("num_proc", max(1, mp.cpu_count() // 2)))
    buffer_size: int = int(cfg.get("buffer_size", 100_000_000))
    ss_buffer_size: int = int(cfg.get("ss_buffer_size", 1_000_000))
    base_latency: float = float(cfg.get("base_latency", 0.0))
    delete_inputs_after: bool = bool(cfg.get("delete_inputs_after", False))
    output_format: str = str(cfg.get("output_format", "npz")).lower()
    if output_format not in ("npz", "npy"):
        raise ValueError("output_format must be 'npz' or 'npy'")

    # Normalize dates
    norm_dates = [_norm_date(d) for d in dates_in]
    sod_dash, sod_ymd = (None, None)
    if sod_date:
        sod_dash, sod_ymd = _norm_date(sod_date)

    # Jobs
    jobs: List[Job] = []
    for sym in symbols:
        for d_dash, d_ymd in norm_dates:
            smode = _pick_snapshot_mode(snapshot_mode_global, d_ymd, sod_ymd)
            jobs.append(
                Job(
                    exchange=exchange,
                    symbol=sym,
                    date_dash=d_dash,
                    date_ymd=d_ymd,
                    data_root=data_root,
                    output_root=output_root,
                    tmp_root=tmp_root,
                    keep_temps=keep_temps,
                    use_quotes=use_quotes,
                    tick_size=tick_size_map.get(sym),
                    lot_size=lot_size_map.get(sym),
                    snapshot_mode=smode,
                    buffer_size=buffer_size,
                    ss_buffer_size=ss_buffer_size,
                    base_latency=base_latency,
                    output_format=output_format,
                    delete_inputs_after=delete_inputs_after,
                )
            )

    print(f"[convert] exchange={exchange} symbols={symbols} dates={[d[1] for d in norm_dates]}")
    print(f"[convert] use_quotes={use_quotes} snapshot_mode={snapshot_mode_global} sod_date={sod_ymd}")
    print(f"[convert] num_proc={num_proc} buffer_size={buffer_size} ss_buffer_size={ss_buffer_size} base_latency={base_latency}")
    print(f"[convert] output_format={output_format} output_root={output_root} tmp_root={tmp_root}")

    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context("fork")
    total = len(jobs)
    with ctx.Pool(processes=num_proc) as pool:
        it = pool.imap_unordered(_convert_one, jobs, chunksize=1)
        for sym, d, err, logs in tqdm(it, total=total, desc="Converting", unit="combo"):
            # print the worker logs cleanly without breaking the bar
            for line in logs:
                tqdm.write(line)
            if err is None:
                tqdm.write(f"[OK] {sym} {d}")
            else:
                tqdm.write(f"[FAIL] {sym} {d} -> {err}")


if __name__ == "__main__":
    main()
