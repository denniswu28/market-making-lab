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
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from tqdm.auto import tqdm
from numba import njit

# hftbacktest
from hftbacktest.data.utils import tardis          # convert / convert_fuse
from hftbacktest import EXCH_EVENT, LOCAL_EVENT    # bitmasks for filter

# -------------------------------- schemas for converter --------------------------------

TRADES_DTYPES = {
    "exchange": pl.String, "symbol": pl.String,
    "timestamp": pl.Int64, "local_timestamp": pl.Int64,
    "id": pl.String, "side": pl.String, "price": pl.Float64, "amount": pl.Float64,
}
DEPTH_DTYPES = {
    "exchange": pl.String, "symbol": pl.String,
    "timestamp": pl.Int64, "local_timestamp": pl.Int64,
    "is_snapshot": pl.Boolean, "side": pl.String, "price": pl.Float64, "amount": pl.Float64,
}
QUOTES_DTYPES = {
    "exchange": pl.String, "symbol": pl.String,
    "timestamp": pl.Int64, "local_timestamp": pl.Int64,
    "ask_amount": pl.Float64, "ask_price": pl.Float64, "bid_price": pl.Float64, "bid_amount": pl.Float64,
}
TRADES_COLS = ["exchange","symbol","timestamp","local_timestamp","id","side","price","amount"]
DEPTH_COLS  = ["exchange","symbol","timestamp","local_timestamp","is_snapshot","side","price","amount"]
QUOTES_COLS = ["exchange","symbol","timestamp","local_timestamp","ask_amount","ask_price","bid_price","bid_amount"]

def _cast_then_select(df: pl.DataFrame, order: List[str], dtypes: Dict[str, pl.DataType]) -> pl.DataFrame:
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
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}", s
    raise ValueError(f"Bad date: {d}")

def _iter_dates(dates: Optional[List[str]], date_from: Optional[str], date_to: Optional[str]) -> List[Tuple[str,str]]:
    if dates:
        return [_norm_date(d) for d in dates]
    if not (date_from and date_to):
        raise ValueError("Provide 'dates' OR ('date_from' & 'date_to').")
    d0 = datetime.strptime(_norm_date(date_from)[1], "%Y%m%d")
    d1 = datetime.strptime(_norm_date(date_to)[1], "%Y%m%d")
    out, cur = [], d0
    while cur <= d1:
        ymd = cur.strftime("%Y%m%d")
        out.append((f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}", ymd))
        cur += timedelta(days=1)
    return out

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _pick_snapshot_mode(global_mode: str, this_date_ymd: str, sod_ymd: Optional[str]) -> str:
    if global_mode != "auto":
        return global_mode
    return "process" if (sod_ymd and this_date_ymd == sod_ymd) else "ignore_sod"

# -------------------------------- converter helpers --------------------------------

@dataclass
class ConvertCfg:
    data_root: str
    feeds_root: str
    tmp_root: str
    keep_temps: bool
    exchange: str
    use_quotes: bool
    tick_size_map: Dict[str, float]
    lot_size_map: Dict[str, float]
    snapshot_mode: str
    sod_ymd: Optional[str]
    buffer_size: int
    ss_buffer_size: int
    base_latency: float
    output_format: str        # must be 'npz' for this pipeline
    delete_inputs_after: bool
    skip_existing_feed: bool

@dataclass
class LatencyCfg:
    latency_root: str
    bucket_every: str
    mul_entry: float
    offset_entry: int
    mul_resp: float
    offset_resp: int
    include_symbol_dir: bool
    latency_filename_template: str
    overwrite_latency: bool

@dataclass
class Job:
    symbol: str
    date_dash: str
    date_ymd: str
    conv: ConvertCfg
    lat: LatencyCfg

def _build_src_paths(conv: ConvertCfg, symbol: str, date_dash: str) -> Dict[str, str]:
    base = os.path.join(conv.data_root, conv.exchange)
    return {
        "trades": os.path.join(base, "trades", date_dash, f"{symbol}.parquet"),
        "depth":  os.path.join(base, "incremental_book_L2", date_dash, f"{symbol}.parquet"),
        "quotes": os.path.join(base, "quotes", date_dash, f"{symbol}.parquet"),
    }

def _feed_out_path(conv: ConvertCfg, symbol: str, date_ymd: str) -> str:
    # /feeds_root/{exchange}/{symbol}/{symbol}_{YYYYMMDD}.npz
    out_dir = os.path.join(conv.feeds_root, conv.exchange, symbol)
    _ensure_dir(out_dir)
    ext = ".npz"  # pipeline expects npz with key 'data'
    return os.path.join(out_dir, f"{symbol}_{date_ymd}{ext}")

def _latency_out_path(lat: LatencyCfg, conv: ConvertCfg, symbol: str, date_ymd: str) -> str:
    fname = lat.latency_filename_template.replace("{date}", date_ymd)
    out_dir = (os.path.join(lat.latency_root, conv.exchange, symbol)
               if lat.include_symbol_dir else
               os.path.join(lat.latency_root, conv.exchange))
    _ensure_dir(out_dir)
    return os.path.join(out_dir, fname)

def _tmp_path(conv: ConvertCfg, symbol: str, date_ymd: str, kind: str) -> str:
    d = os.path.join(conv.tmp_root, conv.exchange, symbol, date_ymd)
    _ensure_dir(d)
    return os.path.join(d, f"{kind}.parquet")

def _prepare_parquet(src_path: str, dst_path: str, kind: str, exchange: str, symbol: str) -> str:
    df = pl.read_parquet(src_path)
    if "exchange" not in df.columns:
        df = df.with_columns(pl.lit(exchange).alias("exchange"))
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    if kind == "trades":
        if "amount" not in df.columns and "qty" in df.columns:
            df = df.rename({"qty": "amount"})
        df = _cast_then_select(df, TRADES_COLS, TRADES_DTYPES)
    elif kind == "depth":
        df = _cast_then_select(df, DEPTH_COLS, DEPTH_DTYPES)
    elif kind == "quotes":
        if "ask_size" in df.columns and "ask_amount" not in df.columns:
            df = df.rename({"ask_size": "ask_amount"})
        if "bid_size" in df.columns and "bid_amount" not in df.columns:
            df = df.rename({"bid_size": "bid_amount"})
        df = _cast_then_select(df, QUOTES_COLS, QUOTES_DTYPES)
    else:
        raise ValueError(kind)

    df.write_parquet(dst_path)
    return dst_path

# -------------------------------- latency kernel --------------------------------

@njit
def _latency_nb(data, out_arr, mul_entry, offset_entry, mul_resp, offset_resp):
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts
        entry = mul_entry * feed_latency + offset_entry
        resp  = mul_resp  * feed_latency + offset_resp
        req_ts = local_ts
        order_exch_ts = req_ts + entry
        resp_ts = order_exch_ts + resp
        out_arr[i].req_ts  = req_ts
        out_arr[i].exch_ts = order_exch_ts
        out_arr[i].resp_ts = resp_ts

# -------------------------------- per-job worker --------------------------------

def _run_one(job: Job) -> Tuple[str, str, Optional[str], List[str]]:
    logs: List[str] = []
    sym, d = job.symbol, job.date_ymd
    try:
        logs.append(f"[START] sym={sym} date={d} quotes={job.conv.use_quotes} "
                    f"mode={_pick_snapshot_mode(job.conv.snapshot_mode, d, job.conv.sod_ymd)}")

        # ---------- CONVERT ----------
        feed_out = _feed_out_path(job.conv, sym, d)
        if job.conv.output_format != "npz":
            return (sym, d, "output_format must be 'npz' for latency step.", logs)

        if os.path.exists(feed_out) and job.conv.skip_existing_feed:
            logs.append(f"[SKIP-CONVERT] exists: {feed_out}")
        else:
            src = _build_src_paths(job.conv, sym, job.date_dash)
            for need, path in src.items():
                if need in ("trades","depth") and not os.path.exists(path):
                    msg = f"Missing input {need}: {path}"
                    logs.append(f"[MISS] {msg}")
                    return (sym, d, msg, logs)

            tmp_trades = _tmp_path(job.conv, sym, d, "trades")
            tmp_depth  = _tmp_path(job.conv, sym, d, "depth")
            trades_file = _prepare_parquet(src["trades"], tmp_trades, "trades", job.conv.exchange, sym)
            depth_file  = _prepare_parquet(src["depth"],  tmp_depth,  "depth",  job.conv.exchange, sym)
            logs.append(f"[READ] trades={trades_file}")
            logs.append(f"[READ] depth={depth_file}")

            snap_mode = _pick_snapshot_mode(job.conv.snapshot_mode, d, job.conv.sod_ymd)

            if job.conv.use_quotes:
                quotes_path = src["quotes"]
                if not os.path.exists(quotes_path):
                    msg = f"Missing input quotes: {quotes_path}"
                    logs.append(f"[MISS] {msg}")
                    return (sym, d, msg, logs)
                tmp_quotes = _tmp_path(job.conv, sym, d, "quotes")
                quotes_file = _prepare_parquet(quotes_path, tmp_quotes, "quotes", job.conv.exchange, sym)
                logs.append(f"[READ] quotes={quotes_file}")

                ts = job.conv.tick_size_map.get(sym)
                ls = job.conv.lot_size_map.get(sym)
                if ts is None or ls is None:
                    msg = "tick_size and lot_size required when use_quotes=true"
                    logs.append(f"[ERROR] {msg}")
                    return (sym, d, msg, logs)

                tardis.convert_fuse(
                    trades_filename=trades_file,
                    depth_filename=depth_file,
                    book_ticker_filename=quotes_file,
                    tick_size=ts, lot_size=ls,
                    output_filename=feed_out,
                    ss_buffer_size=job.conv.ss_buffer_size,
                    base_latency=job.conv.base_latency,
                    snapshot_mode=snap_mode,
                )
            else:
                tardis.convert(
                    input_files=[trades_file, depth_file],  # trades first
                    output_filename=feed_out,
                    buffer_size=job.conv.buffer_size,
                    ss_buffer_size=job.conv.ss_buffer_size,
                    base_latency=job.conv.base_latency,
                    snapshot_mode=snap_mode,
                )

            logs.append(f"[OUT-FEED] {feed_out}")

            if not job.conv.keep_temps:
                base_tmp = os.path.join(job.conv.tmp_root, job.conv.exchange, sym, d)
                shutil.rmtree(base_tmp, ignore_errors=True)

            if job.conv.delete_inputs_after:
                for p in src.values():
                    try:
                        os.remove(p)
                        logs.append(f"[CLEAN] deleted {p}")
                    except Exception:
                        pass

        # ---------- LATENCY ----------
        if not os.path.exists(feed_out):
            msg = f"Feed not found for latency: {feed_out}"
            logs.append(f"[MISS] {msg}")
            return (sym, d, msg, logs)

        lat_out = _latency_out_path(job.lat, job.conv, sym, d)
        if os.path.exists(lat_out) and not job.lat.overwrite_latency:
            logs.append(f"[SKIP-LAT] exists: {lat_out}")
            return (sym, d, None, logs)

        feed_arr = np.load(feed_out)["data"]  # npz with 'data'
        df = pl.DataFrame(feed_arr)
        df = (
            df.filter(
                (pl.col("ev") & EXCH_EVENT == EXCH_EVENT) &
                (pl.col("ev") & LOCAL_EVENT == LOCAL_EVENT)
            )
            .with_columns(pl.col("local_ts").alias("ts"))
            .group_by_dynamic("ts", every=job.lat.bucket_every)
            .agg(pl.col("exch_ts").last().alias("exch_ts"),
                 pl.col("local_ts").last().alias("local_ts"))
            .drop("ts")
        )
        np_data = df.select(["exch_ts", "local_ts"]).to_numpy(structured=True)
        out_dtype = np.dtype([("req_ts","i8"),("exch_ts","i8"),("resp_ts","i8"),("_padding","i8")])
        out_arr = np.zeros(len(np_data), dtype=out_dtype)
        _latency_nb(np_data, out_arr,
                    job.lat.mul_entry, job.lat.offset_entry,
                    job.lat.mul_resp,  job.lat.offset_resp)
        np.savez_compressed(lat_out, data=out_arr)
        logs.append(f"[OUT-LAT] {lat_out}")

        return (sym, d, None, logs)

    except Exception as e:
        logs.append(f"[EXCEPTION] {e}")
        return (sym, d, f"{e}\n{traceback.format_exc(limit=3)}", logs)

# -------------------------------- main --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Tardis parquet to feeds and derive latency — one pipeline.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Universe
    exchange: str = cfg["exchange"]
    symbols: List[str] = cfg["symbols"]
    date_list: Optional[List[str]] = cfg.get("dates")
    date_from: Optional[str] = cfg.get("date_from")
    date_to: Optional[str]   = cfg.get("date_to")
    dates = _iter_dates(date_list, date_from, date_to)

    # Convert config
    data_root: str   = cfg["data_root"]
    feeds_root: str  = cfg["feeds_output_root"]
    tmp_root: str    = cfg.get("tmp_root", os.path.join(feeds_root, ".tmp"))
    keep_temps: bool = bool(cfg.get("keep_temps", False))

    use_quotes: bool = bool(cfg.get("use_quotes", False))
    tick_size_map: Dict[str, float] = cfg.get("tick_size", {}) or {}
    lot_size_map:  Dict[str, float] = cfg.get("lot_size", {})  or {}

    snapshot_mode_global: str = cfg.get("snapshot_mode", "auto")
    sod_date: Optional[str] = cfg.get("sod_date")
    sod_ymd = _norm_date(sod_date)[1] if sod_date else None

    num_proc: int        = int(cfg.get("num_proc", max(1, mp.cpu_count() // 2)))
    buffer_size: int     = int(cfg.get("buffer_size", 100_000_000))
    ss_buffer_size: int  = int(cfg.get("ss_buffer_size", 1_000_000))
    base_latency: float  = float(cfg.get("base_latency", 0.0))
    output_format: str   = str(cfg.get("output_format", "npz")).lower()
    delete_inputs_after: bool = bool(cfg.get("delete_inputs_after", False))
    skip_existing_feed: bool  = bool(cfg.get("skip_existing_feed", True))

    if output_format != "npz":
        raise ValueError("For this pipeline, output_format must be 'npz' (latency step expects npz with key 'data').")

    conv_cfg_partial = dict(
        data_root=data_root, feeds_root=feeds_root, tmp_root=tmp_root, keep_temps=keep_temps,
        exchange=exchange, use_quotes=use_quotes, tick_size_map=tick_size_map, lot_size_map=lot_size_map,
        snapshot_mode=snapshot_mode_global, sod_ymd=sod_ymd,
        buffer_size=buffer_size, ss_buffer_size=ss_buffer_size, base_latency=base_latency,
        output_format=output_format, delete_inputs_after=delete_inputs_after, skip_existing_feed=skip_existing_feed
    )

    # Latency config
    latency_root: str = cfg["latency_output_root"]
    bucket_every: str = cfg.get("bucket_every", "1000000i")     # 1s for microsecond timestamps
    mul_entry: float  = float(cfg.get("mul_entry", 1))
    mul_resp: float   = float(cfg.get("mul_resp", 1))
    offset_entry: int = int(cfg.get("offset_entry", 0))
    offset_resp: int  = int(cfg.get("offset_resp", 0))
    include_symbol_dir: bool = bool(cfg.get("include_symbol_dir", True))
    latency_filename_template: str = cfg.get("latency_filename_template", "latency_{date}.npz")
    overwrite_latency: bool = bool(cfg.get("overwrite_latency", True))

    lat_cfg_partial = dict(
        latency_root=latency_root, bucket_every=bucket_every,
        mul_entry=mul_entry, offset_entry=offset_entry, mul_resp=mul_resp, offset_resp=offset_resp,
        include_symbol_dir=include_symbol_dir, latency_filename_template=latency_filename_template,
        overwrite_latency=overwrite_latency
    )

    # Build jobs
    jobs: List[Job] = []
    for sym in symbols:
        for d_dash, d_ymd in dates:
            jobs.append(
                Job(
                    symbol=sym, date_dash=d_dash, date_ymd=d_ymd,
                    conv=ConvertCfg(**conv_cfg_partial),
                    lat=LatencyCfg(**lat_cfg_partial),
                )
            )

    print(f"[pipeline] exchange={exchange}")
    print(f"[pipeline] symbols={symbols}")
    print(f"[pipeline] dates={[d[1] for d in dates]}")
    print(f"[pipeline] data_root={data_root}")
    print(f"[pipeline] feeds_root={feeds_root} latency_root={latency_root}")
    print(f"[pipeline] snapshot_mode={snapshot_mode_global} sod_date={sod_ymd}")
    print(f"[pipeline] num_proc={num_proc} buffer_size={buffer_size} ss_buffer_size={ss_buffer_size} base_latency={base_latency}")
    print(f"[pipeline] quotes={use_quotes} bucket_every={bucket_every} mul_entry={mul_entry} mul_resp={mul_resp}")

    # Run parallel; each job does convert then latency
    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context("fork")
    total = len(jobs)
    with ctx.Pool(processes=num_proc) as pool:
        it = pool.imap_unordered(_run_one, jobs, chunksize=1)
        for sym, d, err, logs in tqdm(it, total=total, desc="Convert → Latency", unit="combo"):
            for line in logs:
                tqdm.write(line)
            if err is None:
                tqdm.write(f"[OK] {sym} {d}")
            else:
                tqdm.write(f"[FAIL] {sym} {d} -> {err}")

if __name__ == "__main__":
    main()
