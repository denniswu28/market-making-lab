from __future__ import annotations
import os
import sys
import argparse
import yaml
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from numba import njit
from tqdm.auto import tqdm

# hftbacktest bitmasks
from hftbacktest import EXCH_EVENT, LOCAL_EVENT


# ----------------------------- latency kernel -----------------------------

@njit
def generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp):
    """
    data: structured array with fields ['exch_ts', 'local_ts']
    order_latency: structured array with dtype [('req_ts','i8'), ('exch_ts','i8'), ('resp_ts','i8'), ('_padding','i8')]
    """
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts

        order_entry_latency = mul_entry * feed_latency + offset_entry
        order_resp_latency  = mul_resp  * feed_latency + offset_resp

        req_ts = local_ts
        order_exch_ts = req_ts + order_entry_latency
        resp_ts = order_exch_ts + order_resp_latency

        order_latency[i].req_ts  = req_ts
        order_latency[i].exch_ts = order_exch_ts
        order_latency[i].resp_ts = resp_ts


# ----------------------------- helpers -----------------------------

def _norm_date_strs(d: str) -> Tuple[str, str]:
    """Return ('YYYY-MM-DD', 'YYYYMMDD') from either 'YYYY-MM-DD' or 'YYYYMMDD'."""
    s = d.strip().replace("/", "-")
    if "-" in s:
        yyyy, mm, dd = s.split("-")
        return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}", f"{yyyy}{mm.zfill(2)}{dd.zfill(2)}"
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}", s
    raise ValueError(f"Bad date: {d}")


def _iter_dates(dates: Optional[List[str]], date_from: Optional[str], date_to: Optional[str]) -> List[Tuple[str, str]]:
    if dates:
        return [_norm_date_strs(d) for d in dates]
    if not (date_from and date_to):
        raise ValueError("Provide either 'dates' list or both 'date_from' and 'date_to' in YAML.")
    d0 = datetime.strptime(_norm_date_strs(date_from)[1], "%Y%m%d")
    d1 = datetime.strptime(_norm_date_strs(date_to)[1], "%Y%m%d")
    out: List[Tuple[str, str]] = []
    cur = d0
    while cur <= d1:
        ymd = cur.strftime("%Y%m%d")
        out.append((f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}", ymd))
        cur += timedelta(days=1)
    return out


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _feed_path(feeds_root: str, exchange: str, symbol: str, yyyymmdd: str) -> str:
    # Matches your previous output root structure created by the converter.
    # e.g. /data/tmp/tardis_bn_hft/{exchange}/{symbol}/{YYYYMMDD}.npz
    return os.path.join(feeds_root, exchange, symbol, f"{yyyymmdd}.npz")


def _latency_out_path(latency_root: str, exchange: str, symbol: str, yyyymmdd: str,
                      include_symbol_dir: bool, filename_template: str) -> str:
    # filename_template supports "{date}" placeholder
    fname = filename_template.replace("{date}", yyyymmdd)
    if include_symbol_dir:
        out_dir = os.path.join(latency_root, exchange, symbol)
    else:
        out_dir = os.path.join(latency_root, exchange)
    _ensure_dir(out_dir)
    return os.path.join(out_dir, fname)


@dataclass
class Job:
    feeds_root: str
    latency_root: str
    exchange: str
    symbol: str
    date_dash: str
    date_ymd: str
    bucket_every: str
    mul_entry: float
    offset_entry: int
    mul_resp: float
    offset_resp: int
    include_symbol_dir: bool
    filename_template: str
    overwrite: bool


def _one_latency_job(job: Job) -> Tuple[str, str, Optional[str], List[str]]:
    logs: List[str] = []
    try:
        logs.append(f"[START] exch={job.exchange} sym={job.symbol} date={job.date_ymd} bucket={job.bucket_every}")
        feed_file = _feed_path(job.feeds_root, job.exchange, job.symbol, job.symbol+"_"+job.date_ymd)
        if not os.path.exists(feed_file):
            msg = f"Missing feed file: {feed_file}"
            logs.append(f"[MISS] {msg}")
            return (job.symbol, job.date_ymd, msg, logs)

        # Load structured events
        data = np.load(feed_file)["data"]
        df = pl.DataFrame(data)

        # Filter events with both EXCH_EVENT and LOCAL_EVENT bits set
        df = (
            df.filter(
                (pl.col("ev") & EXCH_EVENT == EXCH_EVENT) & (pl.col("ev") & LOCAL_EVENT == LOCAL_EVENT)
            )
            .with_columns(pl.col("local_ts").alias("ts"))
            .group_by_dynamic("ts", every=job.bucket_every)
            .agg(
                pl.col("exch_ts").last().alias("exch_ts"),
                pl.col("local_ts").last().alias("local_ts"),
            )
            .drop("ts")
        )

        # Convert to structured array with fields ['exch_ts', 'local_ts']
        g = df.to_struct()  # single struct column
        # Safer: build numpy structured explicitly to ensure fields/order
        np_data = df.select(["exch_ts", "local_ts"]).to_numpy(structured=True)

        # Allocate output latency array
        out_dtype = np.dtype([("req_ts", "i8"), ("exch_ts", "i8"), ("resp_ts", "i8"), ("_padding", "i8")])
        out_arr = np.zeros(len(np_data), dtype=out_dtype)

        # Compute order latency
        generate_order_latency_nb(
            np_data,
            out_arr,
            job.mul_entry,
            job.offset_entry,
            job.mul_resp,
            job.offset_resp,
        )

        # Output path
        out_path = _latency_out_path(
            job.latency_root,
            job.exchange,
            job.symbol,
            job.date_ymd,
            job.include_symbol_dir,
            job.filename_template,
        )
        if os.path.exists(out_path) and not job.overwrite:
            logs.append(f"[SKIP] exists: {out_path}")
            return (job.symbol, job.date_ymd, None, logs)

        np.savez_compressed(out_path, data=out_arr)
        logs.append(f"[OUT] {out_path}")
        return (job.symbol, job.date_ymd, None, logs)

    except Exception as e:
        logs.append(f"[EXCEPTION] {e}")
        return (job.symbol, job.date_ymd, f"{e}", logs)


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate order-latency npz from order feed npz using YAML and multiprocessing.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # IO roots
    feeds_root: str   = cfg.get("feeds_root", "/data/tmp/tardis_bn_hft")
    latency_root: str = cfg["latency_root"]

    # Universe
    exchange: str = cfg["exchange"]
    symbols: List[str] = cfg.get("symbols") or []
    symbol_single: Optional[str] = cfg.get("symbol")  # allow single symbol
    if not symbols and not symbol_single:
        raise ValueError("Provide either 'symbol' or 'symbols' in YAML.")
    if symbols and symbol_single:
        # de-dup if both provided
        symbols = sorted(set(symbols + [symbol_single]))
    elif not symbols:
        symbols = [symbol_single]

    # Dates
    dates_cfg: Optional[List[str]] = cfg.get("dates")
    date_from: Optional[str] = cfg.get("date_from")
    date_to: Optional[str] = cfg.get("date_to")
    date_pairs = _iter_dates(dates_cfg, date_from, date_to)

    # Group-by bucket (match sample default: '1000000i'; make configurable)
    bucket_every: str = cfg.get("bucket_every", "1000000i")  # tune if your timestamps are μs -> e.g., '1000000i' for 1s

    # Latency synthesis params
    mul_entry: float   = float(cfg.get("mul_entry", 1))
    mul_resp: float    = float(cfg.get("mul_resp", 1))
    offset_entry: int  = int(cfg.get("offset_entry", 0))
    offset_resp: int   = int(cfg.get("offset_resp", 0))

    # Output naming
    include_symbol_dir: bool = bool(cfg.get("include_symbol_dir", True))
    filename_template: str   = cfg.get("filename_template", "latency_{date}.npz")  # {date} -> YYYYMMDD
    overwrite: bool          = bool(cfg.get("overwrite", True))

    # Build jobs
    jobs: List[Job] = []
    for sym in symbols:
        for d_dash, d_ymd in date_pairs:
            jobs.append(
                Job(
                    feeds_root=feeds_root,
                    latency_root=latency_root,
                    exchange=exchange,
                    symbol=sym,
                    date_dash=d_dash,
                    date_ymd=d_ymd,
                    bucket_every=bucket_every,
                    mul_entry=mul_entry,
                    offset_entry=offset_entry,
                    mul_resp=mul_resp,
                    offset_resp=offset_resp,
                    include_symbol_dir=include_symbol_dir,
                    filename_template=filename_template,
                    overwrite=overwrite,
                )
            )

    print(f"[latency] exchange={exchange}")
    print(f"[latency] symbols={symbols}")
    print(f"[latency] dates={[d[1] for d in date_pairs]}")
    print(f"[latency] feeds_root={feeds_root}")
    print(f"[latency] latency_root={latency_root}")
    print(f"[latency] bucket_every={bucket_every} mul_entry={mul_entry} offset_entry={offset_entry} mul_resp={mul_resp} offset_resp={offset_resp}")
    print(f"[latency] filename_template={filename_template} include_symbol_dir={include_symbol_dir} overwrite={overwrite}")

    # Multiprocessing
    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context("fork")
    total = len(jobs)
    with ctx.Pool(processes=int(cfg.get("num_proc", max(1, mp.cpu_count() // 2)))) as pool:
        it = pool.imap_unordered(_one_latency_job, jobs, chunksize=1)
        for sym, d, err, logs in tqdm(it, total=total, desc="Generating order latency", unit="combo"):
            for line in logs:
                tqdm.write(line)
            if err is None:
                tqdm.write(f"[OK] {sym} {d}")
            else:
                tqdm.write(f"[FAIL] {sym} {d} -> {err}")


if __name__ == "__main__":
    main()
