#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yaml

# Optional: fetch tick/lot via your helper (SOCKS5 supported)
try:
    from util.ticker_helper import fetch_binance_futures_info
except Exception:  # pragma: no cover
    fetch_binance_futures_info = None  # not fatal

# Prefer the built-in helper; otherwise fall back to a local impl
try:
    from hftbacktest.data.utils.snapshot import create_last_snapshot as hbt_create_last_snapshot  # type: ignore
except Exception:  # pragma: no cover
    hbt_create_last_snapshot = None


# --- local fallback of create_last_snapshot (matches snippet you posted) ---
def _fallback_create_last_snapshot(
    data: List[str],
    tick_size: float,
    lot_size: float,
    initial_snapshot: Optional[str] = None,
    output_snapshot_filename: Optional[str] = None,
):
    import numpy as np
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest  # type: ignore

    asset = BacktestAsset().data(data).tick_size(tick_size).lot_size(lot_size)
    if initial_snapshot is not None:
        asset.initial_snapshot(initial_snapshot)

    hbt = HashMapMarketDepthBacktest([asset])

    # Move to end
    if hbt._goto_end() not in [0, 1]:  # noqa: SLF001 (matches upstream)
        raise RuntimeError("goto_end failed")

    depth = hbt.depth(0)
    snapshot = depth.snapshot()
    snapshot_copied = snapshot.copy()
    depth.snapshot_free(snapshot)

    if output_snapshot_filename is not None:
        np.savez_compressed(output_snapshot_filename, data=snapshot_copied)

    return snapshot_copied


def create_last_snapshot(
    data: List[str],
    tick_size: float,
    lot_size: float,
    initial_snapshot: Optional[str],
    output_snapshot_filename: Optional[str],
):
    if hbt_create_last_snapshot is not None:
        return hbt_create_last_snapshot(
            data=data,
            tick_size=tick_size,
            lot_size=lot_size,
            initial_snapshot=initial_snapshot,
            output_snapshot_filename=output_snapshot_filename,
        )
    print("failed")
    return _fallback_create_last_snapshot(
        data=data,
        tick_size=tick_size,
        lot_size=lot_size,
        initial_snapshot=initial_snapshot,
        output_snapshot_filename=output_snapshot_filename,
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_local_timestamp(data_files: List[str]) -> int:
    import numpy as np

    latest: Optional[int] = None
    for path in data_files:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(f"market-data file is missing the data array: {path}")
            events = archive["data"]
        if events.ndim != 1 or events.size == 0 or "local_ts" not in (events.dtype.names or ()):
            raise ValueError(f"market-data file has no usable local timestamps: {path}")
        file_latest = int(events["local_ts"].max())
        latest = file_latest if latest is None else max(latest, file_latest)
    if latest is None:
        raise ValueError("cannot write snapshot metadata without source events")
    return latest


def _write_snapshot_manifest(snapshot_path: str, data_files: List[str]) -> None:
    manifest = {
        "schema_version": 1,
        "as_of_ns": _latest_local_timestamp(data_files),
        "snapshot_sha256": _sha256_file(snapshot_path),
        "source": "generated-eod",
    }
    manifest_path = f"{snapshot_path}.manifest.json"
    temporary = f"{manifest_path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, manifest_path)


# ----------------------------- helpers -----------------------------
def _norm_date(d: str) -> Tuple[str, str]:
    s = str(d).strip().replace("/", "-")
    if "-" in s:
        yyyy, mm, dd = s.split("-")
        return (
            f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}",
            f"{yyyy}{mm.zfill(2)}{dd.zfill(2)}",
        )
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}", s
    raise ValueError(f"Bad date: {d}")


def _iter_dates(
    dates: Optional[List[str]], date_from: Optional[int], date_to: Optional[int]
) -> List[str]:
    if dates:
        return [_norm_date(d)[1] for d in dates]
    if not (date_from and date_to):
        raise ValueError("Provide 'dates' OR ('date_from' & 'date_to').")
    s = datetime.strptime(str(date_from), "%Y%m%d")
    e = datetime.strptime(str(date_to), "%Y%m%d")
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _files_for(base_root: str, exchange: str, symbol: str, yyyymmdd: str) -> str:
    # Convert output layout from your pipeline:
    # {base_root}/data/{exchange}/{symbol}/{symbol}_{YYYYMMDD}.npz
    return os.path.join(base_root, "data", exchange, symbol, f"{symbol}_{yyyymmdd}.npz")


def _guess_prev_sod_path(
    output_root: str, exchange: str, symbol: str, first_date: str, suffix: str
) -> str:
    d0 = datetime.strptime(first_date, "%Y%m%d") - timedelta(days=1)
    prev = d0.strftime("%Y%m%d")
    return os.path.join(output_root, exchange, symbol, f"{symbol}_{prev}_{suffix}.npz")


def _load_tickers_json(path: Optional[str]) -> Dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Create SOD/EOD snapshots from converted NPZ feeds."
    )
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--proxy",
        default="socks5h://127.0.0.1:1080",
        help="SOCKS/HTTP proxy for REST (if fetching sizes)",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_root = cfg["base_root"]  # where data/latency live (same as backtest config)
    exchange = cfg["exchange"]
    symbols = cfg["symbols"]
    dates = _iter_dates(cfg.get("dates"), cfg.get("date_from"), cfg.get("date_to"))
    per_day = bool(
        cfg.get("per_day", False)
    )  # if True, emit one snapshot per day (for next day's SOD)
    strict = bool(
        cfg.get("strict", True)
    )  # if True, fail if any requested data file missing
    suffix = str(cfg.get("suffix", "SOD"))  # filename suffix, e.g. SOD
    output_root = cfg.get("output_root", os.path.join(base_root, "data"))
    tickers_json = cfg.get("tickers_json")  # optional path to tickers.json
    ts_map = cfg.get("tick_size", {}) or {}
    ls_map = cfg.get("lot_size", {}) or {}

    _ensure_dir(output_root)
    # tickers = _load_tickers_json(tickers_json)

    # Optionally back-fill tick/lot using REST
    missing_syms = [s for s in symbols if (s not in ts_map or s not in ls_map)]
    if missing_syms and fetch_binance_futures_info:
        fetched = fetch_binance_futures_info(missing_syms, proxy=args.proxy)
        for s in missing_syms:
            if s in fetched:
                ts_map.setdefault(s, float(fetched[s]["tick_size"]))
                ls_map.setdefault(s, float(fetched[s]["lot_size"]))

    # Final sanity for tick/lot
    still_missing = [s for s in symbols if (s not in ts_map or s not in ls_map)]
    if still_missing:
        raise SystemExit(
            f"[snapshot] Missing tick/lot for: {still_missing}. "
            f"Provide in config.tick_size/lot_size or tickers_json."
        )

    for sym in symbols:
        # Build list of data files
        data_files = []
        for d in dates:
            p = _files_for(base_root, exchange, sym, d)
            if os.path.exists(p):
                data_files.append(p)
            elif strict:
                raise SystemExit(f"[snapshot] Missing data file: {p}")
        if not data_files:
            print(f"[snapshot] No data for {sym}; skipping.")
            continue

        # Decide initial SOD
        initial_snapshot = None
        # if initial_snapshot == "auto-prev":
        #     guess = _guess_prev_sod_path(output_root, exchange, sym, dates[0], suffix)
        #     if os.path.exists(guess):
        #         initial_snapshot = guess
        #         print(f"[snapshot] {sym}: using initial_snapshot {guess}")
        #     else:
        #         print(f"[snapshot] {sym}: no previous SOD found at {guess}; starting without initial snapshot.")
        #         initial_snapshot = None

        # Emit
        out_dir = os.path.join(output_root, exchange, sym)
        _ensure_dir(out_dir)

        if per_day:
            # iteratively roll per day; each EOD becomes next SOD
            cur_init = initial_snapshot
            for d in dates:
                files_upto_d = [f for f in data_files if f.endswith(f"_{d}.npz")]
                if not files_upto_d:
                    # should not happen because we built from existing files, but keep safe
                    print(f"[snapshot] {sym} {d}: no feed; skip")
                    continue
                out_path = os.path.join(out_dir, f"{sym}_{d}_{suffix}.npz")
                print(f"[snapshot] {sym} {d}: -> {out_path}")
                create_last_snapshot(
                    data=[files_upto_d[0]],  # only that day's file for per-day EOD
                    tick_size=float(ts_map[sym]),
                    lot_size=float(ls_map[sym]),
                    initial_snapshot=cur_init,
                    output_snapshot_filename=out_path,
                )
                _write_snapshot_manifest(out_path, files_upto_d)
                # next day starts from this snapshot
                cur_init = out_path
        else:
            # one combined snapshot at the end of the range (useful as SOD for the next period)
            last_date = dates[-1]
            out_path = os.path.join(out_dir, f"{sym}_{last_date}_{suffix}.npz")
            print(f"[snapshot] {sym} {dates[0]}..{last_date}: -> {out_path}")
            create_last_snapshot(
                data=data_files,
                tick_size=float(ts_map[sym]),
                lot_size=float(ls_map[sym]),
                initial_snapshot=initial_snapshot,
                output_snapshot_filename=out_path,
            )
            _write_snapshot_manifest(out_path, data_files)

    print("[snapshot] Done.")


if __name__ == "__main__":
    main()
