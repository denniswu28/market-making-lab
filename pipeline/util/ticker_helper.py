# pipeline/util/ticker_helper.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, Iterable, Optional
import os
import requests

BINANCE_24HR = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_EXINFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def _build_session(proxy_url: Optional[str]) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "statmm-tickers/1.0"})
    chosen = proxy_url or os.environ.get("YF_PROXY") or os.environ.get("ALL_PROXY")
    if chosen:
        s.proxies.update({"http": chosen, "https": chosen})
    return s


def _get_json(session: requests.Session, url: str):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_binance_futures_info(
    symbols: Optional[Iterable[str]] = None,
    proxy: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Returns a dict[symbol] -> {
        weighted_avg_price, quote_volume, onboard_date, tick_size, lot_size, min_qty
    }
    If `symbols` is None, returns the full universe (you can filter yourself).
    """
    sess = _build_session(proxy)

    tickers = _get_json(sess, BINANCE_24HR)
    exch_info = _get_json(sess, BINANCE_EXINFO)

    # Build base dict from 24hr
    info = {}
    for t in tickers:
        sym = t["symbol"]
        info[sym] = {
            "weighted_avg_price": t["weightedAvgPrice"],
            "quote_volume": t["quoteVolume"],
        }

    # Enrich with exchangeInfo
    for s in exch_info.get("symbols", []):
        sym = s["symbol"]
        if sym not in info:
            # create if not present, so we can still return requested symbols
            info[sym] = {}
        info[sym]["onboard_date"] = datetime.fromtimestamp(s["onboardDate"] / 1000).strftime("%Y%m%d")
        for f in s["filters"]:
            ft = f["filterType"]
            if ft == "PRICE_FILTER":
                info[sym]["tick_size"] = f["tickSize"]
            elif ft == "LOT_SIZE":
                info[sym]["lot_size"] = f["stepSize"]
                info[sym]["min_qty"] = f["minQty"]
            elif ft == "MARKET_LOT_SIZE":
                # sanity
                if ("lot_size" in info[sym] and info[sym]["lot_size"] != f["stepSize"]) or (
                    "min_qty" in info[sym] and info[sym]["min_qty"] != f["minQty"]
                ):
                    raise ValueError(f"{sym}: MARKET_LOT_SIZE != LOT_SIZE")

    if symbols is not None:
        symbols = list(symbols)
        info = {s: info[s] for s in symbols if s in info}

    return info
