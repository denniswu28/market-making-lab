#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tickers-json", required=True)
    ap.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        help="Explicit HftBacktest NPZ files used to validate the snapshot",
    )
    args = ap.parse_args()

    tj = json.load(open(args.tickers_json))
    ts = float(tj[args.symbol]["tick_size"])
    ls = float(tj[args.symbol]["lot_size"])
    print(ts, ls)

    hbt = ROIVectorMarketDepthBacktest(
        [
            BacktestAsset()
            .tick_size(ts)
            .lot_size(ls)
            .initial_snapshot(args.snapshot)
            .data(args.data_files)
            .roi_lb(0.0)
            .roi_ub(1000.0)
        ]
    )
    # hbt.elapse(100000000000)
    d = hbt.depth(0)
    print(
        f"best_bid={d.best_bid} best_ask={d.best_ask} bid_tick={d.best_bid_tick} ask_tick={d.best_ask_tick}"
    )

    # sanity
    assert d.best_ask == d.best_ask, "ask is NaN"
    assert d.best_ask > 0, "ask not positive"
    # bid can be empty on some markets, but for liquid futures you should have both:
    assert d.best_bid == d.best_bid, "bid is NaN"


if __name__ == "__main__":
    main()
