# pipeline/0_ticker.py
from __future__ import annotations
import argparse
import json
import pprint
import sys
from util.ticker_helper import fetch_binance_futures_info


def main():
    p = argparse.ArgumentParser(description="Fetch Binance Futures tickers.")
    p.add_argument(
        "--proxy", default="socks5h://127.0.0.1:1080", help="socks5h://127.0.0.1:1080"
    )
    # p.add_argument("--symbols", nargs="*", help="Specific symbols to fetch (space-separated).")
    p.add_argument(
        "--num-tickers",
        type=int,
        default=50,
        help="Top-N alts by quote volume (when --symbols is omitted)",
    )
    p.add_argument("--out", default="pipeline/tickers_new.json", help="Output JSON file")
    args = p.parse_args()

    args.symbols = ["SOLUSDC", "SUIUSDC", "BNBUSDC", "WIFUSDC", "WLDUSDC", "1000SHIBUSDC", "1000PEPEUSDC", "ORDIUSDC", "UNIUSDC", "NEOUSDC", "KAITOUSDC"]

    try:
        info = fetch_binance_futures_info(args.symbols, proxy=args.proxy)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if not args.symbols:
        # Filter to top-N alts by quote volume
        alts = {
            s: d
            for s, d in info.items()
            if not s.startswith("BTCUSD") and not s.startswith("ETHUSD")
        }
        info = dict(
            sorted(
                alts.items(), key=lambda kv: float(kv[1]["quote_volume"]), reverse=True
            )[: args.num_tickers]
        )

    pprint.pprint(info, compact=True)
    with open(args.out, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
