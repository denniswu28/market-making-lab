from __future__ import annotations
import argparse
import os
import glob
from typing import List

import pandas as pd
import polars as pl
import yaml
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from hftbacktest.stats import LinearAssetRecord


def _find_csvs(out_path: str, symbols: List[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sym in symbols:
        patt = os.path.join(out_path, f"{sym}*.csv")
        matches = sorted(glob.glob(patt))
        if matches:
            out[sym] = matches[0]
    return out


def _load_record(csv_path: str) -> LinearAssetRecord:
    df = pd.read_csv(csv_path)

    # timestamps → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # normalize price column name
    if "price" not in df.columns and "mid_price" in df.columns:
        df = df.rename(columns={"mid_price": "price"})

    # confirm required columns exist
    for col in ("balance", "position", "price", "fee"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {csv_path}")

    # sort & convert to Polars
    df = df.set_index("timestamp").sort_index()
    pl_df = pl.from_pandas(
        df.reset_index()[["timestamp", "balance", "position", "price", "fee"]]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime))

    rec = LinearAssetRecord(pl_df)
    rec.prepare()  # compute equity_wo_fee, trading_value_ if missing
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="pipeline/backtest_config.yaml")
    ap.add_argument("--plot-dir", default=None, help="Default: {out_path}/plots_stats")
    ap.add_argument("--stats-dir", default=None, help="Default: {out_path}/stats")
    ap.add_argument("--freq", default="5m", help="Resample frequency, e.g. 5m, 10s")
    ap.add_argument(
        "--partition",
        choices=["none", "monthly", "daily", "hourly"],
        default="monthly",
        help="How to split metrics (default monthly).",
    )
    ap.add_argument(
        "--price-as-ret",
        action="store_true",
        help="In plots, show price panel as cumulative returns.",
    )
    ap.add_argument(
        "--book-size",
        type=float,
        default=None,
        help="If set, equity is plotted as % of this notional (also used by price-as-ret).",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = cfg["out_path"]
    symbols = cfg["symbols"]

    plot_dir = args.plot_dir or os.path.join(out_path, "plots_stats")
    stats_dir = args.stats_dir or os.path.join(out_path, "stats")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    csv_map = _find_csvs(out_path, symbols)
    if not csv_map:
        print(f"[stats] No result CSVs found in {out_path} for {symbols}")
        return

    summary_rows = []
    for sym, csv_path in csv_map.items():
        try:
            rec = _load_record(csv_path)

            # resample + partition (use methods, not split=)
            rec = rec.resample(args.freq)
            if args.partition == "monthly":
                rec = rec.monthly()
            elif args.partition == "daily":
                rec = rec.daily()
            elif args.partition == "hourly":
                # available in utils; Record has hourly() wired
                rec = rec.hourly()  # type: ignore[attr-defined]
            # else: no partition

            # pass book_size via kwargs so plot() can use it
            st = rec.stats(book_size=args.book_size)

            # metrics table
            summary_pl = st.summary()
            summary_pd = summary_pl.to_pandas()

            # tag rows: earlier rows are partitions, last row is entire
            kind = ["split"] * len(summary_pd)
            if len(kind):
                kind[-1] = "entire"
                summary_pd.insert(0, "kind", kind)

            sym_stats_csv = os.path.join(stats_dir, f"{sym}_metrics.csv")
            summary_pd.to_csv(sym_stats_csv, index=False)

            # figure
            fig = st.plot_matplotlib(price_as_ret=args.price_as_ret)
            for ax in fig.get_axes():
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.tight_layout()
            fig_path = os.path.join(plot_dir, f"{sym}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # headline numbers from entire row (last row)
            entire = summary_pd.tail(1)
            row = {
                "symbol": sym,
                "csv": csv_path,
                "plot": fig_path,
                "stats_csv": sym_stats_csv,
            }
            if not entire.empty:
                row.update(entire.iloc[0].to_dict())
            summary_rows.append(row)

            print(f"[stats] OK: {sym} -> {fig_path}, {sym_stats_csv}")

        except Exception as e:
            print(f"[stats] FAIL: {sym}: {e}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(stats_dir, "summary.csv"), index=False
        )
        print(f"[stats] Wrote {os.path.join(stats_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
