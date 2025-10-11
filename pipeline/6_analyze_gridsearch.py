# pipeline/6_analyze_gridsearch.py
from __future__ import annotations
import argparse
import itertools
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _detect_numeric_param_cols(df: pd.DataFrame) -> List[str]:
    # Anything numeric that isn't an outcome/meta column we treat as a parameter
    exclude = {
        "name", "symbol", "status",
        "ret_abs", "ret_pct", "start_eq", "end_eq",
        "approx_avg_daily_trades",
        "csv", "plot",
        # transform categorical; we’ll facet by this (not numeric)
        "transform_kind"
    }
    num_cols = [c for c in df.columns
                if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    # keep only those that actually vary
    return [c for c in num_cols if df[c].nunique(dropna=True) > 1]


def _topn_per_symbol(df: pd.DataFrame, score: str, topn: int) -> pd.DataFrame:
    rows = []
    for sym, g in df.groupby("symbol"):
        gg = g.sort_values(score, ascending=False)
        top = gg.head(topn).copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        rows.append(top)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


def _plot_pair_heat_or_scatter(sym: str,
                               g: pd.DataFrame,
                               x: str, y: str, score: str,
                               outdir: str) -> str:
    """
    Try a heatmap via pivot; if the grid is ragged or too sparse, fall back to a scatter
    with color = performance.
    """
    # Make sure we only use complete rows
    gg = g[[x, y, score]]
    print(gg)
    gg = gg.dropna()
    if gg.empty:
        return ""

    # Attempt pivot (heatmap) when both axes are reasonably gridded
    x_vals = np.sort(gg[x].unique())
    y_vals = np.sort(gg[y].unique())
    can_heatmap = (len(x_vals) >= 2 and len(y_vals) >= 2
                   and len(gg) >= len(x_vals) * len(y_vals) * 0.5)  # >=50% coverage

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    if can_heatmap:
        pv = gg.pivot_table(index=y, columns=x, values=score, aggfunc="mean")
        # Ensure ordered axes
        pv = pv.reindex(index=np.sort(pv.index.values), columns=np.sort(pv.columns.values))
        im = ax.imshow(pv.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_yticks(range(len(pv.index)))
        ax.set_xticklabels([f"{v:g}" for v in pv.columns], rotation=45, ha="right")
        ax.set_yticklabels([f"{v:g}" for v in pv.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{sym} | {score} heatmap: {y} vs {x}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(score)
        mode = "heatmap"
    else:
        sc = ax.scatter(gg[x], gg[y], c=gg[score], s=36)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{sym} | {score} scatter: {y} vs {x}")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(score)
        mode = "scatter"

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{sym}_{y}_vs_{x}_{mode}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _per_param_correlation(sym: str, g: pd.DataFrame, score: str,
                           numeric_params: List[str]) -> pd.DataFrame:
    """
    Spearman rank correlation of each numeric parameter vs performance metric.
    Gives a monotonic-trend hint (+ increases with param; - decreases).
    """
    cols = [c for c in numeric_params if c in g.columns]
    if not cols or score not in g.columns:
        return pd.DataFrame(columns=["symbol", "param", "spearman_rho"])
    corr = g[cols + [score]].corr(method="spearman")[score].drop(labels=[score])
    return pd.DataFrame({
        "symbol": sym,
        "param": corr.index,
        "spearman_rho": corr.values
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default="out/gridsearch_summary.csv",
                    help="Path to gridsearch_summary.csv produced by 4_gridsearch.py")
    ap.add_argument("--score", default="ret_pct", choices=["ret_abs", "ret_pct"],
                    help="Metric to rank/plot.")
    ap.add_argument("--topn", type=int, default=5,
                    help="Top N parameter sets per symbol.")
    ap.add_argument("--out-dir", default=None,
                    help="Directory to write reports/plots (default: same folder as CSV).")
    ap.add_argument("--facets-by-transform", action="store_true",
                    help="Facet trend plots by transform_kind (if it varies).")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    base_dir = os.path.dirname(os.path.abspath(args.summary_csv))
    out_dir = args.out_dir or os.path.join(base_dir, "analysis")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Keep successful runs only and coerce likely numeric columns
    df = df[df["status"] == "ok"].copy()
    numeric_like = [
        "ret_abs", "ret_pct", "start_eq", "end_eq",
        "rel_half_spread", "relative_grid_interval", "grid_num",
        "skew", "order_qty", "max_position",
        "transform_window", "transform_ema_alpha",
        "approx_avg_daily_trades",
    ]
    df = _coerce_numeric(df, [c for c in numeric_like if c in df.columns])

    # Per-symbol: Top N
    topn = _topn_per_symbol(df, args.score, args.topn)
    topn_path = os.path.join(out_dir, f"top{args.topn}_per_symbol_{args.score}.csv")
    topn.to_csv(topn_path, index=False)
    print(f"[report] wrote {topn_path}")

    # Per-symbol: parameter trend plots and correlations
    corr_rows = []
    for sym, g in df.groupby("symbol"):
        numeric_params = _detect_numeric_param_cols(g)

        # Optional facet by transform kind
        facets = [("", g)]
        if args.facets_by_transform and "transform_kind" in g.columns and g["transform_kind"].nunique() > 1:
            facets = [(f"_xf_{xf}", gg) for xf, gg in g.groupby("transform_kind")]

        # Correlations (Spearman)
        corr_rows.append(_per_param_correlation(sym, g, args.score, numeric_params))

        # All numeric param pairs
        for suffix, gg in facets:
            for x, y in itertools.combinations(numeric_params, 2):
                outpath = _plot_pair_heat_or_scatter(
                    sym, gg, x, y, args.score,
                    outdir=os.path.join(plot_dir, f"{sym}{suffix}")
                )
                if outpath:
                    print(f"[plot] {outpath}")

    if corr_rows:
        corr_df = pd.concat(corr_rows, ignore_index=True)
        corr_path = os.path.join(out_dir, f"param_spearman_{args.score}.csv")
        corr_df.to_csv(corr_path, index=False)
        print(f"[report] wrote {corr_path}")

    print("[done]")


if __name__ == "__main__":
    main()
