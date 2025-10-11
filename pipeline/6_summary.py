from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
from matplotlib import pyplot as plt

def _read_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], unit="ns", errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.index = ts
    if "price" not in df.columns and "mid_price" in df.columns:
        df = df.rename(columns={"mid_price": "price"})
    for c in ("balance","position","price","fee"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="out/gridsearch_summary.csv")
    ap.add_argument("--usd-per-order", type=float, default=100.0)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    changed = 0

    for i, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        need = any(pd.isna(row.get(k)) for k in ("start_eq","end_eq","ret_abs","ret_pct"))
        if not need:
            continue
        csv_path = row.get("csv")
        if not (isinstance(csv_path, str) and os.path.exists(csv_path)):
            continue

        r = _read_result_csv(csv_path)
        idx = r.index
        b   = (r["balance"] if "balance" in r else pd.Series(0.0, index=idx)).astype(float).fillna(method="ffill").fillna(0.0)
        p   = r["price"].astype(float).fillna(method="ffill")
        pos = (r["position"] if "position" in r else pd.Series(0.0, index=idx)).astype(float).fillna(0.0)
        fee = (r["fee"] if "fee" in r else pd.Series(0.0, index=idx)).astype(float).fillna(0.0)

        equity = (b + pos * p - fee).dropna()
        if len(equity):
            start_eq = float(equity.iloc[0])
            end_eq   = float(equity.iloc[-1])
            ret_abs  = end_eq - start_eq
            ret_pct  = (ret_abs / start_eq * 100.0) if start_eq != 0 else np.nan
        else:
            start_eq = end_eq = ret_abs = ret_pct = np.nan

        # daily trade count approx
        try:
            mid_1d_last = p.resample("1D").last()
            notional_qty = pos.diff().abs().rolling("1D").sum().resample("1D").last()
            approx_trades = (notional_qty * mid_1d_last) / max(1e-9, args.usd_per_order)
            avg_tr = float(approx_trades.dropna().mean()) if len(approx_trades.dropna()) else np.nan
        except Exception:
            avg_tr = np.nan

        df.loc[i, ["start_eq","end_eq","ret_abs","ret_pct","approx_avg_daily_trades"]] = [
            start_eq, end_eq, ret_abs, ret_pct, avg_tr
        ]
        changed += 1

    out = args.summary.replace(".csv", "_repaired.csv")
    df.to_csv(out, index=False)
    print(f"[repair] fixed rows: {changed}, wrote {out}")

if __name__ == "__main__":
    main()
