#!/usr/bin/env python3
"""
AEGIS Step 2 — Real-world historical escalation detection (multi-threshold)

Goal: detect BOTH a lower-threshold escalation (e.g., 2014 conflict onset)
and a higher-threshold escalation (e.g., 2022 full-scale invasion) from the
same country time series.

Example (Colab / terminal):
  python step2_realworld_historical.py \
    --csv GEDEvent_v25_1.csv \
    --country "Ukraine" \
    --date-col date_start \
    --fatalities-col best \
    --thresholds 25,1000 \
    --window 30 \
    --persist-days 3 \
    --out ukraine_escalation_plot.png
"""

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_thresholds(s: str) -> List[float]:
    # Accept "25,1000" or "25 1000" etc.
    parts = re.split(r"[,\s]+", s.strip())
    vals = [float(p) for p in parts if p != ""]
    if not vals:
        raise ValueError("No thresholds provided.")
    return vals


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_and_aggregate_daily(
    csv_path: str,
    country: str,
    country_col: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Basic column checks
    for col in [country_col, date_col, fatalities_col]:
        if col not in df.columns:
            raise KeyError(
                f"Missing column '{col}'. Available columns include: "
                f"{', '.join(list(df.columns)[:30])} ..."
            )

    # Filter country
    sub = df[df[country_col].astype(str) == str(country)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for {country_col} == '{country}'")

    # Parse date + fatalities
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col]).copy()
    sub[fatalities_col] = coerce_numeric(sub[fatalities_col])

    # Aggregate to daily totals
    sub["date"] = sub[date_col].dt.floor("D")
    daily = (
        sub.groupby("date", as_index=False)[fatalities_col]
        .sum()
        .rename(columns={fatalities_col: "fatalities"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Fill missing days with 0 fatalities (important for rolling windows)
    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(all_days).fillna(0.0).rename_axis("date").reset_index()
    daily["fatalities"] = coerce_numeric(daily["fatalities"])

    return daily


def rolling_sum(df_daily: pd.DataFrame, window: int) -> pd.Series:
    # min_periods=window makes the first (window-1) days NaN; that’s OK.
    return df_daily["fatalities"].rolling(window=window, min_periods=window).sum()


def detect_starts(
    dates: pd.Series,
    rolling_vals: pd.Series,
    threshold: float,
    persist_days: int,
) -> pd.DataFrame:
    """
    A "start" occurs when we have persist_days consecutive days >= threshold,
    and the day before that run was not already in a persistent run.

    We report the start date as the FIRST day of that consecutive run.
    """
    above = (rolling_vals >= threshold).fillna(False)

    if persist_days <= 1:
        persistent = above
    else:
        # True when last persist_days are all True
        persistent = above.rolling(persist_days, min_periods=persist_days).sum() == persist_days
        persistent = persistent.fillna(False)

    # Start of a persistent regime
    start_trigger = persistent & (~persistent.shift(1, fill_value=False))

    # Convert trigger day to FIRST day in the persist run
    idx_trigger = np.where(start_trigger.to_numpy())[0]
    idx_start = idx_trigger - (persist_days - 1)
    idx_start = idx_start[idx_start >= 0]

    out = pd.DataFrame({
        "threshold": threshold,
        "start_date": dates.iloc[idx_start].to_numpy(),
        "rolling_at_start": rolling_vals.iloc[idx_start].to_numpy(),
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the CSV file")
    ap.add_argument("--country", required=True, help='Country filter (e.g., "Ukraine")')
    ap.add_argument("--country-col", default="country", help="Country column name")
    ap.add_argument("--date-col", default="date_start", help="Date column name")
    ap.add_argument("--fatalities-col", default="best", help="Fatalities column name")

    ap.add_argument("--window", type=int, default=30, help="Rolling window (days)")
    ap.add_argument("--thresholds", default="25,1000",
                    help="Comma/space-separated thresholds (e.g., '25,1000')")
    ap.add_argument("--persist-days", type=int, default=3,
                    help="How many consecutive days above threshold to confirm a start")

    ap.add_argument("--out", default=None, help="Output PNG filename")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI")

    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    thresholds_sorted = sorted(thresholds)

    # Load + daily aggregate
    df_daily = load_and_aggregate_daily(
        csv_path=args.csv,
        country=args.country,
        country_col=args.country_col,
        date_col=args.date_col,
        fatalities_col=args.fatalities_col,
    )

    df_daily["rolling"] = rolling_sum(df_daily, window=args.window)

    # Detect starts for each threshold
    starts_all = []
    for thr in thresholds_sorted:
        starts = detect_starts(
            dates=df_daily["date"],
            rolling_vals=df_daily["rolling"],
            threshold=thr,
            persist_days=args.persist_days,
        )
        starts_all.append(starts)

    starts_df = pd.concat(starts_all, ignore_index=True) if starts_all else pd.DataFrame()

    # Print summary
    print(f"Country: {args.country}")
    print(f"Rows (daily): {len(df_daily)}")
    print(f"Rolling window: {args.window} days")
    print(f"Persistence requirement: {args.persist_days} day(s)")
    print()

    for thr in thresholds_sorted:
        above_days = int(((df_daily["rolling"] >= thr).fillna(False)).sum())
        n_starts = int((starts_df["threshold"] == thr).sum()) if not starts_df.empty else 0
        print(f"Threshold {thr:g}:")
        print(f"  Days above threshold: {above_days}")
        print(f"  Escalation starts detected: {n_starts}")
        if n_starts > 0:
            show = starts_df[starts_df["threshold"] == thr].head(20)
            print("  Start dates (first 20):")
            for _, r in show.iterrows():
                d = pd.to_datetime(r["start_date"]).date()
                rv = r["rolling_at_start"]
                print(f"   - {d} (rolling={rv:.1f})")
        print()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_daily["date"], df_daily["rolling"])
    plt.title(
        f"AEGIS Escalation Detection — {args.country} "
        f"(rolling={args.window}d, thresholds={','.join(str(int(t)) if t.is_integer() else str(t) for t in thresholds_sorted)})"
    )
    plt.xlabel("Date")
    plt.ylabel("Rolling window fatalities")

    # Threshold lines + markers
    for thr in thresholds_sorted:
        plt.axhline(y=thr, linestyle="--")
        if not starts_df.empty:
            s = starts_df[starts_df["threshold"] == thr]
            if not s.empty:
                plt.scatter(s["start_date"], s["rolling_at_start"], s=60)

    plt.tight_layout()

    out = args.out
    if out is None:
        safe_country = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.country.strip())
        out = f"{safe_country.lower()}_escalation_plot.png"

    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot: {out}")

    # In scripts/CLI, we usually don't call plt.show(); in notebooks it may still display.
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
