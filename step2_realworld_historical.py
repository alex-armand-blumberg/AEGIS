#!/usr/bin/env python3
"""
AEGIS — Step 2: Real-world historical escalation detection

Reads an events CSV (ex: UCDP GED), filters by country, aggregates to daily fatalities,
computes rolling 30-day fatalities, and detects "escalation starts" using:
- threshold on rolling 30-day total
- persistence requirement: must stay above threshold for N consecutive days

Outputs:
- Printed summary stats
- A PNG plot saved to disk (default: ukraine_escalation_plot.png)

Example (UCDP GED):
  python step2_realworld_historical.py \
    --csv GEDEvent_v25_1.csv \
    --country "Ukraine" \
    --country-col country \
    --date-col date_start \
    --fatalities-col best \
    --threshold 25 \
    --window 30 \
    --persistence-days 3 \
    --out-png ukraine_escalation_plot.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def load_and_aggregate_csv(
    csv_path: str,
    country: str,
    country_col: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
    """
    Load events CSV, filter by country, and aggregate to daily fatalities.

    Returns a DataFrame with columns:
      - date (datetime64[ns])
      - fatalities (float)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read only needed columns if possible (faster + less RAM)
    usecols = None
    try:
        # This will error if a column doesn't exist; we'll fallback to full read
        usecols = [country_col, date_col, fatalities_col]
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)

    # Basic column existence checks
    for col in [country_col, date_col, fatalities_col]:
        if col not in df.columns:
            raise KeyError(
                f"Missing required column '{col}'. "
                f"Available columns include: {list(df.columns)[:50]} ..."
            )

    # Filter to the country
    sub = df[df[country_col].astype(str).str.strip() == str(country).strip()].copy()
    if sub.empty:
        # Try a case-insensitive fallback
        sub = df[df[country_col].astype(str).str.lower().str.strip() == str(country).lower().strip()].copy()

    if sub.empty:
        raise ValueError(
            f"No rows matched country='{country}' in column '{country_col}'. "
            f"Try checking spelling/casing or use a different --country-col."
        )

    # ---- KEY FIX: create a real date_day column, then group by it ----
    sub["date_tmp"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=["date_tmp"]).copy()

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # This must be a real column (prevents the pandas FutureWarning + missing key)
    sub["date_day"] = sub["date_tmp"].dt.floor("D")

    sub_daily = (
        sub.groupby("date_day", as_index=False)[fatalities_col]
           .sum()
           .rename(columns={"date_day": "date", fatalities_col: "fatalities"})
    )

    sub_daily["date"] = pd.to_datetime(sub_daily["date"])
    sub_daily = sub_daily.sort_values("date").reset_index(drop=True)

    return sub_daily


def detect_escalations(
    df_daily: pd.DataFrame,
    window: int,
    threshold: float,
    persistence_days: int,
) -> pd.DataFrame:
    """
    Given daily fatalities, compute rolling window total and detect escalation starts.
    """
    df = df_daily.copy()

    df["rolling_window"] = df["fatalities"].rolling(window=window, min_periods=window).sum()
    df["above_threshold"] = df["rolling_window"] >= threshold

    # Persistence: require above-threshold for N consecutive days
    if persistence_days <= 1:
        df["persistent"] = df["above_threshold"].fillna(False)
    else:
        # rolling over boolean -> count Trues in the last N days
        df["persistent"] = (
            df["above_threshold"]
              .rolling(window=persistence_days, min_periods=persistence_days)
              .sum()
              .fillna(0)
              .ge(persistence_days)
        )

    # Escalation start: first day persistent becomes True
    df["escalation_start"] = df["persistent"] & (~df["persistent"].shift(1).fillna(False))

    return df


def make_plot(
    df: pd.DataFrame,
    threshold: float,
    out_png: str,
    title: str,
):
    """
    Save plot to out_png and (optionally) show it.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["rolling_window"], label="Rolling window fatalities")
    plt.axhline(y=threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold}")

    # Mark escalation starts
    starts = df[df["escalation_start"]]
    if not starts.empty:
        plt.scatter(starts["date"], starts["rolling_window"], s=80, label="Escalation starts")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling window fatalities")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV file (in Colab: uploaded filename)")
    p.add_argument("--country", required=True, help="Country to filter on (exact match preferred)")
    p.add_argument("--country-col", default="country", help="Column name for country (default: country)")
    p.add_argument("--date-col", default="date_start", help="Column name for event date (default: date_start)")
    p.add_argument("--fatalities-col", default="best", help="Column name for fatalities (default: best)")

    p.add_argument("--window", type=int, default=30, help="Rolling window length in days (default: 30)")
    p.add_argument("--threshold", type=float, default=25, help="Escalation threshold on rolling total (default: 25)")
    p.add_argument("--persistence-days", type=int, default=3, help="Consecutive days above threshold required (default: 3)")

    p.add_argument("--out-png", default="ukraine_escalation_plot.png", help="Output PNG filename")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        df_daily = load_and_aggregate_csv(
            csv_path=args.csv,
            country=args.country,
            country_col=args.country_col,
            date_col=args.date_col,
            fatalities_col=args.fatalities_col,
        )

        df = detect_escalations(
            df_daily=df_daily,
            window=args.window,
            threshold=args.threshold,
            persistence_days=args.persistence_days,
        )

        # Summary stats
        days_above = int(df["above_threshold"].fillna(False).sum())
        days_persistent = int(df["persistent"].fillna(False).sum())
        n_starts = int(df["escalation_start"].fillna(False).sum())

        print(f"Days above threshold: {days_above}")
        print(f"Days persistent: {days_persistent}")
        print(f"Escalation starts detected: {n_starts}")

        # Print the actual start dates (first 20)
        starts = df.loc[df["escalation_start"], ["date", "rolling_window"]].head(20)
        if starts.empty:
            print("Escalation start dates: none")
        else:
            print("\nEscalation start dates (first 20):")
            for _, row in starts.iterrows():
                d = row["date"].strftime("%Y-%m-%d")
                rw = float(row["rolling_window"])
                print(f"  - {d} (rolling={rw:.1f})")

        title = f"AEGIS Escalation Detection — {args.country} (rolling={args.window}d, threshold={args.threshold})"
        make_plot(df, threshold=args.threshold, out_png=args.out_png, title=title)

        print(f"\nSaved plot: {args.out_png}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
