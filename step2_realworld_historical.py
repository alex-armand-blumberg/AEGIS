#!/usr/bin/env python3
"""
AEGIS — Step 2 (Real-world historical data)
------------------------------------------
Takes an event-level conflict dataset (e.g., UCDP GED), filters by country,
aggregates fatalities to daily totals, computes a rolling window sum, and
detects escalation starts using a persistence requirement.

Outputs:
- summary stats to stdout
- a PNG plot saved to disk (default: ukraine_escalation_plot.png)

Example:
  python step2_realworld_historical.py --csv GEDEvent_v25_1.csv --country "Ukraine"
"""

import argparse
import os
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_aggregate_csv(
    csv_path: str,
    country: str,
    date_col: str,
    fatalities_col: str,
    country_col: str,
    fatality_cap: float,
) -> pd.DataFrame:
    """
    Load event-level CSV, filter to a country, coerce types, cap extreme outliers,
    then aggregate to daily fatalities.

    Returns a DataFrame with columns:
      - date (datetime64[ns])
      - fatalities (float)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Validate columns
    for col in [country_col, date_col, fatalities_col]:
        if col not in df.columns:
            raise KeyError(
                f"Missing required column '{col}'. Available columns include: {list(df.columns)[:20]} ..."
            )

    # Filter to country (case-sensitive match as default; you can adjust later)
    sub = df[df[country_col] == country].copy()
    if len(sub) == 0:
        # Try case-insensitive fallback to be friendly
        sub = df[df[country_col].astype(str).str.lower() == str(country).lower()].copy()

    if len(sub) == 0:
        raise ValueError(
            f"No rows found for {country_col} == '{country}'. "
            f"Check spelling/case or inspect unique values in '{country_col}'."
        )

    # Parse date
    sub["date_tmp"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=["date_tmp"]).copy()

    # Coerce fatalities to numeric + fill missing
    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # Cap extreme outliers (prevents single-record estimates from blowing up rolling sums)
    if fatality_cap is not None and fatality_cap > 0:
        sub[fatalities_col] = sub[fatalities_col].clip(upper=fatality_cap)

    # Aggregate to daily totals
    sub["date"] = sub["date_tmp"].dt.floor("D")
    daily = (
        sub.groupby("date", as_index=False)[fatalities_col]
        .sum()
        .rename(columns={fatalities_col: "fatalities"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


def detect_escalations(
    daily: pd.DataFrame,
    window_days: int,
    threshold: float,
    persistence_days: int,
    cooldown_days: int,
) -> pd.DataFrame:
    """
    Add rolling sum and escalation flags to daily DataFrame.

    Columns added:
      - rolling (rolling window sum)
      - above_threshold
      - persistent (above_threshold for >= persistence_days consecutively)
      - escalation_start (first day of a persistent run, with cooldown gating)
    """
    df = daily.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling sum (min_periods makes early window partial; set to window_days if you want strict)
    df["rolling"] = df["fatalities"].rolling(window=window_days, min_periods=window_days).sum()

    df["above_threshold"] = df["rolling"] >= threshold

    # Persistence requirement: consecutive True count >= persistence_days
    # We compute run lengths using group id on changes
    grp = (df["above_threshold"] != df["above_threshold"].shift(1)).cumsum()
    run_len = df.groupby(grp)["above_threshold"].cumcount() + 1
    df["persistent"] = df["above_threshold"] & (run_len >= persistence_days)

    # Escalation starts: transition into persistent True
    df["escalation_start_raw"] = df["persistent"] & (~df["persistent"].shift(1).fillna(False))

    # Cooldown gating: after an escalation start, require cooldown_days of NOT persistent before another start
    if cooldown_days and cooldown_days > 0:
        last_start_idx = -10**9
        starts = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if df.loc[i, "escalation_start_raw"]:
                if i - last_start_idx >= cooldown_days:
                    starts[i] = True
                    last_start_idx = i
        df["escalation_start"] = starts
    else:
        df["escalation_start"] = df["escalation_start_raw"]

    return df


def make_plot(
    df: pd.DataFrame,
    country: str,
    window_days: int,
    threshold: float,
    out_png: str,
) -> None:
    """
    Plot rolling fatalities with threshold and markers at escalation starts.
    Saves to out_png.
    """
    # Drop NaNs in rolling for plotting nicer
    plot_df = df.dropna(subset=["rolling"]).copy()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["date"], plot_df["rolling"])

    # Threshold line
    plt.axhline(y=threshold, linestyle="--")

    # Mark escalation starts
    starts = plot_df[plot_df["escalation_start"]]
    if len(starts) > 0:
        plt.scatter(starts["date"], starts["rolling"], s=60)

    plt.title(f"AEGIS Escalation Detection — {country} (rolling={window_days}d, threshold={threshold})")
    plt.xlabel("Date")
    plt.ylabel("Rolling window fatalities")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS Step 2 — Real-world historical escalation detection")

    p.add_argument("--csv", required=True, help="Path to event-level CSV (e.g., GEDEvent_v25_1.csv)")
    p.add_argument("--country", required=True, help="Country value to filter on (matches --country-col)")
    p.add_argument("--country-col", default="country", help="Column name for country (default: country)")

    p.add_argument("--date-col", default="date_start", help="Column name for event date (default: date_start)")
    p.add_argument(
        "--fatalities-col",
        default="best",
        help="Column name for fatalities (default: best). For UCDP GED, 'best' is common.",
    )

    p.add_argument("--window-days", type=int, default=30, help="Rolling window size in days (default: 30)")
    p.add_argument("--threshold", type=float, default=25.0, help="Escalation threshold on rolling fatalities (default: 25)")

    p.add_argument(
        "--persistence-days",
        type=int,
        default=7,
        help="Require >= this many consecutive days above threshold before triggering (default: 7)",
    )

    p.add_argument(
        "--cooldown-days",
        type=int,
        default=30,
        help="Minimum spacing (in days) between escalation starts (default: 30)",
    )

    p.add_argument(
        "--fatality-cap",
        type=float,
        default=1000.0,
        help="Cap (winsorize) event fatalities to reduce extreme outlier impact (default: 1000). Set 0 to disable.",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Output PNG filename. Default: '<country>_escalation_plot.png' (lowercase, spaces->_).",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_png = args.out
    if not out_png:
        safe_country = str(args.country).strip().lower().replace(" ", "_")
        out_png = f"{safe_country}_escalation_plot.png"

    try:
        cap = None if (args.fatality_cap is None or args.fatality_cap <= 0) else args.fatality_cap

        daily = load_and_aggregate_csv(
            csv_path=args.csv,
            country=args.country,
            date_col=args.date_col,
            fatalities_col=args.fatalities_col,
            country_col=args.country_col,
            fatality_cap=cap,
        )

        df = detect_escalations(
            daily=daily,
            window_days=args.window_days,
            threshold=args.threshold,
            persistence_days=args.persistence_days,
            cooldown_days=args.cooldown_days,
        )

        # Summary stats
        above = int(df["above_threshold"].fillna(False).sum())
        persistent = int(df["persistent"].fillna(False).sum())
        starts = df[df["escalation_start"]].copy()

        print(f"Days above threshold: {above}")
        print(f"Days persistent: {persistent}")
        print(f"Escalation starts detected: {len(starts)}")

        if len(starts) > 0:
            print("\nEscalation start dates (first 20):")
            for _, r in starts.head(20).iterrows():
                roll = r["rolling"]
                d = pd.to_datetime(r["date"]).date()
                print(f" - {d} (rolling={roll:.1f})")

        make_plot(
            df=df,
            country=args.country,
            window_days=args.window_days,
            threshold=args.threshold,
            out_png=out_png,
        )
        print(f"\nSaved plot: {out_png}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
