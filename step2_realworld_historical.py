import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_escalations(
    df_daily: pd.DataFrame,
    date_col: str,
    fatalities_col: str,
    threshold: float,
    window_days: int,
    persistence_days: int,
) -> pd.DataFrame:
    """
    df_daily must already be aggregated to one row per day.
    Required columns: date_col (datetime), fatalities_col (numeric)
    """
    df = df_daily.copy()
    df = df.sort_values(date_col)

    # rolling window (sum over last `window_days`)
    df["rolling_sum"] = (
        df[fatalities_col]
        .rolling(window=window_days, min_periods=window_days)
        .sum()
    )

    df["above_threshold"] = df["rolling_sum"] >= threshold

    # consecutive-run length of above-threshold days
    # (counts how many days in a row we've been above threshold)
    run_id = (df["above_threshold"] != df["above_threshold"].shift(1)).cumsum()
    df["above_run"] = df["above_threshold"].astype(int).groupby(run_id).cumsum()

    df["persistent"] = df["above_threshold"] & (df["above_run"] >= persistence_days)

    # escalation start = first persistent day after not persistent
    df["escalation_start"] = df["persistent"] & (~df["persistent"].shift(1).fillna(False))

    return df


def load_and_aggregate_csv(
    csv_path: str,
    country: str,
    date_col: str,
    fatalities_col: str,
    country_col: str,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Loads a large CSV safely by reading in chunks, filtering to country,
    and aggregating fatalities per day.
    """
    # Only load the columns we need (MUCH faster for a 250MB file)
    usecols = [date_col, fatalities_col, country_col]

    daily_parts = []
    found_any = False

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        # filter country (exact match)
        sub = chunk[chunk[country_col] == country]
        if sub.empty:
            continue

        found_any = True

        # parse date, coerce fatalities
        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

        sub = sub.dropna(subset=[date_col])

        # aggregate to daily totals
        sub_daily = (
            sub.groupby(sub[date_col].dt.date, as_index=False)[fatalities_col]
            .sum()
            .rename(columns={date_col: "date_tmp"})
        )
        sub_daily["date_tmp"] = pd.to_datetime(sub_daily["date_tmp"])
        daily_parts.append(sub_daily)

    if not found_any:
        return pd.DataFrame(columns=["date", "fatalities"])

    df_daily = pd.concat(daily_parts, ignore_index=True)

    # combine days across chunks
    df_daily = (
        df_daily.groupby("date_tmp", as_index=False)[fatalities_col]
        .sum()
        .rename(columns={"date_tmp": "date", fatalities_col: "fatalities"})
    )

    return df_daily


def main():
    parser = argparse.ArgumentParser(description="AEGIS: Escalation detection from event CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--country", required=True, help="Country name exactly as in the CSV")
    parser.add_argument("--date-col", default="date_start", help="Date column name in CSV")
    parser.add_argument("--fatalities-col", default="best", help="Fatalities column name in CSV")
    parser.add_argument("--country-col", default="country", help="Country column name in CSV")

    parser.add_argument("--threshold", type=float, default=25, help="Rolling-window fatalities threshold")
    parser.add_argument("--window-days", type=int, default=30, help="Rolling window length in days")
    parser.add_argument("--persistence-days", type=int, default=3, help="Days above threshold required (persistence)")

    parser.add_argument("--out-png", default=None, help="Output PNG filename (optional)")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # load + aggregate
    df_daily = load_and_aggregate_csv(
        csv_path=csv_path,
        country=args.country,
        date_col=args.date_col,
        fatalities_col=args.fatalities_col,
        country_col=args.country_col,
    )

    if df_daily.empty:
        print(f"No rows found for country='{args.country}'.")
        print("This usually means:")
        print("  (1) your CSV is already filtered to a different country, OR")
        print("  (2) the country name doesn't match exactly what's in the 'country' column.")
        return

    # rename to standard columns for compute step
    df_daily = df_daily.rename(columns={"date": "date", "fatalities": "fatalities"})

    df_out = compute_escalations(
        df_daily=df_daily,
        date_col="date",
        fatalities_col="fatalities",
        threshold=args.threshold,
        window_days=args.window_days,
        persistence_days=args.persistence_days,
    )

    # summary
    days_above = int(df_out["above_threshold"].sum())
    days_persistent = int(df_out["persistent"].sum())
    starts = int(df_out["escalation_start"].sum())

    print("Country:", args.country)
    print("Days above threshold:", days_above)
    print("Days persistent:", days_persistent)
    print("Escalation starts detected:", starts)

    # plot
    out_png = args.out_png or f"{args.country.lower().replace(' ', '_')}_escalation_plot.png"

    plt.figure(figsize=(12, 6))
    plt.plot(df_out["date"], df_out["rolling_sum"], linewidth=2)
    plt.axhline(y=args.threshold, linestyle="--")
    starts_df = df_out[df_out["escalation_start"]]
    if not starts_df.empty:
        plt.scatter(starts_df["date"], starts_df["rolling_sum"], s=80)

    plt.title(f"AEGIS Escalation Detection — {args.country}")
    plt.xlabel("Date")
    plt.ylabel(f"Rolling {args.window_days}-Day Fatalities")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot:", out_png)


if __name__ == "__main__":
    main()
