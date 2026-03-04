# step2_realworld_historical.py
# Step 2: Real-world historical escalation detection on a CSV (e.g., UCDP GED)
#
# Example (Colab / terminal):
#   python step2_realworld_historical.py --csv GEDEvent_v25_1.csv --country "Ukraine" --date-col date_start --fatalities-col best
#
# Output:
#   - Prints basic counts + escalation start dates
#   - Saves a PNG plot (default: ukraine_escalation_plot.png)
#
# Notes:
# - This script expects "country" to be a column in your CSV by default. You can change with --country-col.
# - For UCDP GED, common choices:
#     --date-col date_start
#     --fatalities-col best

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS Step 2 — Real-world historical escalation detection from CSV.")
    p.add_argument("--csv", required=True, help="Path to CSV file (e.g., GEDEvent_v25_1.csv)")
    p.add_argument("--country", required=True, help="Country name to filter (must match values in --country-col)")

    p.add_argument("--country-col", default="country", help="Column name for country (default: country)")
    p.add_argument("--date-col", default="date_start", help="Column name for date (default: date_start)")
    p.add_argument("--fatalities-col", default="best", help="Column name for fatalities/deaths (default: best)")

    p.add_argument("--window-days", type=int, default=30, help="Rolling window length in days (default: 30)")
    p.add_argument("--threshold", type=float, default=25, help="Escalation threshold (default: 25)")

    # Persistence: require N consecutive days above threshold before we count an escalation start
    p.add_argument("--persistence-days", type=int, default=3, help="Days above threshold required to confirm escalation (default: 3)")
    # Optional cooldown: require N days below threshold before allowing a *new* escalation start
    p.add_argument("--cooldown-days", type=int, default=14, help="Cooldown days below threshold before another start (default: 14)")

    p.add_argument("--plot-file", default="ukraine_escalation_plot.png", help="PNG filename to save plot (default: ukraine_escalation_plot.png)")
    p.add_argument("--max-years", type=int, default=None, help="Optional: limit to last N years (e.g., 5). Default: no limit.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info")

    return p.parse_args()


def load_and_aggregate_csv(
    csv_path: str,
    country: str,
    country_col: str,
    date_col: str,
    fatalities_col: str,
    max_years: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV (try a couple common encodings automatically)
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    required = [country_col, date_col, fatalities_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns include: {list(df.columns)[:25]} ...\n"
            f"Tip: pass different names with --country-col / --date-col / --fatalities-col."
        )

    # Filter to the chosen country (make a real copy to avoid SettingWithCopyWarning)
    sub = df[df[country_col].astype(str).str.strip() == country].copy()
    if sub.empty:
        # Try a more forgiving match (case-insensitive contains) if exact match fails
        sub2 = df[df[country_col].astype(str).str.lower().str.contains(str(country).lower(), na=False)].copy()
        if not sub2.empty:
            sub = sub2
        else:
            raise ValueError(
                f"No rows found for country='{country}' in column '{country_col}'.\n"
                f"Check spelling/case. Example values: {df[country_col].dropna().astype(str).unique()[:10]}"
            )

    # Parse date + fatalities safely
    sub["date_tmp"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # Drop rows with missing dates
    sub = sub.dropna(subset=["date_tmp"])

    if sub.empty:
        raise ValueError(f"After parsing, no valid dates remained in column '{date_col}' for country '{country}'.")

    # Optional limit to last N years
    if max_years is not None and max_years > 0:
        latest = sub["date_tmp"].max()
        cutoff = latest - pd.DateOffset(years=max_years)
        sub = sub[sub["date_tmp"] >= cutoff].copy()

    # Aggregate to DAILY totals (IMPORTANT: keep the date as a column, not the index)
    # We group by the date part only (drop time-of-day if present)
    sub_daily = (
        sub.groupby(sub["date_tmp"].dt.date, as_index=False)[fatalities_col]
          .sum()
          .rename(columns={"date_tmp": "date"})
    )

    # Convert back to datetime for rolling time series operations
    sub_daily["date"] = pd.to_datetime(sub_daily["date"])
    sub_daily = sub_daily.sort_values("date").reset_index(drop=True)

    if verbose:
        print("Loaded rows (filtered):", len(sub))
        print("Daily rows:", len(sub_daily))
        print("Date range:", sub_daily["date"].min(), "->", sub_daily["date"].max())
        print("Daily fatalities sample:", sub_daily.head(3).to_dict(orient="records"))

    return sub_daily.rename(columns={fatalities_col: "fatalities"})


def compute_escalation(
    df_daily: pd.DataFrame,
    window_days: int,
    threshold: float,
    persistence_days: int,
    cooldown_days: int,
) -> pd.DataFrame:
    out = df_daily.copy()
    out["fatalities"] = pd.to_numeric(out["fatalities"], errors="coerce").fillna(0)

    # Rolling sum
    out["rolling_sum"] = out["fatalities"].rolling(window=window_days, min_periods=window_days).sum()

    # Above threshold (bool)
    out["above_threshold"] = out["rolling_sum"] >= threshold

    # Persistence: require N consecutive days above threshold
    if persistence_days <= 1:
        out["persistent"] = out["above_threshold"]
    else:
        # Rolling window over boolean -> all True in last N days
        out["persistent"] = (
            out["above_threshold"]
            .rolling(window=persistence_days, min_periods=persistence_days)
            .apply(lambda x: 1.0 if np.all(x.astype(bool)) else 0.0, raw=False)
            .fillna(0)
            .astype(int)
            .astype(bool)
        )

    # Cooldown: require N consecutive days below threshold before a new escalation start is allowed
    if cooldown_days <= 0:
        out["cooldown_met"] = True
    else:
        below = ~out["above_threshold"].fillna(False)
        out["cooldown_met"] = (
            below.rolling(window=cooldown_days, min_periods=cooldown_days)
            .apply(lambda x: 1.0 if np.all(x.astype(bool)) else 0.0, raw=False)
            .fillna(0)
            .astype(int)
            .astype(bool)
        )

    # Escalation start:
    # - persistent today
    # - NOT persistent yesterday
    # - cooldown was met yesterday (or earlier) to allow a new start
    prev_persistent = out["persistent"].shift(1).fillna(False)
    prev_cooldown = out["cooldown_met"].shift(1).fillna(True)
    out["escalation_start"] = (out["persistent"] & (~prev_persistent) & prev_cooldown)

    return out


def make_plot(df: pd.DataFrame, threshold: float, country: str, plot_file: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["rolling_sum"], label="Rolling 30-day fatalities")
    plt.axhline(y=threshold, linestyle="--", label=f"Threshold = {threshold}")

    # Mark escalation starts
    starts = df[df["escalation_start"]]
    if not starts.empty:
        plt.scatter(starts["date"], starts["rolling_sum"], s=70, label="Escalation start")

    plt.title(f"AEGIS Escalation Detection — {country}")
    plt.xlabel("Date")
    plt.ylabel("Rolling 30-day fatalities")
    plt.tight_layout()

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    # In notebooks, show() helps you see it immediately
    plt.show()


def main() -> None:
    args = parse_args()

    # Some people upload files without .csv extension in Colab.
    # If they passed a path with no extension but there is a .csv next to it, try that.
    csv_path = args.csv
    if not os.path.exists(csv_path) and not csv_path.lower().endswith(".csv"):
        alt = csv_path + ".csv"
        if os.path.exists(alt):
            csv_path = alt

    df_daily = load_and_aggregate_csv(
        csv_path=csv_path,
        country=args.country,
        country_col=args.country_col,
        date_col=args.date_col,
        fatalities_col=args.fatalities_col,
        max_years=args.max_years,
        verbose=args.verbose,
    )

    df_out = compute_escalation(
        df_daily=df_daily,
        window_days=args.window_days,
        threshold=args.threshold,
        persistence_days=args.persistence_days,
        cooldown_days=args.cooldown_days,
    )

    # Summary stats
    days_above = int(df_out["above_threshold"].fillna(False).sum())
    days_persistent = int(df_out["persistent"].fillna(False).sum())
    starts = df_out[df_out["escalation_start"]]

    print(f"Days above threshold: {days_above}")
    print(f"Days persistent: {days_persistent}")
    print(f"Escalation starts detected: {len(starts)}")

    if len(starts) > 0:
        print("\nFirst few escalation start dates:")
        for d in starts["date"].head(10).dt.strftime("%Y-%m-%d").tolist():
            print(" -", d)

    # Plot + save
    plot_file = args.plot_file
    make_plot(df_out, threshold=args.threshold, country=args.country, plot_file=plot_file)

    print(f"\nSaved plot to: {plot_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        sys.exit(1)
