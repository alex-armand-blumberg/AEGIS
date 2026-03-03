import argparse
import matplotlib.pyplot as plt

from src.config import (
    THRESHOLD_DEATHS_30D, PERSISTENCE_DAYS, COOLDOWN_DAYS,
    DATE_COL, FATALITIES_COL, GROUP_BY
)
from src.aegis import load_data, aggregate_daily, compute_rolling, detect_escalations, summarize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--country", default=None, help="Optional: filter to a single country (exact match)")
    args = parser.parse_args()

    df = load_data(args.csv, DATE_COL, FATALITIES_COL)

    if args.country and "country" in df.columns:
        df = df[df["country"] == args.country]

    daily = aggregate_daily(df, DATE_COL, FATALITIES_COL, group_by=[c for c in GROUP_BY if c in df.columns])
    daily = compute_rolling(daily, window_days=30, group_by=[c for c in GROUP_BY if c in daily.columns])
    out = detect_escalations(
        daily,
        threshold=THRESHOLD_DEATHS_30D,
        persistence_days=PERSISTENCE_DAYS,
        cooldown_days=COOLDOWN_DAYS,
        group_by=[c for c in GROUP_BY if c in daily.columns],
    )

    stats = summarize(out)
    print("Summary:", stats)

    # Plot overall if no groups; otherwise plot first group
    if "country" in out.columns:
        first_country = out["country"].iloc[0]
        plot_df = out[out["country"] == first_country].copy()
        title = f"AEGIS: Rolling 30-day fatalities ({first_country})"
    else:
        plot_df = out.copy()
        title = "AEGIS: Rolling 30-day fatalities"

    plt.figure(figsize=(12,6))
    plt.plot(plot_df["date"], plot_df["rolling_30"])
    plt.axhline(y=THRESHOLD_DEATHS_30D, linestyle="--")
    starts = plot_df[plot_df["escalation_start"]]
    if len(starts) > 0:
        plt.scatter(starts["date"], starts["rolling_30"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling 30-day fatalities")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
