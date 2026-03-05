import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------

def parse_thresholds(raw: str) -> List[float]:
    """
    Accepts:
      "25" -> [25.0]
      "25,1000" -> [25.0, 1000.0]
      "25 1000" -> [25.0, 1000.0]
    """
    if raw is None:
        return [25.0]
    raw = raw.strip()
    if not raw:
        return [25.0]

    # allow commas or spaces
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    vals = sorted(list(dict.fromkeys(vals)))  # unique, sorted
    return vals


def smart_parse_dates(s: pd.Series) -> pd.Series:
    """
    Try hard to convert a column into pandas datetime.
    Handles:
      - normal date strings
      - YYYYMMDD integers
      - epoch seconds
      - epoch milliseconds
    Returns datetime64[ns] with NaT for unparseable rows.
    """
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce")

    # If already datetime-like
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")

    # Try numeric heuristics
    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors="coerce")
        med = np.nanmedian(x.values) if np.isfinite(x.values).any() else np.nan

        # YYYYMMDD (e.g., 20140223)
        if np.isfinite(med) and 1e7 <= med <= 3e8:
            return pd.to_datetime(x.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

        # epoch milliseconds (e.g., 1700000000000)
        if np.isfinite(med) and med >= 1e12:
            return pd.to_datetime(x, unit="ms", errors="coerce")

        # epoch seconds (e.g., 1700000000)
        if np.isfinite(med) and med >= 1e9:
            return pd.to_datetime(x, unit="s", errors="coerce")

        # fallback
        return pd.to_datetime(x, errors="coerce")

    # Otherwise treat as strings
    return pd.to_datetime(s.astype(str), errors="coerce")


def normalize_country(s: str) -> str:
    return (s or "").strip()


def compute_daily_series(
    df: pd.DataFrame,
    country: str,
    country_col: str,
    date_col: str,
    fatalities_col: str
) -> pd.DataFrame:
    """
    Filters by country, converts date + fatalities, aggregates to daily totals.
    Returns DataFrame with columns: date, daily_fatalities
    """
    if country_col not in df.columns:
        raise ValueError(f"Country column '{country_col}' not found in CSV.")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in CSV.")

    if fatalities_col not in df.columns:
        raise ValueError(f"Fatalities column '{fatalities_col}' not found in CSV.")

    target = normalize_country(country)
    if not target:
        raise ValueError("Country is empty. Enter a country name that matches the CSV exactly.")

    sub = df[df[country_col].astype(str).str.strip() == target].copy()
    if sub.empty:
        # helpful debugging hint
        examples = df[country_col].dropna().astype(str).str.strip().unique()[:10]
        raise ValueError(
            f"No rows matched country='{target}'. "
            f"Check spelling/case or country column. Example values: {list(examples)}"
        )

    sub["_date"] = smart_parse_dates(sub[date_col])
    sub["_fatalities"] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    sub = sub.dropna(subset=["_date"]).copy()
    if sub.empty:
        raise ValueError(
            f"After parsing '{date_col}', no valid dates were found. "
            f"Try a different date column name."
        )

    # Normalize to day (strip time)
    sub["_date_day"] = sub["_date"].dt.normalize()

    daily = (
        sub.groupby("_date_day", as_index=False)["_fatalities"]
        .sum()
        .rename(columns={"_date_day": "date", "_fatalities": "daily_fatalities"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


def add_rolling_and_escalation(
    daily: pd.DataFrame,
    rolling_days: int,
    thresholds: List[float],
    persistence_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds rolling fatalities.
    Computes escalation starts for each threshold.

    Returns:
      - daily with rolling column
      - starts_df: escalation start dates with threshold, rolling
    """
    if daily.empty:
        raise ValueError("Daily series is empty.")

    # Fill missing days so rolling is truly day-based
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily_full = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    window = int(rolling_days)
    if window < 1:
        raise ValueError("Rolling window must be >= 1 day.")

    daily_full["rolling"] = daily_full["daily_fatalities"].rolling(window=window, min_periods=1).sum()

    # For each threshold, compute persistent and start
    starts_rows = []
    for thr in thresholds:
        above = daily_full["rolling"] >= float(thr)

        # persistent means: above-threshold for N consecutive days ending today
        N = int(persistence_days)
        if N < 1:
            N = 1
        persistent = above.rolling(window=N, min_periods=N).sum() == N

        start = persistent & (~persistent.shift(1).fillna(False))

        daily_full[f"above_{thr}"] = above
        daily_full[f"persistent_{thr}"] = persistent
        daily_full[f"start_{thr}"] = start

        starts = daily_full.loc[start, ["date", "rolling"]].copy()
        if not starts.empty:
            starts["threshold"] = float(thr)
            starts_rows.append(starts)

    starts_df = pd.concat(starts_rows, ignore_index=True) if starts_rows else pd.DataFrame(
        columns=["date", "rolling", "threshold"]
    )

    return daily_full, starts_df


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AEGIS — Escalation Detection Demo", layout="wide")

st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation start markers.")

with st.sidebar:
    st.header("Inputs")

    use_example = st.button("Use built-in Ukraine example dataset")
    
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    country_col = st.text_input("Country column", value="country")
    country = st.text_input("Country (exact match)", value="Ukraine")

    date_col = st.text_input("Date column", value="date_start")
    fatalities_col = st.text_input("Fatalities column", value="best")

    rolling_days = st.number_input("Rolling window (days)", min_value=1, max_value=365, value=30, step=1)

    thresholds_raw = st.text_input("Threshold(s) (use 1 or 2, e.g. 25 or 25,1000)", value="25")
    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1, max_value=60, value=3, step=1
    )

    run_btn = st.button("Generate plot")


use_example = st.sidebar.button("Use built-in Ukraine example dataset")

if use_example:
    df = pd.read_csv("data/ukraine_sample.csv")
    st.sidebar.success("Loaded built-in Ukraine example dataset.")

elif uploaded is None:
    st.info("Upload a CSV in the sidebar, then click **Generate plot**.")
    st.stop()

if not run_btn:
    st.info("CSV uploaded — now click **Generate plot**.")
    st.stop()

try:
    thresholds = parse_thresholds(thresholds_raw)
    if len(thresholds) > 2:
        st.warning("You entered more than 2 thresholds. For the demo, only the first 2 will be used.")
        thresholds = thresholds[:2]

    # Read CSV
    # (low_memory=False reduces mixed-type inference issues on large files)
    df = pd.read_csv(uploaded, low_memory=False)

    # Build daily series
    daily = compute_daily_series(
        df=df,
        country=country,
        country_col=country_col,
        date_col=date_col,
        fatalities_col=fatalities_col
    )

    daily_full, starts_df = add_rolling_and_escalation(
        daily=daily,
        rolling_days=int(rolling_days),
        thresholds=thresholds,
        persistence_days=int(persistence_days)
    )

    # ---- Plot
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plt.figure(figsize=(12, 5))
        plt.plot(daily_full["date"], daily_full["rolling"])

        # Threshold lines + markers
        for thr in thresholds:
            plt.axhline(y=float(thr), linestyle="--")
            starts_thr = starts_df[starts_df["threshold"] == float(thr)]
            if not starts_thr.empty:
                plt.scatter(starts_thr["date"], starts_thr["rolling"], s=60)

        plt.title(
            f"AEGIS Escalation Detection — {country} (rolling={int(rolling_days)}d, thresholds={','.join(str(int(t)) if t.is_integer() else str(t) for t in thresholds)})"
        )
        plt.xlabel("Date")
        plt.ylabel("Rolling window fatalities")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Summary")

        st.write(f"Rows (daily): **{len(daily_full)}**")

        # Summary for each threshold
        for thr in thresholds:
            st.markdown(f"**Threshold = {thr}**")
            above = daily_full[f"above_{thr}"].sum()
            persistent = daily_full[f"persistent_{thr}"].sum()
            starts = daily_full[f"start_{thr}"].sum()

            st.write(f"Days above threshold: **{int(above)}**")
            st.write(f"Days persistent: **{int(persistent)}**")
            st.write(f"Escalation starts detected: **{int(starts)}**")

        if not starts_df.empty:
            st.write("First escalation starts:")
            st.dataframe(starts_df.sort_values(["threshold", "date"]).head(20), use_container_width=True)

        # Debug hint if dates look wrong
        if daily_full["date"].min() < pd.Timestamp("1980-01-01"):
            st.warning(
                "Your parsed dates start very early (before 1980). "
                "That often means the chosen date column isn't the real event date (or is being parsed as epoch). "
                "Double-check the 'Date column' input."
            )

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
