# app.py
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

# Files expected in your GitHub repo (same folder as app.py)
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")

# HuggingFace-hosted world dataset (direct "resolve" link)
HF_WORLD_CSV_URL = "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"

# Sidebar video (optional

if missing; it won't crash)
SIDEBAR_VIDEO_PATH = Path("logo1.mp4")


# ----------------------------
# Helpers: robust CSV reading
# ----------------------------
def _looks_like_html(b: bytes) -> bool:
    head = b[:600].lower()
    return b"<!doctype html" in head or b"<html" in head or b"<head" in head


def _safe_read_csv_bytes(data: bytes) -> pd.DataFrame:
    """
    Read CSV bytes robustly:
    - try common encodings
    - try auto separator (python engine) if needed
    """
    if not data or len(data) < 10:
        raise ValueError("CSV appears empty.")
    if _looks_like_html(data):
        raise ValueError("Downloaded content looks like HTML, not a CSV.")

    # Try encodings
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            # First try normal fast parse
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception as e:
            last_err = e

        # Fallback: auto-detect delimiter
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e

    raise last_err if last_err else ValueError("Unable to parse CSV.")


@st.cache_data(show_spinner=False)
def read_csv_path(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    b = p.read_bytes()
    return _safe_read_csv_bytes(b)


@st.cache_data(show_spinner=False)
def read_csv_upload(uploaded_file) -> pd.DataFrame:
    b = uploaded_file.getvalue()
    return _safe_read_csv_bytes(b)


@st.cache_data(show_spinner=False)
def download_world_csv_bytes(url: str) -> bytes:
    """
    Download the world dataset once (cached).
    NOTE: Large file, caching helps avoid re-downloading.
    """
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    # Some hosts compress requests handles decoding automatically.
    return r.content


@st.cache_data(show_spinner=False)
def read_world_dataset(url: str) -> pd.DataFrame:
    b = download_world_csv_bytes(url)
    df = _safe_read_csv_bytes(b)
    return df


# ----------------------------
# Column guessing / normalization
# ----------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def guess_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Best-effort guess for country/date/fatalities columns.
    Returns dict keys: 'country', 'date', 'fatalities'
    """
    cols = list(df.columns)
    ncols = {_norm(c): c for c in cols}

    # Country candidates
    country_keys = ["country", "countryname", "location", "state", "nation"]
    country = None
    for k in country_keys:
        if k in ncols:
            country = ncols[k]
            break

    # Date candidates
    date_keys = [
        "datestart",
        "date_start",
        "date",
        "eventdate",
        "event_date",
        "day",
        "time",
        "timestamp",
    ]
    date = None
    for k in date_keys:
        if k in ncols:
            date = ncols[k]
            break

    # Fatalities candidates
    fat_keys = ["best", "fatalities", "deaths", "killed", "fat", "totfatalities"]
    fatalities = None
    for k in fat_keys:
        if k in ncols:
            fatalities = ncols[k]
            break

    return {"country": country, "date": date, "fatalities": fatalities}


def ensure_country_column(
    df: pd.DataFrame,
    guessed_country_col: Optional[str],
    fill_country: str,
) -> Tuple[pd.DataFrame, str]:
    """
    If dataset already has a country-like column, return it.
    Otherwise, add a 'country' column filled with fill_country.
    """
    if guessed_country_col and guessed_country_col in df.columns:
        return df, guessed_country_col

    # Add a stable column name that matches your default UI
    if "country" not in df.columns:
        df = df.copy()
        df["country"] = fill_country
    return df, "country"


# ----------------------------
# Core analysis
# ----------------------------
def parse_thresholds(raw: str) -> List[float]:
    """
    Accept "25" or "25,50" (up to 2 values). Ignores empties.
    """
    parts = [p.strip() for p in str(raw).split(",")]
    vals: List[float] = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("Please enter at least one threshold (e.g., 25 or 25,50).")
    if len(vals) > 2:
        st.warning("You entered more than 2 thresholds. Only the first 2 will be used.")
        vals = vals[:2]
    return vals


def build_daily_series(
    df: pd.DataFrame,
    country_col: str,
    date_col: str,
    fatalities_col: str,
    country_name: str,
) -> pd.DataFrame:
    """
    Builds daily fatalities series for a specific country.
    Includes auto-detect for missing columns (especially for ukraine_sample.csv).
    """
    # If user-provided columns don't exist, try auto-detect
    if country_col not in df.columns or date_col not in df.columns or fatalities_col not in df.columns:
        guessed = guess_columns(df)

        # If sample has no country column, add it
        if country_col not in df.columns:
            df, country_col = ensure_country_column(df, guessed["country"], fill_country=country_name)

        # Auto-fix date/fatalities if missing
        if date_col not in df.columns and guessed["date"] is not None:
            date_col = guessed["date"]
        if fatalities_col not in df.columns and guessed["fatalities"] is not None:
            fatalities_col = guessed["fatalities"]

    # Hard fail only if still missing after guessing
    if country_col not in df.columns:
        raise ValueError(f"Missing column '{country_col}'. Available columns: {list(df.columns)}")
    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'. Available columns: {list(df.columns)}")
    if fatalities_col not in df.columns:
        raise ValueError(f"Missing column '{fatalities_col}'. Available columns: {list(df.columns)}")

    sub = df[df[country_col] == country_name].copy()
    if sub.empty:
        raise ValueError(f"No rows found for country='{country_name}'. Check spelling/case vs dataset.")

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    daily = (
        sub.groupby(sub[date_col].dt.floor("D"))[fatalities_col]
        .sum()
        .reset_index()
    )
    daily.columns = ["date", "fatalities"]
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def rolling_and_escalations(
    daily: pd.DataFrame,
    rolling_window_days: int,
    thresholds: List[float],
    persistence_days: int,
) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame]]:
    """
    Adds rolling sum and detects escalation starts for each threshold.
    Escalation start = first day of a run of >= persistence_days above threshold.
    """
    df = daily.copy()
    df["rolling"] = df["fatalities"].rolling(window=int(rolling_window_days), min_periods=int(rolling_window_days)).sum()

    starts_by_thr: Dict[float, pd.DataFrame] = {}
    for thr in thresholds:
        above = df["rolling"] > thr
        # Count consecutive Trues
        consec = np.where(above, 1, 0).astype(int)
        run = np.zeros_like(consec)
        for i in range(len(consec)):
            run[i] = (run[i - 1] + 1) if (i > 0 and consec[i] == 1) else consec[i]

        df[f"run_{thr}"] = run
        # Escalation start when run hits persistence_days (i.e. the first day reaching required streak)
        starts_mask = df[f"run_{thr}"] == persistence_days
        starts = df.loc[starts_mask, ["date", "rolling"]].copy()
        starts["threshold"] = thr
        starts_by_thr[thr] = starts.reset_index(drop=True)

    return df, starts_by_thr


# ----------------------------
# Map aggregation
# ----------------------------
def map_aggregate(
    df: pd.DataFrame,
    country_col: str,
    date_col: str,
    fatalities_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate fatalities by country in [start, end].
    Works on already-loaded df (world dataset or uploaded).
    """
    guessed = guess_columns(df)

    if country_col not in df.columns and guessed["country"] is not None:
        country_col = guessed["country"]
    if date_col not in df.columns and guessed["date"] is not None:
        date_col = guessed["date"]
    if fatalities_col not in df.columns and guessed["fatalities"] is not None:
        fatalities_col = guessed["fatalities"]

    if country_col not in df.columns or date_col not in df.columns or fatalities_col not in df.columns:
        raise ValueError(
            f"Map needs columns: {country_col}, {date_col}, {fatalities_col}. "
            f"Available: {list(df.columns)}"
        )

    d = df[[country_col, date_col, fatalities_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d[fatalities_col] = pd.to_numeric(d[fatalities_col], errors="coerce").fillna(0)

    mask = (d[date_col] >= start) & (d[date_col] <= end)
    d = d.loc[mask]
    out = d.groupby(country_col, as_index=False)[fatalities_col].sum()
    out.columns = ["country", "fatalities"]
    return out


# ----------------------------
# UI: Sidebar
# ----------------------------
with st.sidebar:
    st.header("Inputs")

    # Sidebar video (optional)
    if SIDEBAR_VIDEO_PATH.exists():
        st.video(str(SIDEBAR_VIDEO_PATH))

    # Map toggle (starts ON to turn the map OFF)
    map_off = st.checkbox("Turn off interactive map", value=True, help="Checked = hide the map section.")

    # Demo checkbox (unchecked by default)
    use_sample = st.checkbox(
        "Use built-in Ukraine example (recommended demo)",
        value=False,
        help="Uses the repo file: ukraine_sample.csv",
    )

    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset as a CSV file.")

    country_name = st.text_input(
        "Country (exact match)",
        "Ukraine",
        help="Must match the dataset exactly (case/spelling). Example: Ukraine, Mexico, Syria.",
    )

    country_col = st.text_input(
        "Country column",
        "country",
        help="Name of the dataset column that contains country names.",
    )
    date_col = st.text_input(
        "Date column",
        "date_start",
        help="Name of the dataset column that contains dates (e.g., date_start).",
    )
    fatalities_col = st.text_input(
        "Fatalities column",
        "best",
        help="Name of the dataset column that contains fatality counts (e.g., best).",
    )

    rolling_window = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Rolling sum window length in days.",
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        "25,50",
        help="Enter one or two thresholds. Example: 25 or 25,50",
    )

    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1,
        max_value=60,
        value=3,
        step=1,
        help="How many consecutive days the rolling value must stay above the threshold to count as an escalation start.",
    )

    st.divider()

    # Map date range controls (default auto advanced edit in sidebar)
    edit_map_range = st.checkbox(
        "Edit map date range",
        value=False,
        help="By default, the map uses the full available date range automatically.",
    )

    run_btn = st.button("Generate plot", type="primary")


# ----------------------------
# Main layout / headers
# ----------------------------
st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")

st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears below.")


# ----------------------------
# Load primary dataset for plot (upload or demo)
# ----------------------------
def load_primary_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Returns (df_for_plot_or_demo, source_label)
    """
    if use_sample:
        if not UKRAINE_SAMPLE_PATH.exists():
            raise FileNotFoundError(
                "Built-in sample not found. Ensure **ukraine_sample.csv** is in the same folder as app.py in GitHub."
            )
        df_demo = read_csv_path(str(UKRAINE_SAMPLE_PATH))
        return df_demo, "Demo: ukraine_sample.csv"

    if uploaded is not None:
        df_up = read_csv_upload(uploaded)
        return df_up, f"Uploaded: {getattr(uploaded, 'name', 'CSV')}"

    return None, "No upload/demo selected"


# ----------------------------
# Interactive Map Section
# ----------------------------
st.header("Interactive map")
st.caption("Data source: HuggingFace hosted world dataset")

if map_off:
    st.info("Map is turned off (toggle it on in the sidebar).")
else:
    # Decide what data to use for the map:
    # - If upload/demo selected, use that (so map matches the same dataset)
    # - Otherwise, load the HuggingFace world dataset (for a useful default)
    try:
        df_plot_or_demo, plot_source = load_primary_dataset()
        if df_plot_or_demo is not None:
            df_map_base = df_plot_or_demo
            st.success(f"Map source: {plot_source}")
        else:
            st.info("No upload/demo selected — loading the HuggingFace world dataset for the map.")
            df_map_base = read_world_dataset(HF_WORLD_CSV_URL)

        # Auto range from data
        g = guess_columns(df_map_base)
        date_col_for_range = date_col if date_col in df_map_base.columns else (g["date"] or date_col)

        if date_col_for_range not in df_map_base.columns:
            raise ValueError(f"Map can't find a date column. Available columns: {list(df_map_base.columns)}")

        dtmp = pd.to_datetime(df_map_base[date_col_for_range], errors="coerce")
        dmin = pd.Timestamp(dtmp.min()).normalize()
        dmax = pd.Timestamp(dtmp.max()).normalize()

        if pd.isna(dmin) or pd.isna(dmax):
            raise ValueError("Map couldn't determine date range (date parsing failed).")

        # Default: automatic full range optional edit in sidebar
        if edit_map_range:
            start_date, end_date = st.sidebar.date_input(
                "Map date range",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
                help="Pick the date range used for the map aggregation.",
            )
            map_start = pd.Timestamp(start_date)
            map_end = pd.Timestamp(end_date)
        else:
            map_start, map_end = dmin, dmax

        st.markdown(f"**Map date range:** {map_start.date()} — {map_end.date()}")

        agg = map_aggregate(
            df_map_base,
            country_col=country_col,
            date_col=date_col,
            fatalities_col=fatalities_col,
            start=map_start,
            end=map_end,
        )

        if agg.empty:
            st.warning("No data in the selected map date range.")
        else:
            fig = px.choropleth(
                agg,
                locations="country",
                locationmode="country names",
                color="fatalities",
                hover_name="country",
                color_continuous_scale="Blues",
                title="Fatalities by country (selected date range)",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Map error: {e}")


# ----------------------------
# Escalation Plot Section
# ----------------------------
st.header("Escalation plot")

# If the user hasn't clicked Generate plot yet, keep the page from being empty.
if not run_btn:
    st.info("Click **Generate plot** to compute the rolling fatalities and escalation starts.")
    st.stop()

try:
    df_raw_plot, plot_source = load_primary_dataset()
    if df_raw_plot is None:
        st.warning("Please upload a CSV or enable the built-in demo.")
        st.stop()

    thresholds = parse_thresholds(thresholds_raw)

    # Build daily country series (auto-fixes demo column mismatches)
    daily = build_daily_series(
        df_raw_plot,
        country_col=country_col,
        date_col=date_col,
        fatalities_col=fatalities_col,
        country_name=country_name,
    )

    series, starts_by_thr = rolling_and_escalations(
        daily=daily,
        rolling_window_days=int(rolling_window),
        thresholds=thresholds,
        persistence_days=int(persistence_days),
    )

    # Summary block
    total_rows = len(series)
    days_above_total = {}
    days_persistent_total = {}
    esc_count = {}

    for thr in thresholds:
        above = series["rolling"] > thr
        days_above_total[thr] = int(above.sum())
        days_persistent_total[thr] = int((series[f"run_{thr}"] >= persistence_days).sum())
        esc_count[thr] = int(len(starts_by_thr[thr]))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series["date"], series["rolling"], linewidth=1.5)
    ax.set_title(f"AEGIS Escalation Detection — {country_name} (rolling={int(rolling_window)}d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling fatalities")

    # Threshold lines + markers
    for thr in thresholds:
        ax.axhline(thr, linestyle="--", linewidth=1.2)
        starts = starts_by_thr[thr]
        if not starts.empty:
            ax.scatter(starts["date"], starts["rolling"], s=35)

    st.pyplot(fig, clear_figure=True)

    # Right-side style summary + table
    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.subheader("Summary")
        st.write(f"**Source:** {plot_source}")
        st.write(f"**Rows (daily):** {total_rows}")

        for thr in thresholds:
            st.write(f"**Threshold = {thr}**")
            st.write(f"- Days above threshold: {days_above_total[thr]}")
            st.write(f"- Days persistent (≥ {persistence_days} consecutive): {days_persistent_total[thr]}")
            st.write(f"- Escalation starts detected: {esc_count[thr]}")

    with c2:
        st.subheader("First escalation starts")
        rows = []
        for thr in thresholds:
            starts = starts_by_thr[thr]
            if not starts.empty:
                for _, r in starts.head(10).iterrows():
                    rows.append(
                        {
                            "threshold": thr,
                            "date": r["date"],
                            "rolling": float(r["rolling"]),
                        }
                    )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("No escalation starts detected with the current settings.")

except Exception as e:
    st.error(str(e))
