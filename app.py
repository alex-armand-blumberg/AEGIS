# app.py
# AEGIS — Escalation Detection Demo (Streamlit)
#
# Features:
# - Upload CSV OR use built-in Ukraine demo (ukraine_sample.csv in repo) (unchecked by default)
# - Two thresholds supported (comma-separated)
# - Rolling fatalities plot + escalation-start markers (with persistence)
# - Interactive map placeholder area under the intro:
#     • If no upload/demo is selected, app downloads a global CSV from Google Drive (cached)
#     • Map uses latitude/longitude columns if present; otherwise shows a clear message
# - Robust Google Drive download using /uc?export=download&id=...
#
# Repo files you may optionally include:
# - ukraine_sample.csv
# - logo1.mp4  (sidebar video banner)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st


# -----------------------------
# Page config + small CSS tweaks
# -----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
/* Make sidebar content breathe a bit */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Constants
# -----------------------------
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")

# Your Google Drive file
GDRIVE_FILE_ID = "1lg3YUmyxb8aqXCLtnPgGXnoJb8pIAEGF"
GDRIVE_DIRECT_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
GDRIVE_LOCAL_CACHE = Path("aegis_global.csv")

# Optional sidebar video banner (if present in repo)
SIDEBAR_VIDEO = Path("logo1.mp4")


# -----------------------------
# Utilities
# -----------------------------
def parse_thresholds(raw: str, max_n: int = 2) -> List[float]:
    """
    Parse comma-separated thresholds like "25" or "25,1000".
    Returns up to max_n thresholds.
    """
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            raise ValueError(f"Invalid threshold value: {p!r}")
    if not vals:
        raise ValueError("Please enter at least 1 threshold (e.g., 25 or 25,1000).")
    return vals[:max_n]


def looks_like_html(text_head: str) -> bool:
    head = (text_head or "").lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html") or "<title" in head


@st.cache_data(show_spinner=False)
def download_google_drive_csv_to_disk(url: str, local_path_str: str) -> str:
    """
    Download a large CSV from Google Drive to local disk (cached by Streamlit).
    Uses the /uc?export=download&id=... direct endpoint.

    If Drive returns HTML (virus-scan interstitial), we fail with a clear error.
    """
    local_path = Path(local_path_str)

    # If already on disk in the current container, reuse it
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    r = requests.get(url, stream=True, timeout=120)

    # Peek at the beginning to ensure it's not HTML
    # (We read a small chunk, then continue streaming the full content.)
    it = r.iter_content(chunk_size=64 * 1024)
    first_chunk = b""
    try:
        first_chunk = next(it)
    except StopIteration:
        raise RuntimeError("Google Drive returned an empty response.")

    head_text = first_chunk[:4096].decode("utf-8", errors="ignore")
    if looks_like_html(head_text):
        raise RuntimeError(
            "Google Drive returned an HTML page (often the virus-scan confirmation page) "
            "instead of the CSV. Fixes:\n"
            "1) Ensure the file is shared as 'Anyone with the link' and is downloadable.\n"
            "2) If Drive still blocks, the most reliable option is to host the file on "
            "HuggingFace Datasets, S3, or another direct file host."
        )

    # Write first chunk + rest
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(first_chunk)
        for chunk in it:
            if chunk:
                f.write(chunk)

    if local_path.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty on disk.")

    return str(local_path)


@st.cache_data(show_spinner=False)
def read_csv_any(source: str) -> pd.DataFrame:
    """
    Read a CSV from a local path (string). Cached.
    """
    return pd.read_csv(source)


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def coerce_datetime_series(s: pd.Series) -> pd.Series:
    # Handles strings like "2017-07-31 00:00:00.000"
    return pd.to_datetime(s, errors="coerce")


@dataclass
class EscalationResult:
    daily: pd.DataFrame
    thresholds: List[float]
    persistence_days: int
    starts: pd.DataFrame  # columns: ["date", "rolling", "threshold"]


def build_daily_series(
    df: pd.DataFrame,
    country_col: str,
    date_col: str,
    fatalities_col: str,
    country_name: str,
) -> pd.DataFrame:
    """
    Returns a daily aggregated dataframe with columns:
    date (datetime), fatalities (float)
    """
    if country_col not in df.columns or date_col not in df.columns or fatalities_col not in df.columns:
        missing = [c for c in [country_col, date_col, fatalities_col] if c not in df.columns]
        raise KeyError(f"Missing column(s): {missing}. Available: {list(df.columns)[:50]}")

    sub = df.loc[df[country_col].astype(str) == str(country_name)].copy()

    sub[date_col] = coerce_datetime_series(sub[date_col])
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = coerce_numeric_series(sub[fatalities_col])

    daily = (
        sub.groupby(sub[date_col].dt.floor("D"), as_index=False)[fatalities_col]
        .sum()
        .rename(columns={date_col: "date", fatalities_col: "fatalities"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def detect_escalations(
    daily: pd.DataFrame,
    rolling_window_days: int,
    thresholds: List[float],
    persistence_days: int,
) -> EscalationResult:
    """
    Adds rolling sum + computes escalation-start dates for each threshold.
    """
    if daily.empty:
        starts = pd.DataFrame(columns=["date", "rolling", "threshold"])
        return EscalationResult(daily=daily, thresholds=thresholds, persistence_days=persistence_days, starts=starts)

    out = daily.copy()
    out["rolling"] = out["fatalities"].rolling(int(rolling_window_days), min_periods=1).sum()

    starts_rows = []
    for thr in thresholds:
        above = out["rolling"] >= thr

        # Persistence: consecutive days above threshold >= persistence_days
        # Use rolling window on the boolean series
        # persistent[t] = all(above[t-p+1 : t])  (persistence_days)
        p = int(persistence_days)
        if p <= 1:
            persistent = above
        else:
            persistent = above.rolling(p, min_periods=p).sum() == p

        start = persistent & (~persistent.shift(1).fillna(False))
        if start.any():
            for _, row in out.loc[start, ["date", "rolling"]].iterrows():
                starts_rows.append({"date": row["date"], "rolling": float(row["rolling"]), "threshold": float(thr)})

        out[f"above_{thr:g}"] = above
        out[f"persistent_{thr:g}"] = persistent

    starts = pd.DataFrame(starts_rows).sort_values("date").reset_index(drop=True) if starts_rows else pd.DataFrame(
        columns=["date", "rolling", "threshold"]
    )
    return EscalationResult(daily=out, thresholds=thresholds, persistence_days=persistence_days, starts=starts)


def plot_escalation(res: EscalationResult, country_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))

    if res.daily.empty:
        ax.set_title(f"AEGIS Escalation Detection — {country_name} (no data)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling window fatalities")
        return fig

    ax.plot(res.daily["date"], res.daily["rolling"], linewidth=1.6)

    # Draw threshold lines
    for thr in res.thresholds:
        ax.axhline(float(thr), linestyle="--", linewidth=1)

    # Mark escalation starts
    if not res.starts.empty:
        # Different marker per threshold (but keep it simple)
        markers = ["o", "s", "D", "^"]
        for i, thr in enumerate(res.thresholds):
            pts = res.starts.loc[res.starts["threshold"] == float(thr)]
            if pts.empty:
                continue
            ax.scatter(pts["date"], pts["rolling"], s=55, marker=markers[i % len(markers)], zorder=3)

    thr_label = ",".join([f"{t:g}" for t in res.thresholds])
    ax.set_title(f"AEGIS Escalation Detection — {country_name} (rolling={int(res.daily.shape[0] and 0 or 0)}d, thresholds={thr_label})")
    # Use the actual rolling window in title elsewhere; keep this clean here
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling window fatalities")
    fig.autofmt_xdate()
    return fig


def safe_read_demo() -> pd.DataFrame:
    if not UKRAINE_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            "Built-in demo file not found. Add 'ukraine_sample.csv' to your repo root."
        )
    return pd.read_csv(UKRAINE_SAMPLE_PATH)


def show_map(df: pd.DataFrame) -> None:
    """
    Shows an interactive map if latitude/longitude columns exist.
    Otherwise shows a helpful message.
    """
    # Try common column names
    cand_lat = [c for c in df.columns if c.lower() in ("latitude", "lat", "event_lat", "geom_lat")]
    cand_lon = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng", "event_lon", "geom_lon")]

    if not cand_lat or not cand_lon:
        st.info(
            "Map note: Your dataset doesn't include latitude/longitude columns, so the app can't plot points on a map.\n\n"
            "If you want the interactive map, add columns like `latitude` and `longitude` (or `lat`/`lon`)."
        )
        return

    lat_col = cand_lat[0]
    lon_col = cand_lon[0]

    tmp = df[[lat_col, lon_col]].copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna().head(20000)  # safety cap for performance

    if tmp.empty:
        st.info("Map note: Latitude/longitude columns exist, but no valid coordinates were found.")
        return

    st.map(tmp.rename(columns={lat_col: "lat", lon_col: "lon"}))


# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.header("Inputs")

# Optional video banner in sidebar (if present)
if SIDEBAR_VIDEO.exists():
    st.sidebar.video(str(SIDEBAR_VIDEO))
else:
    # keep layout consistent
    st.sidebar.markdown("")

use_demo = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,  # IMPORTANT: starts unchecked
    help="Uses `ukraine_sample.csv` from the GitHub repo (small demo file).",
)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload your own CSV. Streamlit Cloud has a visible per-upload size cap in the UI; large datasets are better hosted externally.",
    )

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Must match the dataset values exactly (e.g., 'Ukraine', 'Mexico', 'Syria').",
)

country_col = st.sidebar.text_input(
    "Country column",
    "country",
    help="Column name that contains the country name for each row.",
)

date_col = st.sidebar.text_input(
    "Date column",
    "date_start",
    help="Column name that contains the event date (will be parsed as a date).",
)

fatalities_col = st.sidebar.text_input(
    "Fatalities column",
    "best",
    help="Column name for fatalities (numeric). Non-numeric values will be treated as 0.",
)

rolling_window_days = st.sidebar.number_input(
    "Rolling window (days)",
    min_value=1,
    max_value=365,
    value=30,
    step=1,
    help="Rolling window size in days for the sum of fatalities.",
)

thresholds_raw = st.sidebar.text_input(
    "Escalation threshold(s) (comma-separated)",
    "25",
    help="Enter one or two thresholds, e.g. `25` or `25,1000`.",
)

persistence_days = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
    help="How many consecutive days above the threshold are required to count as an escalation.",
)

run_btn = st.sidebar.button("Generate plot")


# -----------------------------
# Main header
# -----------------------------
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

# -----------------------------
# Decide data source
# -----------------------------
data_source_label = None
df_raw: Optional[pd.DataFrame] = None

# 1) Demo file
if use_demo:
    data_source_label = "Built-in demo file: ukraine_sample.csv"
    try:
        df_raw = safe_read_demo()
    except Exception as e:
        st.error(str(e))
        st.stop()

# 2) Uploaded
elif uploaded is not None:
    data_source_label = "Uploaded CSV"
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

# 3) No upload/demo -> Google Drive dataset (for the map area)
else:
    data_source_label = "Google Drive dataset (auto for map)"
    st.info("No upload/demo selected — loading the Google Drive dataset for the map.")
    try:
        local_path = download_google_drive_csv_to_disk(GDRIVE_DIRECT_URL, str(GDRIVE_LOCAL_CACHE))
        df_raw = read_csv_any(local_path)
    except Exception as e:
        st.error(f"Data loading error: {e}")
        df_raw = None


# -----------------------------
# Interactive map area (beneath the intro)
# -----------------------------
st.subheader("Interactive map")
if df_raw is None:
    st.info("Load data (upload, demo, or Google Drive) to display the interactive map here.")
else:
    show_map(df_raw)


# -----------------------------
# Plot generation section
# -----------------------------
st.subheader("Escalation plot")

if df_raw is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**.")
    st.stop()

st.caption(f"Data source: {data_source_label}")

# If user hasn’t clicked Generate plot yet, keep the page non-empty
if not run_btn:
    st.info("Click **Generate plot** in the sidebar to generate the escalation plot.")
    st.stop()

# Parse thresholds
try:
    thresholds = parse_thresholds(thresholds_raw, max_n=2)
except Exception as e:
    st.error(str(e))
    st.stop()

# Build daily series + detect escalations
try:
    daily = build_daily_series(
        df=df_raw,
        country_col=country_col,
        date_col=date_col,
        fatalities_col=fatalities_col,
        country_name=country_name,
    )
except Exception as e:
    st.error(f"Column/parsing error: {e}")
    st.stop()

res = detect_escalations(
    daily=daily,
    rolling_window_days=int(rolling_window_days),
    thresholds=thresholds,
    persistence_days=int(persistence_days),
)

# Plot + summary layout
col_plot, col_summary = st.columns([3, 1], gap="large")

with col_plot:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    if res.daily.empty:
        ax.set_title(f"AEGIS Escalation Detection — {country_name} (no data)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling window fatalities")
    else:
        ax.plot(res.daily["date"], res.daily["rolling"], linewidth=1.6)
        for thr in thresholds:
            ax.axhline(float(thr), linestyle="--", linewidth=1)

        if not res.starts.empty:
            markers = ["o", "s", "D", "^"]
            for i, thr in enumerate(thresholds):
                pts = res.starts.loc[res.starts["threshold"] == float(thr)]
                if pts.empty:
                    continue
                ax.scatter(pts["date"], pts["rolling"], s=55, marker=markers[i % len(markers)], zorder=3)

        thr_label = ",".join([f"{t:g}" for t in thresholds])
        ax.set_title(
            f"AEGIS Escalation Detection — {country_name} "
            f"(rolling={int(rolling_window_days)}d, thresholds={thr_label})"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling window fatalities")
        fig.autofmt_xdate()

    st.pyplot(fig, clear_figure=True)

with col_summary:
    st.markdown("## Summary")
    st.write(f"Rows (daily): **{len(res.daily)}**")

    if not res.daily.empty:
        for thr in thresholds:
            above = (res.daily["rolling"] >= float(thr)).sum()
            # persistent for that threshold if computed
            pcol = f"persistent_{thr:g}"
            persistent_ct = int(res.daily[pcol].sum()) if pcol in res.daily.columns else 0
            st.write(f"Threshold = **{thr:g}**")
            st.write(f"Days above threshold: **{int(above)}**")
            st.write(f"Days persistent: **{persistent_ct}**")
            st.divider()

    st.write(f"Escalation starts detected: **{len(res.starts)}**")

    if not res.starts.empty:
        st.write("First escalation starts:")
        st.dataframe(res.starts.head(10), use_container_width=True)
    else:
        st.info("No escalation starts detected with the current settings.")
