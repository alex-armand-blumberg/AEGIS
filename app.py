import io
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

# HuggingFace-hosted (large) world dataset (your link)
HF_WORLD_CSV_URL = (
    "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"
)

# Optional built-in Ukraine demo (small file stored in your repo)
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")

# Optional sidebar media (put these files in your repo if you want them to show)
SIDEBAR_VIDEO_PATH = Path("logo1.mp4")  # your looping-ish banner video
SIDEBAR_LOGO_PATH = Path("logo.png")    # optional static logo if you ever add one


# -----------------------------
# HELPERS
# -----------------------------
def _safe_read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV from raw bytes with a couple of fallbacks."""
    # Try utf-8 first, then latin-1
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    # Last attempt: let pandas guess
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def read_csv_path(path_str: str) -> pd.DataFrame:
    with open(path_str, "rb") as f:
        b = f.read()
    return _safe_read_csv_bytes(b)


@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    # pandas can read HTTPS URLs directly
    return pd.read_csv(url)


@st.cache_data(show_spinner=False)
def read_uploaded(uploaded_file) -> pd.DataFrame:
    b = uploaded_file.getvalue()
    return _safe_read_csv_bytes(b)


def parse_thresholds(raw: str) -> List[float]:
    """
    Parse 1–2 thresholds from user input, e.g.:
      "25" or "25,1000" or "25 1000"
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Enter at least 1 threshold (e.g., 25 or 25,1000).")

    parts = re.split(r"[,\s]+", raw)
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) == 0:
        raise ValueError("Could not parse thresholds. Try: 25 or 25,1000")
    if len(vals) > 2:
        vals = vals[:2]
    vals = sorted(vals)
    return vals


def ensure_columns(df: pd.DataFrame, country_col: str, date_col: str, fat_col: str) -> pd.DataFrame:
    missing = [c for c in [country_col, date_col, fat_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Available columns: {list(df.columns)[:50]}")

    out = df[[country_col, date_col, fat_col]].copy()
    out.columns = ["country", "date", "fatalities"]

    # Parse date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    # Parse fatalities numeric
    out["fatalities"] = pd.to_numeric(out["fatalities"], errors="coerce").fillna(0.0)

    # Normalize country
    out["country"] = out["country"].astype(str).str.strip()

    return out


def compute_daily_country(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily totals per country (date, country, fatalities)."""
    daily = (
        df_norm.groupby(["country", "date"], as_index=False)["fatalities"]
        .sum()
        .sort_values(["country", "date"])
    )
    return daily


def compute_rolling_for_country(
    daily_country: pd.DataFrame,
    country_name: str,
    rolling_window_days: int
) -> pd.DataFrame:
    """
    For a single country, build a daily series with rolling sum.
    """
    sub = daily_country[daily_country["country"] == country_name].copy()
    if sub.empty:
        return sub

    # Ensure continuous daily index
    sub = sub.set_index("date").sort_index()
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
    sub = sub.reindex(full_idx)
    sub.index.name = "date"

    # Fill missing days with 0 fatalities
    sub["fatalities"] = sub["fatalities"].fillna(0.0)
    sub["country"] = country_name

    # Rolling sum
    win = int(max(1, rolling_window_days))
    sub["rolling"] = sub["fatalities"].rolling(window=win, min_periods=win).sum()

    sub = sub.reset_index()
    return sub


def detect_escalation_starts(
    series_df: pd.DataFrame,
    threshold: float,
    persistence_days: int
) -> pd.DataFrame:
    """
    Mark escalation "start" days: when rolling crosses above threshold and stays above
    for >= persistence_days.
    """
    if series_df.empty:
        return pd.DataFrame(columns=["date", "rolling", "threshold"])

    s = series_df.copy()
    s["above"] = s["rolling"] >= threshold
    p = int(max(1, persistence_days))

    # consecutive above-threshold count
    s["above_run"] = s["above"].groupby((~s["above"]).cumsum()).cumcount() + 1
    s.loc[~s["above"], "above_run"] = 0

    # escalation start = day when above_run hits p (first day of a valid persistent run)
    starts = s[s["above_run"] == p][["date", "rolling"]].copy()
    starts["threshold"] = threshold

    # Optionally, you can label the start as the FIRST day of the run instead of day p:
    # shift back p-1 days
    starts["date"] = starts["date"] - pd.to_timedelta(p - 1, unit="D")

    # Drop duplicates (two thresholds might shift onto same start day etc.)
    starts = starts.drop_duplicates(subset=["date", "threshold"]).sort_values("date")
    return starts


def plot_escalation(
    series_df: pd.DataFrame,
    country_name: str,
    thresholds: List[float],
    starts_all: pd.DataFrame,
    rolling_window_days: int
):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series_df["date"], series_df["rolling"], linewidth=1.5)
    ax.set_title(f"AEGIS Escalation Detection — {country_name} (rolling={rolling_window_days}d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling window fatalities")

    # Threshold lines
    for t in thresholds:
        ax.axhline(t, linestyle="--", linewidth=1.2)

    # Start markers
    if not starts_all.empty:
        ax.scatter(starts_all["date"], starts_all["rolling"], s=40)

    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def build_choropleth(
    daily_country: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
):
    """
    Interactive world map (choropleth) of fatalities by country for a date range.
    Works even if there are no lat/lon columns.
    """
    df = daily_country.copy()

    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]

    if df.empty:
        st.info("No data available for that date range (for the map).")
        return

    agg = df.groupby("country", as_index=False)["fatalities"].sum()
    agg = agg[agg["fatalities"] > 0].sort_values("fatalities", ascending=False)

    # Plotly choropleth using country names (best-effort; some names may not match)
    fig = px.choropleth(
        agg,
        locations="country",
        locationmode="country names",
        color="fatalities",
        hover_name="country",
        hover_data={"fatalities": True, "country": False},
        projection="natural earth",
        title="Fatalities by country (selected date range)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# SIDEBAR (branding + inputs)
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    # Sidebar media (optional)
    if SIDEBAR_VIDEO_PATH.exists():
        st.video(str(SIDEBAR_VIDEO_PATH))
    elif SIDEBAR_LOGO_PATH.exists():
        st.image(str(SIDEBAR_LOGO_PATH), use_container_width=True)

    use_demo = st.checkbox(
        "Use built-in Ukraine example (recommended demo)",
        value=False,
        help="Uses a small file in the repo: ukraine_sample.csv",
    )

    uploaded = None
    if not use_demo:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Upload a CSV with at least: country, date, fatalities columns.",
        )

    st.divider()
    st.subheader("Columns")
    country_col = st.text_input(
        "Country column",
        value="country",
        help="Name of the column that contains country names (e.g., 'country').",
    )
    date_col = st.text_input(
        "Date column",
        value="date_start",
        help="Name of the column that contains dates (e.g., 'date_start').",
    )
    fatalities_col = st.text_input(
        "Fatalities column",
        value="best",
        help="Name of the column that contains fatalities (e.g., 'best').",
    )

    st.divider()
    st.subheader("Escalation settings")

    rolling_window = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Rolling sum window in days (e.g., 30).",
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        value="25,1000",
        help="Enter 1 or 2 thresholds, e.g. 25 or 25,1000.",
    )

    persistence = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1,
        max_value=60,
        value=3,
        step=1,
        help="How many consecutive days above the threshold count as an escalation.",
    )

    st.divider()
    st.subheader("Map settings")
    use_hf_world_for_map = st.checkbox(
        "Use hosted world dataset (HuggingFace) for the map when no upload/demo is selected",
        value=True,
        help="Loads the large worldwide dataset from HuggingFace for the interactive map.",
    )

    # Date range for the map
    map_date_mode = st.selectbox(
        "Map date range",
        options=["All time", "Custom range (pick dates)"],
        index=0,
        help="Controls which dates are included in the choropleth map.",
    )
    map_start = None
    map_end = None
    if map_date_mode.startswith("Custom"):
        map_start = st.date_input("Map start date")
        map_end = st.date_input("Map end date")


# -----------------------------
# MAIN UI (header)
# -----------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">AEGIS — Escalation Detection Demo</h1>
    <p style="margin-top:0;color:#cfcfcf;">
      Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.
    </p>
    """,
    unsafe_allow_html=True,
)

# Info banner
st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears below.")


# -----------------------------
# LOAD DATA (upload/demo) for plot
# -----------------------------
def load_primary_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Returns (df_raw, source_label)
    - df_raw is the dataset used for the escalation plot (upload or demo).
    """
    if use_demo:
        if not UKRAINE_SAMPLE_PATH.exists():
            return None, "Demo file missing (ukraine_sample.csv not found in repo)."
        df_demo = read_csv_path(str(UKRAINE_SAMPLE_PATH))
        return df_demo, "Built-in demo (ukraine_sample.csv)"
    if uploaded is not None:
        df_up = read_uploaded(uploaded)
        return df_up, f"Uploaded file ({uploaded.name})"
    return None, "No upload/demo selected"


df_raw_plot, plot_source = load_primary_dataset()


# -----------------------------
# MAP SECTION (shows even when plot not generated)
# -----------------------------
st.markdown("## Interactive map")

# Decide which dataset powers the map:
# Priority: upload/demo (if present) else HF world dataset (if enabled)
df_raw_map = None
map_source = None

if df_raw_plot is not None:
    df_raw_map = df_raw_plot
    map_source = plot_source
elif use_hf_world_for_map:
    map_source = "HuggingFace hosted world dataset"
    with st.spinner("Loading worldwide dataset for the map (cached after first load)…"):
        try:
            df_raw_map = read_csv_url(HF_WORLD_CSV_URL)
        except Exception as e:
            df_raw_map = None
            st.error(
                "Could not load the HuggingFace dataset. "
                "If this persists, confirm the link is correct and publicly accessible.\n\n"
                f"Error: {e}"
            )

if df_raw_map is None:
    st.info("Load data (upload or demo) to display the interactive map here.")
else:
    try:
        df_norm_map = ensure_columns(df_raw_map, country_col, date_col, fatalities_col)
        daily_map = compute_daily_country(df_norm_map)

        # Convert sidebar date inputs -> pd.Timestamp
        map_start_ts = pd.to_datetime(map_start) if map_start is not None else None
        map_end_ts = pd.to_datetime(map_end) if map_end is not None else None

        st.caption(f"Data source: {map_source}")
        build_choropleth(daily_map, map_start_ts, map_end_ts)

    except Exception as e:
        st.error(f"Map error: {e}")


# -----------------------------
# PLOT SECTION (only when user clicks)
# -----------------------------
st.markdown("## Escalation plot")
run_btn = st.button("Generate plot")

if not run_btn:
    st.info("Click **Generate plot** to compute escalation and show the chart.")
    st.stop()

if df_raw_plot is None:
    st.error("No dataset selected for the plot. Upload a CSV or enable the built-in demo.")
    st.stop()

# Parse thresholds
try:
    thresholds = parse_thresholds(thresholds_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# Normalize and compute
try:
    df_norm = ensure_columns(df_raw_plot, country_col, date_col, fatalities_col)
    daily = compute_daily_country(df_norm)
except Exception as e:
    st.error(f"Column/parsing error: {e}")
    st.stop()

# Country selection
countries = sorted(daily["country"].dropna().unique().tolist())
default_country = "Ukraine" if "Ukraine" in countries else (countries[0] if countries else "")
country_name = st.selectbox(
    "Country (exact match)",
    options=countries,
    index=countries.index(default_country) if default_country in countries else 0,
)

series = compute_rolling_for_country(daily, country_name, int(rolling_window))
if series.empty:
    st.warning(f"No rows found for country='{country_name}'. Check the country name and column settings.")
    st.stop()

# Detect escalation starts for each threshold and combine
starts_list = []
for t in thresholds:
    starts_list.append(detect_escalation_starts(series, float(t), int(persistence)))
starts_all = pd.concat(starts_list, ignore_index=True) if len(starts_list) else pd.DataFrame()

# Summary
colA, colB = st.columns([2, 1], gap="large")
with colA:
    plot_escalation(series, country_name, thresholds, starts_all, int(rolling_window))

with colB:
    st.subheader("Summary")
    # Note: "rolling" has NaNs for the first (window-1) days
    valid_rolling = series.dropna(subset=["rolling"]).copy()

    st.write(f"**Data source:** {plot_source}")
    st.write(f"**Rows (daily):** {len(series):,}")
    st.write(f"**Thresholds:** {', '.join(str(t) for t in thresholds)}")
    st.write(f"**Persistence:** {int(persistence)} day(s)")
    st.write(f"**Rolling window:** {int(rolling_window)} day(s)")

    for t in thresholds:
        days_above = int((valid_rolling["rolling"] >= t).sum())
        st.write(f"**Days above {t}:** {days_above:,}")

    st.write(f"**Escalation starts detected:** {len(starts_all):,}")

    if not starts_all.empty:
        st.caption("First escalation starts:")
        st.dataframe(starts_all.sort_values(["threshold", "date"]).head(25), use_container_width=True)
    else:
        st.caption("No escalation starts detected for the current settings.")
