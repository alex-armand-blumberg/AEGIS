# app.py
# AEGIS — Escalation Detection Demo (Streamlit)
# - Upload your own CSV OR use built-in Ukraine demo CSV from repo
# - OR load the worldwide CSV from Google Drive (large file)
# - Supports 1–2 thresholds (comma-separated)
# - Shows rolling-window fatalities + escalation-start markers + summary table

import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Google Drive downloader (add `gdown` to requirements.txt)
import gdown


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

# Your Google Drive file (worldwide dataset)
WORLD_DRIVE_FILE_ID = "1lg3YUmyxb8aqXCLtnPgGXnoJb8pIAEGF"
WORLD_LOCAL_PATH = "world_data.csv"  # cached on Streamlit Cloud ephemeral disk

# Built-in demo file expected in repo root
UKRAINE_DEMO_PATH = Path("ukraine_sample.csv")


# -----------------------------
# Helpers
# -----------------------------
def parse_thresholds(raw: str) -> List[float]:
    """
    Accepts: "25" or "25,1000" etc. Returns up to 2 floats.
    """
    parts = [p.strip() for p in (raw or "").split(",") if p.strip() != ""]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            continue
    # Keep only first 2
    return vals[:2]


@st.cache_data(show_spinner=True)
def download_world_csv_from_drive(file_id: str, out_path: str) -> str:
    """
    Download a large CSV from Google Drive and return local file path.
    Cached so we don't re-download on every rerun.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(out_path):
        gdown.download(url, out_path, quiet=False)
    return out_path


@st.cache_data(show_spinner=True)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=True)
def read_csv_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_dataframe(
    uploaded_file,
    use_ukraine_demo: bool,
    use_world_drive: bool,
) -> pd.DataFrame:
    if use_world_drive:
        local_path = download_world_csv_from_drive(WORLD_DRIVE_FILE_ID, WORLD_LOCAL_PATH)
        return read_csv_path(local_path)

    if use_ukraine_demo:
        if not UKRAINE_DEMO_PATH.exists():
            raise FileNotFoundError(
                "Built-in demo file not found. Expected ukraine_sample.csv in the repo root."
            )
        return read_csv_path(str(UKRAINE_DEMO_PATH))

    if uploaded_file is None:
        raise ValueError("No CSV provided.")
    return read_csv_bytes(uploaded_file.getvalue())


def build_daily_series(
    df: pd.DataFrame,
    country_col: str,
    country_value: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
    """
    Filters to country, converts date + fatalities, aggregates to daily totals.
    Returns columns: [date, fatalities]
    """
    missing = [c for c in [country_col, date_col, fatalities_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in CSV: {missing}")

    sub = df[df[country_col].astype(str) == str(country_value)].copy()
    if sub.empty:
        raise ValueError(
            f"No rows found for {country_value!r} in column {country_col!r}. "
            "Make sure the name matches exactly."
        )

    # Parse date + fatalities robustly
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)
    sub[fatalities_col] = sub[fatalities_col].clip(lower=0)

    # daily aggregation
    daily = (
        sub.groupby(sub[date_col].dt.date, as_index=False)[fatalities_col]
        .sum()
        .rename(columns={fatalities_col: "fatalities"})
    )
    daily["date"] = pd.to_datetime(daily[date_col].astype(str), errors="coerce")
    daily = daily.drop(columns=[date_col]).sort_values("date")

    # Make a complete daily index (fills missing days with 0)
    full = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full.merge(daily, on="date", how="left")
    daily["fatalities"] = daily["fatalities"].fillna(0)

    return daily


def compute_escalation(
    daily: pd.DataFrame,
    rolling_days: int,
    threshold: float,
    persistence_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds rolling window + escalation flags.
    Returns:
      - df with columns [date, fatalities, rolling, above, persistent, escalation_start]
      - escalation starts table [date, rolling, threshold]
    """
    df = daily.copy()
    df["rolling"] = df["fatalities"].rolling(window=rolling_days, min_periods=rolling_days).sum()
    df["rolling"] = df["rolling"].fillna(0)

    df["above"] = df["rolling"] >= threshold

    # persistent: TRUE if last `persistence_days` are all above threshold
    # (requires consecutive days above threshold)
    if persistence_days <= 1:
        df["persistent"] = df["above"]
    else:
        df["persistent"] = (
            df["above"]
            .rolling(window=persistence_days, min_periods=persistence_days)
            .apply(lambda x: 1.0 if x.all() else 0.0, raw=False)
            .fillna(0)
            .astype(bool)
        )

    df["escalation_start"] = df["persistent"] & (~df["persistent"].shift(1).fillna(False))

    starts = df.loc[df["escalation_start"], ["date", "rolling"]].copy()
    starts["threshold"] = threshold

    return df, starts


def make_plot(
    df: pd.DataFrame,
    starts_all: pd.DataFrame,
    country: str,
    rolling_days: int,
    thresholds: List[float],
):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["rolling"], linewidth=2)
    ax.set_title(
        f"AEGIS Escalation Detection — {country} (rolling={rolling_days}d, thresholds={','.join(str(t) for t in thresholds)})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling window fatalities")

    # Draw threshold lines + markers
    for t in thresholds:
        ax.axhline(t, linestyle="--", linewidth=1)

    if not starts_all.empty:
        ax.scatter(starts_all["date"], starts_all["rolling"], s=70)

    fig.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

st.sidebar.header("Inputs")

# Demo / data source toggles
use_ukraine_demo = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help="Loads ukraine_sample.csv from this repo (small demo file).",
)

use_world_drive = st.sidebar.checkbox(
    "Use worldwide dataset from Google Drive (large)",
    value=False,
    help="Downloads the large CSV from Google Drive (cached). Turn this off if you want to upload your own file.",
)

# If both checked, worldwide wins
if use_ukraine_demo and use_world_drive:
    st.sidebar.info("Both toggles are on. Using the worldwide Google Drive dataset.")
    use_ukraine_demo = False

uploaded = None
if not use_ukraine_demo and not use_world_drive:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

country_col = st.sidebar.text_input(
    "Country column",
    "country",
    help="Name of the column that contains the country name for each row (e.g., 'country').",
)

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Must match the dataset exactly (examples: 'Ukraine', 'Mexico', 'Syria').",
)

date_col = st.sidebar.text_input(
    "Date column",
    "date_start",
    help="Name of the column that contains dates (e.g., 'date_start').",
)

fatalities_col = st.sidebar.text_input(
    "Fatalities column",
    "best",
    help="Name of the column with fatalities (numeric). For some datasets this is 'best'.",
)

rolling_days = st.sidebar.number_input(
    "Rolling window (days)",
    min_value=1,
    max_value=365,
    value=30,
    step=1,
)

thresholds_raw = st.sidebar.text_input(
    "Escalation threshold(s) (comma-separated)",
    "25",
    help="Enter one or two thresholds, e.g. '25' or '25,1000'.",
)

persistence_days = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
    help="Escalation start is flagged when the rolling value stays above threshold for this many consecutive days.",
)

run_btn = st.sidebar.button("Generate plot")

# Placeholder when nothing is run
if not run_btn:
    st.info("Upload a CSV (or enable a demo), then click **Generate plot**.")
    st.stop()

# -----------------------------
# Run analysis
# -----------------------------
try:
    thresholds = parse_thresholds(thresholds_raw)
    if len(thresholds) == 0:
        st.warning("Please enter at least one valid threshold (e.g., 25).")
        st.stop()

    df_raw = load_dataframe(
        uploaded_file=uploaded,
        use_ukraine_demo=use_ukraine_demo,
        use_world_drive=use_world_drive,
    )

    daily = build_daily_series(
        df=df_raw,
        country_col=country_col,
        country_value=country_name,
        date_col=date_col,
        fatalities_col=fatalities_col,
    )

    # Compute for each threshold (1–2)
    df_base = None
    starts_list = []
    for t in thresholds:
        df_t, starts_t = compute_escalation(
            daily=daily,
            rolling_days=int(rolling_days),
            threshold=float(t),
            persistence_days=int(persistence_days),
        )
        # use the first computed df as the plot line (same rolling regardless of threshold)
        if df_base is None:
            df_base = df_t
        starts_list.append(starts_t)

    starts_all = pd.concat(starts_list, ignore_index=True) if starts_list else pd.DataFrame()

    # Layout: plot + summary
    left, right = st.columns([2, 1], gap="large")

    with left:
        fig = make_plot(
            df=df_base,
            starts_all=starts_all,
            country=country_name,
            rolling_days=int(rolling_days),
            thresholds=thresholds,
        )
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Summary")
        st.write(f"Rows (daily): **{len(daily)}**")

        for t in thresholds:
            # for each threshold, compute counts from df_t by recomputing persistent quickly
            df_t, starts_t = compute_escalation(
                daily=daily,
                rolling_days=int(rolling_days),
                threshold=float(t),
                persistence_days=int(persistence_days),
            )
            st.write(f"**Threshold = {t}**")
            st.write(f"Days above threshold: **{int(df_t['above'].sum())}**")
            st.write(f"Days persistent: **{int(df_t['persistent'].sum())}**")
            st.write(f"Escalation starts detected: **{len(starts_t)}**")
            st.write("---")

        st.subheader("First escalation starts")
        if starts_all.empty:
            st.write("None detected with the current settings.")
        else:
            show = starts_all.sort_values("date").head(15).copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show, use_container_width=True)

except Exception as e:
    st.error(str(e))
    st.stop()
