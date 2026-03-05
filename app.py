import io
import csv
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import requests


# =============================
# Config
# =============================
st.set_page_config(page_title="AEGIS — Escalation Detection Demo", layout="wide")

APP_DIR = Path(__file__).resolve().parent

# Your Hugging Face direct file URL (world dataset)
HF_WORLD_CSV_URL = "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"

# Demo file must exist in repo next to app.py
UKRAINE_SAMPLE_PATH = APP_DIR / "ukraine_sample.csv"

# Optional: sidebar header video file (put next to app.py)
SIDEBAR_VIDEO_PATH = APP_DIR / "logo1.mp4"  # rename if yours differs


# =============================
# Robust CSV reading helpers
# =============================
def _looks_like_html(b: bytes) -> bool:
    head = b[:2048].lower()
    return b"<html" in head or b"<!doctype html" in head


def _sniff_delimiter(text_sample: str) -> str | None:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return None


def safe_read_csv_bytes(data: bytes) -> pd.DataFrame:
    if not data or len(data) < 5:
        raise ValueError("File is empty or too small to be a CSV.")

    if _looks_like_html(data):
        raise ValueError(
            "This file looks like HTML (not a CSV). If this came from a host, it may be a blocked download page."
        )

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    sample_text = None
    used_encoding = None
    for enc in encodings_to_try:
        try:
            sample_text = data[:200_000].decode(enc)
            used_encoding = enc
            break
        except Exception:
            continue

    if sample_text is None:
        sample_text = data[:200_000].decode("utf-8", errors="replace")
        used_encoding = "utf-8 (errors=replace)"

    delim = _sniff_delimiter(sample_text)

    read_attempts = []
    if delim is not None:
        read_attempts.append(dict(encoding=used_encoding, sep=delim, engine="c", on_bad_lines="skip"))
        read_attempts.append(dict(encoding=used_encoding, sep=delim, engine="python", on_bad_lines="skip"))
    else:
        read_attempts.append(dict(encoding=used_encoding, sep=None, engine="python", on_bad_lines="skip"))

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        if enc != used_encoding:
            if delim is not None:
                read_attempts.append(dict(encoding=enc, sep=delim, engine="python", on_bad_lines="skip"))
            else:
                read_attempts.append(dict(encoding=enc, sep=None, engine="python", on_bad_lines="skip"))

    last_err = None
    for kwargs in read_attempts:
        try:
            return pd.read_csv(io.BytesIO(data), **kwargs)
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not parse CSV after multiple attempts. Last error: {last_err}")


@st.cache_data(show_spinner=False)
def read_csv_path(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    b = p.read_bytes()
    return safe_read_csv_bytes(b)


@st.cache_data(show_spinner=False)
def read_csv_uploaded(uploaded_file) -> pd.DataFrame:
    b = uploaded_file.getvalue()
    return safe_read_csv_bytes(b)


# =============================
# Column auto-detection (fixes demo missing 'country')
# =============================
def _normalize_col(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch == "_")


def guess_columns(df: pd.DataFrame) -> dict:
    """
    Return best guesses for country/date/fatalities using common aliases.
    """
    cols = list(df.columns)
    norm = {c: _normalize_col(str(c)) for c in cols}

    country_aliases = {"country", "countryname", "cntry", "location", "state", "iso3", "iso"}
    date_aliases = {"date", "datestart", "date_start", "startdate", "eventdate", "time", "timestamp"}
    fat_aliases = {"best", "fatalities", "deaths", "killed", "fatality", "totfatalities"}

    def find_one(aliases: set[str]) -> str | None:
        # exact normalized match first
        for c, n in norm.items():
            if n in aliases:
                return c
        # contains match second (e.g., "date_start_utc")
        for c, n in norm.items():
            for a in aliases:
                if a in n:
                    return c
        return None

    return {
        "country": find_one(country_aliases),
        "date": find_one(date_aliases),
        "fatalities": find_one(fat_aliases),
    }


def ensure_country_column(df: pd.DataFrame, country_col: str | None, fill_country: str) -> tuple[pd.DataFrame, str]:
    """
    If country_col missing/None, create a 'country' column filled with fill_country.
    """
    out = df.copy()
    if country_col is None or country_col not in out.columns:
        out["country"] = fill_country
        return out, "country"
    return out, country_col


# =============================
# World dataset (HF) download & map aggregation
# =============================
@st.cache_data(show_spinner=False)
def download_hf_world_csv(url: str) -> bytes:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    return resp.content


def aggregate_country_fatalities_from_bytes(
    csv_bytes: bytes,
    country_col: str,
    date_col: str,
    fatalities_col: str,
    start_d: date,
    end_d: date,
) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            usecols=[country_col, date_col, fatalities_col],
            low_memory=False,
        )
    except Exception:
        df_full = safe_read_csv_bytes(csv_bytes)
        missing = [c for c in [country_col, date_col, fatalities_col] if c not in df_full.columns]
        if missing:
            raise ValueError(
                f"HF dataset missing required columns: {missing}. Found: {list(df_full.columns)[:20]} ..."
            )
        df = df_full[[country_col, date_col, fatalities_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    start_ts = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)]

    df[fatalities_col] = pd.to_numeric(df[fatalities_col], errors="coerce").fillna(0)

    out = (
        df.groupby(country_col, as_index=False)[fatalities_col]
        .sum()
        .rename(columns={country_col: "country", fatalities_col: "fatalities"})
    )
    return out


# =============================
# Escalation logic
# =============================
def parse_thresholds(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) == 0:
        raise ValueError("Please enter at least one threshold (e.g., 25 or 25,1000).")
    if len(vals) > 2:
        vals = vals[:2]
    return vals


def build_daily_series(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str, country_name: str) -> pd.DataFrame:
    # If user-provided columns don't exist, try auto-detect
    if country_col not in df.columns or date_col not in df.columns or fatalities_col not in df.columns:
        guessed = guess_columns(df)

        # If Ukraine-only sample has no country column, add it
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


def detect_escalations(daily: pd.DataFrame, rolling_window: int, thresholds: list[float], persistence: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = daily.copy()
    df["rolling"] = df["fatalities"].rolling(window=int(rolling_window), min_periods=1).sum()

    starts_all = []
    for threshold in thresholds:
        above = df["rolling"] >= threshold
        streak = above.astype(int).groupby((~above).cumsum()).cumsum()
        persistent = streak >= persistence
        start = persistent & (~persistent.shift(1).fillna(False))

        s = df.loc[start, ["date", "rolling"]].copy()
        s["threshold"] = threshold
        starts_all.append(s)

    starts = pd.concat(starts_all, ignore_index=True) if starts_all else pd.DataFrame(columns=["date", "rolling", "threshold"])
    return df, starts


# =============================
# Header
# =============================
st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")


# =============================
# Sidebar
# =============================
st.sidebar.header("Inputs")

# Sidebar video (looping HTML)
if SIDEBAR_VIDEO_PATH.exists():
    st.sidebar.markdown(
        f"""
        <video autoplay loop muted playsinline style="width:100%; border-radius:12px;">
          <source src="{SIDEBAR_VIDEO_PATH.name}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help="Uses the local repo file: ukraine_sample.csv (must be next to app.py).",
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload your conflict dataset. Must include country/date/fatalities columns.",
    )

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Name must match the dataset exactly (case/spaces).",
)

country_col = st.sidebar.text_input(
    "Country column",
    "country",
    help="Column that contains country names.",
)

date_col = st.sidebar.text_input(
    "Date column",
    "date_start",
    help="Column that contains dates/timestamps (e.g., 2022-02-24 00:00:00.000).",
)

fatalities_col = st.sidebar.text_input(
    "Fatalities column",
    "best",
    help="Column that contains fatalities counts (numeric).",
)

rolling_window = st.sidebar.number_input(
    "Rolling window (days)",
    min_value=1,
    max_value=365,
    value=30,
    step=1,
)

thresholds_raw = st.sidebar.text_input(
    "Escalation threshold(s) (comma-separated, max 2)",
    "25,1000",
    help="Examples: 25  OR  25,1000",
)

persistence = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=7,
    step=1,
)

# ---- Map controls (requested behavior) ----
show_map = st.sidebar.checkbox(
    "Show interactive map",
    value=True,  # starts ON
    help="Turn off if you want the page to load faster.",
)

# default map range: last 365 days ending today (automatic)
today = date.today()
default_map_start = today - timedelta(days=365)
map_start, map_end = st.sidebar.date_input(
    "Map date range",
    value=(default_map_start, today),
    help="Defaults automatically; you can edit it here if you want.",
)

run_btn = st.sidebar.button("Generate plot")


# =============================
# Load dataset for plot (upload or demo)
# =============================
plot_df = None

if use_sample:
    st.info("Using built-in demo file: ukraine_sample.csv")
    if not UKRAINE_SAMPLE_PATH.exists():
        st.error("Demo file not found: ukraine_sample.csv must be in the repo next to app.py.")
        st.stop()

    df_demo = read_csv_path(str(UKRAINE_SAMPLE_PATH))

    # Auto-detect columns / fix missing country column
    guessed = guess_columns(df_demo)
    df_demo, demo_country_col = ensure_country_column(df_demo, guessed["country"], fill_country=country_name)

    demo_date_col = guessed["date"] or date_col
    demo_fat_col = guessed["fatalities"] or fatalities_col

    # Override sidebar defaults ONLY for the demo dataset if needed
    # (keeps your setup smooth: demo just works)
    country_col = demo_country_col
    date_col = demo_date_col
    fatalities_col = demo_fat_col

    plot_df = df_demo

else:
    if uploaded is not None:
        plot_df = read_csv_uploaded(uploaded)


# =============================
# Interactive Map (under header; can be turned off)
# =============================
st.subheader("Interactive map")
st.caption("Data source: HuggingFace hosted world dataset")

if not show_map:
    st.info("Map is turned off in the sidebar.")
else:
    # Streamlit date_input can sometimes return a single date; normalize
    if isinstance(map_start, (list, tuple)) and len(map_start) == 2:
        map_start, map_end = map_start[0], map_start[1]

    if map_start > map_end:
        st.warning("Map date range invalid: start is after end.")
    else:
        with st.spinner("Loading world dataset for the map (Hugging Face)…"):
            try:
                world_bytes = download_hf_world_csv(HF_WORLD_CSV_URL)

                map_agg = aggregate_country_fatalities_from_bytes(
                    world_bytes,
                    country_col="country",
                    date_col="date_start",
                    fatalities_col="best",
                    start_d=map_start,
                    end_d=map_end,
                )

                fig_map = px.choropleth(
                    map_agg,
                    locations="country",
                    locationmode="country names",
                    color="fatalities",
                    hover_name="country",
                    color_continuous_scale="Blues",
                    title="Fatalities by country (selected date range)",
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=60, b=0))
                st.plotly_chart(fig_map, use_container_width=True)

            except Exception as e:
                st.error(f"Map load failed. {e}")

st.divider()


# =============================
# Escalation plot
# =============================
st.subheader("Escalation plot")

if plot_df is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears above.")
    st.stop()

if not run_btn:
    st.info("Data loaded — now click **Generate plot**.")
    st.stop()

try:
    thresholds = parse_thresholds(thresholds_raw)
    daily = build_daily_series(plot_df, country_col, date_col, fatalities_col, country_name)
    daily2, starts = detect_escalations(daily, int(rolling_window), thresholds, int(persistence))
except Exception as e:
    st.error(str(e))
    st.stop()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily2["date"], daily2["rolling"])

for t in thresholds:
    ax.axhline(t, linestyle="--")

if not starts.empty:
    ax.scatter(starts["date"], starts["rolling"], s=80)

ax.set_title(f"AEGIS Escalation Detection — {country_name} (rolling={rolling_window}d, thresholds={thresholds})")
ax.set_xlabel("Date")
ax.set_ylabel("Rolling fatalities")
st.pyplot(fig)

_, right = st.columns([3, 1])
with right:
    st.subheader("Summary")
    st.write(f"Rows (daily): {len(daily2)}")
    for t in thresholds:
        st.write(f"Days above threshold {t}: {int((daily2['rolling'] >= t).sum())}")
    st.write(f"Escalation starts detected: {len(starts)}")
    if not starts.empty:
        st.write("First escalation starts:")
        st.dataframe(starts[["date", "rolling", "threshold"]].head(10), use_container_width=True)
