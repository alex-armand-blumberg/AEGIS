import io
import csv
from pathlib import Path
from datetime import date

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
    """
    Try to detect delimiter (comma, tab, semicolon, pipe).
    Returns a delimiter or None if unsure.
    """
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return None


def safe_read_csv_bytes(data: bytes) -> pd.DataFrame:
    """
    Read CSV bytes robustly:
    - Detect & reject HTML (common when downloads are blocked)
    - Try multiple encodings
    - Sniff delimiter
    - Use python engine fallback + on_bad_lines skip
    """
    if not data or len(data) < 5:
        raise ValueError("File is empty or too small to be a CSV.")

    if _looks_like_html(data):
        raise ValueError(
            "This file looks like HTML (not a CSV). If this came from a host, it may be a blocked download page."
        )

    # Try decoding a small sample for delimiter sniffing
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
        # last resort: decode replacing errors
        sample_text = data[:200_000].decode("utf-8", errors="replace")
        used_encoding = "utf-8 (errors=replace)"

    delim = _sniff_delimiter(sample_text)

    # Try a "normal" read first (fast C engine)
    # If delimiter sniff failed, let pandas infer with sep=None on python engine.
    read_attempts = []

    if delim is not None:
        read_attempts.append(
            dict(encoding=used_encoding, sep=delim, engine="c", on_bad_lines="skip")
        )
        # Sometimes the C engine is stricter; try python too
        read_attempts.append(
            dict(encoding=used_encoding, sep=delim, engine="python", on_bad_lines="skip")
        )
    else:
        # Let pandas infer delimiter
        read_attempts.append(
            dict(encoding=used_encoding, sep=None, engine="python", on_bad_lines="skip")
        )

    # Also try alternate encodings if parse fails
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        if enc != used_encoding:
            if delim is not None:
                read_attempts.append(
                    dict(encoding=enc, sep=delim, engine="python", on_bad_lines="skip")
                )
            else:
                read_attempts.append(
                    dict(encoding=enc, sep=None, engine="python", on_bad_lines="skip")
                )

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
# World dataset (HF) download & map aggregation
# =============================
@st.cache_data(show_spinner=False)
def download_hf_world_csv(url: str) -> bytes:
    """
    Download the HF CSV bytes once (cached by Streamlit).
    """
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
    """
    Efficient-enough aggregation:
    - Read only needed columns
    - Convert date, filter range
    - Group by country sum fatalities
    """
    # Try a direct pandas read with minimal cols; if that fails, fall back to safe reader then subset.
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
            raise ValueError(f"HF dataset missing required columns: {missing}. Found: {list(df_full.columns)[:20]} ...")
        df = df_full[[country_col, date_col, fatalities_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # inclusive date range
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
        vals = vals[:2]  # keep UI promise: up to 2
    return vals


def build_daily_series(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str, country_name: str) -> pd.DataFrame:
    if country_col not in df.columns:
        raise ValueError(f"Missing column '{country_col}'")
    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'")
    if fatalities_col not in df.columns:
        raise ValueError(f"Missing column '{fatalities_col}'")

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

        # streak counts consecutive Trues
        streak = above.astype(int).groupby((~above).cumsum()).cumsum()
        persistent = streak >= persistence
        start = persistent & (~persistent.shift(1).fillna(False))

        s = df.loc[start, ["date", "rolling"]].copy()
        s["threshold"] = threshold
        starts_all.append(s)

    starts = pd.concat(starts_all, ignore_index=True) if starts_all else pd.DataFrame(columns=["date", "rolling", "threshold"])
    return df, starts


# =============================
# UI: Title/header (keep your vibe)
# =============================
st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")


# =============================
# Sidebar: Video + Inputs
# =============================
st.sidebar.header("Inputs")

# Sidebar video (HTML makes it loop cleanly; Streamlit's st.video does not always loop seamlessly)
if SIDEBAR_VIDEO_PATH.exists():
    st.sidebar.markdown(
        f"""
        <video autoplay loop muted playsinline style="width:100%; border-radius:12px;">
          <source src="{SIDEBAR_VIDEO_PATH.name}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.caption("Tip: add 'logo1.mp4' next to app.py for the sidebar video.")

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

run_btn = st.sidebar.button("Generate plot")


# =============================
# Decide which dataset powers the plot
# =============================
plot_df = None
plot_source = None

if use_sample:
    st.info("Using built-in demo file: ukraine_sample.csv")
    if not UKRAINE_SAMPLE_PATH.exists():
        st.error("Demo file not found: ukraine_sample.csv must be in the repo next to app.py.")
        st.stop()
    try:
        plot_df = read_csv_path(str(UKRAINE_SAMPLE_PATH))
        plot_source = "demo"
    except Exception as e:
        st.error(f"Could not load ukraine_sample.csv. {e}")
        st.stop()
else:
    if uploaded is not None:
        try:
            plot_df = read_csv_uploaded(uploaded)
            plot_source = "upload"
        except Exception as e:
            st.error(f"Could not parse uploaded CSV. {e}")
            st.stop()


# =============================
# Interactive map section (shows even when plot isn't generated)
# =============================
st.subheader("Interactive map")
st.caption("Data source: HuggingFace hosted world dataset")

# Date range for map
# (Default: last 365 days ending today; easy + fast enough)
today = date.today()
default_start = date(today.year - 1, today.month, min(today.day, 28))
map_start, map_end = st.date_input(
    "Map date range",
    value=(default_start, today),
    help="Aggregates fatalities by country across this date range.",
)

if isinstance(map_start, (list, tuple)) and len(map_start) == 2:
    map_start, map_end = map_start[0], map_start[1]

if map_start > map_end:
    st.warning("Map date range invalid: start is after end.")
else:
    with st.spinner("Loading world dataset for the map (Hugging Face)…"):
        try:
            world_bytes = download_hf_world_csv(HF_WORLD_CSV_URL)
            # The HF file columns should match your GED columns: country/date_start/best
            map_agg = aggregate_country_fatalities_from_bytes(
                world_bytes,
                country_col="country",
                date_col="date_start",
                fatalities_col="best",
                start_d=map_start,
                end_d=map_end,
            )

            # Plotly choropleth (interactive)
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
            st.error(
                "Map load failed. "
                f"{e}"
            )


st.divider()


# =============================
# Escalation plot section
# =============================
st.subheader("Escalation plot")

if plot_df is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears above.")
    st.stop()

if not run_btn:
    st.info("Data loaded — now click **Generate plot**.")
    st.stop()

# Prepare + compute
try:
    thresholds = parse_thresholds(thresholds_raw)
    daily = build_daily_series(plot_df, country_col, date_col, fatalities_col, country_name)
    daily2, starts = detect_escalations(daily, int(rolling_window), thresholds, int(persistence))
except Exception as e:
    st.error(str(e))
    st.stop()

# Plot (matplotlib)
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

# Summary (right column)
left, right = st.columns([3, 1])

with right:
    st.subheader("Summary")
    st.write(f"Rows (daily): {len(daily2)}")

    for t in thresholds:
        days_above = int((daily2["rolling"] >= t).sum())
        st.write(f"Days above threshold {t}: {days_above}")

    st.write(f"Days persistent (>= {persistence}): {int((daily2['rolling'] >= min(thresholds)).sum())}")
    st.write(f"Escalation starts detected: {len(starts)}")

    if not starts.empty:
        st.write("First escalation starts:")
        st.dataframe(starts[["date", "rolling", "threshold"]].head(10), use_container_width=True)
