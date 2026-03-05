import io
import os
from pathlib import Path
from datetime import date
import base64

import numpy as np
import pandas as pd
import streamlit as st

# Optional: used for the interactive map (recommended).
# If plotly isn't installed, the app will still run but will show a friendly message.
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False



import time
import xml.etree.ElementTree as ET

import requests
import streamlit as st


@st.cache_data(ttl=900)  # refresh every 15 minutes
def fetch_rss_items(rss_url: str, max_items: int = 6):
    r = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    root = ET.fromstring(r.content)

    items = []
    # RSS 2.0 usually: <rss><channel><item>...
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = ""
        src = item.find("source")
        if src is not None and src.text:
            source = src.text.strip()

        if title and link:
            items.append({"title": title, "link": link, "pub_date": pub_date, "source": source})

    return items


import time
import xml.etree.ElementTree as ET

import requests
import streamlit as st


@st.cache_data(ttl=900)  # refresh every 15 minutes
def fetch_rss_items(rss_url: str, max_items: int = 6):
    r = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    root = ET.fromstring(r.content)

    items = []
    # RSS 2.0 usually: <rss><channel><item>...
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = ""
        src = item.find("source")
        if src is not None and src.text:
            source = src.text.strip()

        if title and link:
            items.append({"title": title, "link": link, "pub_date": pub_date, "source": source})

    return items


def render_news():
    st.markdown("## Current Conflict News")

    # Google News RSS query (easy + reliable)
    rss_url = (
        "https://news.google.com/rss/search?"
        "q=(war+OR+conflict+OR+invasion+OR+insurgency)+when:7d&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        items = fetch_rss_items(rss_url, max_items=6)
        if not items:
            st.info("No items returned from the news feed right now.")
            return

        for it in items:
            # Simple “card” look
            st.markdown(
                f"""
                <div style="
                    padding:14px 16px;
                    border:1px solid rgba(255,255,255,0.10);
                    border-radius:14px;
                    margin-bottom:10px;
                    background: rgba(255,255,255,0.03);
                ">
                    <div style="font-size:18px; font-weight:700; line-height:1.25;">
                        <a href="{it['link']}" target="_blank" style="text-decoration:none;">
                            {it['title']}
                        </a>
                    </div>
                    <div style="opacity:0.7; margin-top:6px; font-size:13px;">
                        {it['source'] or ""} {("• " + it['pub_date']) if it['pub_date'] else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("Updates automatically ~every 15 minutes (RSS).")

    except Exception as e:
        st.warning(f"Live news feed failed to load: {e}")



# ----------------------------
# Config
# ----------------------------


st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="ZoomedLogo.png",
    layout="wide"
)

# ----------------------------
# Current conflict news
# ----------------------------

with st.expander("Current Conflict News", expanded=False):
    render_news()

# ----------------------------
# Dataset paths
# ----------------------------

HF_WORLD_CSV_URL = "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")  # must exist in repo root

# ----------------------------
# Robust CSV loading helpers
# ----------------------------
def _read_csv_attempt(data: bytes, *, encoding: str, sep):
    """
    One attempt to parse bytes into a DataFrame.
    sep can be None (auto) or an explicit delimiter.
    """
    bio = io.BytesIO(data)

    # If sep is None: pandas can sniff delimiter with engine="python"
    if sep is None:
        return pd.read_csv(bio, encoding=encoding, sep=None, engine="python", on_bad_lines="skip")

    # Explicit delimiter
    bio.seek(0)
    return pd.read_csv(bio, encoding=encoding, sep=sep, engine="python", on_bad_lines="skip")


@st.cache_data(show_spinner=False)
def read_csv_bytes_robust(data: bytes) -> pd.DataFrame:
    """
    Tries multiple encodings + delimiters. Returns a DataFrame or raises the last exception.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    seps = [None, ",", ";", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = _read_csv_attempt(data, encoding=enc, sep=sep)
                # sanity: must have at least 2 columns and 2 rows to be meaningful
                if df.shape[1] >= 2 and df.shape[0] >= 1:
                    return df
            except Exception as e:
                last_err = e

    raise last_err if last_err else ValueError("Could not parse CSV (unknown error).")


@st.cache_data(show_spinner=False)
def read_csv_path_robust(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    b = p.read_bytes()
    return read_csv_bytes_robust(b)


@st.cache_data(show_spinner=False)
def download_bytes(url: str) -> bytes:
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


# ----------------------------
# Domain logic
# ----------------------------
def parse_thresholds(raw: str) -> list[float]:
    """
    Accepts "25" or "25,50". Returns up to 2 thresholds (floats).
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = []
    for p in parts[:2]:
        vals.append(float(p))
    return vals


def require_columns(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: Missing column(s): {missing}. Available columns: {list(df.columns)[:50]}")


def build_country_daily(df: pd.DataFrame, country_col: str, date_col: str, fatal_col: str) -> pd.DataFrame:
    """
    Returns daily totals per country: [country, date, fatalities]
    """
    d = df[[country_col, date_col, fatal_col]].copy()
    d[country_col] = d[country_col].astype(str)

    # Parse dates robustly
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    # Numeric fatalities
    d[fatal_col] = pd.to_numeric(d[fatal_col], errors="coerce").fillna(0)

    # Daily sum
    out = (
        d.groupby([country_col, pd.Grouper(key=date_col, freq="D")], as_index=False)[fatal_col]
        .sum()
        .rename(columns={country_col: "country", date_col: "date", fatal_col: "fatalities"})
        .sort_values(["country", "date"])
    )
    return out


def compute_escalation_starts(series: pd.Series, threshold: float, persistence_days: int) -> pd.Series:
    """
    series: rolling values indexed by date (float)
    Returns a boolean Series marking "start" points of escalation:
    - rolling > threshold for persistence_days consecutive days
    - and previous day was not already in an escalation run
    """
    above = series > threshold
    # rolling count of consecutive "above"
    consec = above.groupby((~above).cumsum()).cumcount() + 1
    consec = consec.where(above, 0)

    in_escalation = consec >= persistence_days
    starts = in_escalation & (~in_escalation.shift(1, fill_value=False))
    return starts


# ----------------------------
# Sidebar: branding + inputs
# ----------------------------
# Sidebar "hero" video (optional)
# Sidebar "hero" video (optional)
# Sidebar "hero" video (optional)
VIDEO_PATH = Path("logo1.mp4")
if VIDEO_PATH.exists():
    video_bytes = open(VIDEO_PATH, "rb").read()
    video_base64 = base64.b64encode(video_bytes).decode()

    st.sidebar.markdown(
        f"""
        <video autoplay loop muted playsinline style="width:100%; border-radius:12px;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )
st.sidebar.markdown("---")
st.sidebar.header("Inputs")

use_demo = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended for demo)",
    value=False,
    help="Loads ukraine_sample.csv from the repo (must be present alongside app.py)."
)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload any CSV that includes country/date/fatalities columns.")

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Must match the country values in your dataset exactly (e.g., 'Ukraine')."
)

# ----------------------------
# Advanced settings
# ----------------------------
st.write("")
with st.sidebar.expander("Advanced Settings"):

    country_col = st.text_input(
        "Name of Country Column",
        "country",
        help="Column name that contains the country name for each row."
    )

    date_col = st.text_input(
        "Name of Date Column",
        "date_start",
        help="Column name that contains the event date."
    )

    fatalities_col = st.text_input(
        "Name of Fatalities Column",
        "best",
        help="Column name that contains fatalities."
    )

    rolling_window = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Number of days used to compute rolling fatalities."
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        "25,1000",
        help="One or two thresholds. Example: 25 or 25,50"
    )

    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
        help="How many consecutive days the rolling fatalities must exceed the threshold."
    )
    
run_btn = st.sidebar.button("Generate plot")

# Map controls
st.sidebar.markdown("---")
show_map = st.sidebar.checkbox(
    "Show interactive map",
    value=True,
    help="Turn the map section on/off."
)

override_map_dates = st.sidebar.checkbox(
    "Override map date range",
    value=False,
    help="If off, the map automatically uses the min/max dates in your dataset."
)

st.sidebar.markdown("---")


with st.sidebar.expander("Purpose"):
    st.markdown("""

        **Made as Demo for Palantir© Valley Forge Grants**
    
        AEGIS is designed to identify and visualize patterns of conflict escalation using structured event data.

        By aggregating fatalities and applying rolling thresholds, the system highlights periods where violence intensifies beyond normal levels.

        The goal is to provide analysts with an intuitive tool for exploring global conflict dynamics and detecting potential escalation signals early. 
""")



with st.sidebar.expander("Limitations"):
    st.markdown("""
**Current limitations of AEGIS**

- Fatality totals aggregate all events since 1989.

- Some conflicts may be overcounted due to event duplication.

- Escalation detection currently uses simple rolling thresholds.

- Geographic precision is limited to country-level aggregation.


**Planned improvements**

- Subnational geolocation mapping

- Actor-level escalation detection

- Real-time conflict ingestion

- Improved fatality normalization across datasets
""")


# ----------------------------
# Main header
# ----------------------------
col1, col2 = st.columns([1, 12])

with col1:
    st.image("logo.png", width=2000)

with col2:
    st.title("AEGIS — Escalation Detection Demo")
st.caption("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")

# ----------------------------
# Load primary dataset for plot (demo or upload)
# ----------------------------
def load_primary_dataset_for_plot():
    """
    Returns (df_raw, source_label)
    """
    if use_demo:
        if not UKRAINE_SAMPLE_PATH.exists():
            raise FileNotFoundError(
                "Demo file not found. Put ukraine_sample.csv in the SAME folder as app.py (repo root)."
            )
        df_demo = read_csv_path_robust(str(UKRAINE_SAMPLE_PATH))
        return df_demo, "Built-in demo (ukraine_sample.csv)"

    if uploaded is None:
        return None, None

    df_up = read_csv_bytes_robust(uploaded.getvalue())
    return df_up, "Uploaded CSV"


# ----------------------------
# Load world dataset for map (HF hosted)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_world_dataset_for_map() -> pd.DataFrame:
    b = download_bytes(HF_WORLD_CSV_URL)
    return read_csv_bytes_robust(b)


# ----------------------------
# Map section
# ----------------------------

# ----------------------------
# Map section
# ----------------------------
if show_map:
    st.markdown("## Interactive map")
    st.caption("Data source: HuggingFace hosted UCDP world dataset")

    if not _HAS_PLOTLY:
        st.info("Interactive map requires Plotly. Add `plotly` to requirements.txt to enable it.")
    else:
        try:
            # Load the HF world dataset
            df_world = load_world_dataset_for_map()

            # Expected columns in the world dataset
            world_country_col = "country"
            world_date_col = "date_start"
            world_fatal_col = "best"

            require_columns(df_world, [world_country_col, world_date_col, world_fatal_col], "World map dataset")

            # Parse dates + numeric fatalities
            df_world = df_world[[world_country_col, world_date_col, world_fatal_col]].copy()
            df_world[world_date_col] = pd.to_datetime(df_world[world_date_col], errors="coerce")
            df_world = df_world.dropna(subset=[world_date_col])

            df_world[world_fatal_col] = pd.to_numeric(df_world[world_fatal_col], errors="coerce").fillna(0)

            # Auto date range from the dataset
            min_dt = df_world[world_date_col].min().date()
            max_dt = df_world[world_date_col].max().date()

            # Optional override in the sidebar
            if override_map_dates:
                start_dt, end_dt = st.sidebar.date_input(
                    "Map date range",
                    value=(min_dt, max_dt),
                    min_value=min_dt,
                    max_value=max_dt,
                    key="map_date_range",
                )
                # Streamlit can sometimes return a single date if user clicks weirdly
                if isinstance(start_dt, date) and isinstance(end_dt, date) and start_dt <= end_dt:
                    df_world = df_world[
                        (df_world[world_date_col].dt.date >= start_dt)
                        & (df_world[world_date_col].dt.date <= end_dt)
                    ]
            else:
                # Show the auto range (no big control in the main area)
                start_dt, end_dt = min_dt, max_dt

            # Aggregate fatalities by country for the selected range
            by_country = (
                df_world.groupby(world_country_col, as_index=False)[world_fatal_col]
                .sum()
                .rename(columns={world_country_col: "country", world_fatal_col: "fatalities"})
                .sort_values("fatalities", ascending=False)
            )

            # Make higher fatalities darker:
            # Use "Blues" (NOT Blues_r)
            fig = px.choropleth(
                by_country,
                locations="country",
                locationmode="country names",
                color="fatalities",
                hover_name="country",
                hover_data={"country": False, "fatalities": ":,"},
                title=f"Fatalities by country ({start_dt.year}–{end_dt.year})",
                color_continuous_scale="Blues",
            )

            # Cleaner hover (optional but nicer)
            fig.update_traces(
                hovertemplate="<b>%{location}</b><br>fatalities=%{z:,}<extra></extra>"
            )

            fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
            st.caption("To change the date range, enable **Override map date range** in the sidebar.")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Map error: {e}")

# ----------------------------
# Escalation plot section
# ----------------------------
st.subheader("Escalation plot")

df_raw_plot, plot_source = (None, None)
try:
    df_raw_plot, plot_source = load_primary_dataset_for_plot()
except Exception as e:
    st.error(str(e))

if df_raw_plot is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears above.")
    st.stop()

st.caption(f"Plot dataset source: {plot_source}")

# Validate columns for plot dataset
try:
    require_columns(df_raw_plot, [country_col, date_col, fatalities_col], "Plot dataset")
except Exception as e:
    st.error(str(e))
    st.stop()

if not run_btn:
    st.info("CSV loaded — now click **Generate plot**.")
    st.stop()

# Build daily series for selected country
try:
    daily = build_country_daily(df_raw_plot, country_col, date_col, fatalities_col)
    c_daily = daily[daily["country"] == country_name].copy()

    if c_daily.empty:
        st.warning(f"No rows found for country='{country_name}'. Check spelling/case or your country column.")
        st.stop()

    c_daily = c_daily.set_index("date").sort_index()
    c_daily["rolling"] = c_daily["fatalities"].rolling(int(rolling_window), min_periods=1).sum()

    thresholds = parse_thresholds(thresholds_raw)
    if not thresholds:
        st.error("Please provide at least one threshold (e.g., 25 or 25,50).")
        st.stop()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(c_daily.index, c_daily["rolling"], label="Rolling fatalities")

    # Draw thresholds + starts
    for i, thr in enumerate(thresholds):
        ax.axhline(thr, linestyle="--", linewidth=1, label=f"Threshold {i+1}: {thr:g}")
        starts = compute_escalation_starts(c_daily["rolling"], thr, int(persistence_days))
        ax.scatter(
            c_daily.index[starts],
            c_daily["rolling"][starts],
            s=40,
            label=f"Escalation starts (thr={thr:g})"
        )

    ax.set_title(f"AEGIS Escalation Detection — {country_name} (rolling={int(rolling_window)}d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling fatalities")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    # Summary table of first few starts
    st.markdown("### Summary")
    for thr in thresholds:
        starts = compute_escalation_starts(c_daily["rolling"], thr, int(persistence_days))
        starts_df = (
            c_daily.loc[starts, ["rolling"]]
            .reset_index()
            .rename(columns={"index": "date"})
            .assign(threshold=thr)
            .sort_values("date")
        )
        st.write(f"**Threshold {thr:g}: escalation starts detected = {len(starts_df)}**")
        st.dataframe(starts_df.head(10), use_container_width=True)

except Exception as e:
    st.error(str(e))
