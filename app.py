import io
import os
from pathlib import Path
from datetime import date
import base64

import numpy as np
import pandas as pd
import streamlit as st

from urllib.parse import urlencode
import streamlit.components.v1 as components

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

st.sidebar.header("AEGIS Control Bar")

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
    "Use built-in dataset (recommended for demo)",
    value=False,
    help="A small number of countries may not have data."
)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload any CSV that includes country/date/fatalities columns.")

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Demo data currently only goes until the end of 2024."
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

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
<div style="opacity:0.6; font-size:13px;">
Data sources: UCDP GED (1989–present) via HuggingFace.<br>
News headlines via Google News RSS.
</div>
""",
    unsafe_allow_html=True
)

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

- Some conflicts may be overcounted due to event duplication in the datasets.

- Escalation detection currently uses simple rolling thresholds.

- Geographic precision is limited to country-level aggregation.

- Dataset upload currently limited to 200MB / file.


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
# Escalation plot section
# ----------------------------

st.subheader("Escalation plot")

df_raw_plot, plot_source = (None, None)
try:
    df_raw_plot, plot_source = load_primary_dataset_for_plot()
except Exception as e:
    st.error(str(e))

plot_ready = True

if df_raw_plot is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**. The interactive map appears below.")
    plot_ready = False
else:
    st.caption(f"Plot dataset source: {plot_source}")
    st.caption("Source: Uppsala Conflict Data Program (UCDP) Georeferenced Event Dataset via HuggingFace.")

    try:
        require_columns(df_raw_plot, [country_col, date_col, fatalities_col], "Plot dataset")
    except Exception as e:
        st.error(str(e))
        plot_ready = False

    if plot_ready and (not run_btn):
        st.info("CSV loaded — now click **Generate plot**.")
        plot_ready = False

if plot_ready:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        # Keep only needed columns and standardize names
        df_plot = df_raw_plot[[country_col, date_col, fatalities_col]].copy()
        df_plot = df_plot.rename(
            columns={
                country_col: "country",
                date_col: "date",
                fatalities_col: "fatalities",
            }
        )

        # Clean and coerce types
        df_plot["country"] = df_plot["country"].astype(str).str.strip()
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        df_plot["fatalities"] = pd.to_numeric(df_plot["fatalities"], errors="coerce")

        # Drop unusable rows
        df_plot = df_plot.dropna(subset=["country", "date", "fatalities"])
        df_plot = df_plot[df_plot["fatalities"] >= 0]

        # Aggregate to one row per country-date
        daily = (
            df_plot.groupby(["country", "date"], as_index=False)["fatalities"]
            .sum()
            .sort_values(["country", "date"])
        )

        # Exact country match after trimming whitespace
        selected_country = str(country_name).strip()
        c_daily = daily[daily["country"] == selected_country].copy()

        if c_daily.empty:
            st.warning(
                f"No rows found for country='{selected_country}'. "
                "Check spelling/case or your country column."
            )
        else:
            # Create a full daily calendar so rolling(window=N) means N calendar days
            c_daily = c_daily.sort_values("date").set_index("date")

            full_index = pd.date_range(
                start=c_daily.index.min(),
                end=c_daily.index.max(),
                freq="D"
            )

            c_daily = c_daily.reindex(full_index)
            c_daily.index.name = "date"
            c_daily["country"] = selected_country
            c_daily["fatalities"] = pd.to_numeric(
                c_daily["fatalities"], errors="coerce"
            ).fillna(0.0)

            # Rolling daily fatalities over a true daily series
            window_days = int(rolling_window)
            c_daily["rolling"] = (
                c_daily["fatalities"]
                .rolling(window=window_days, min_periods=1)
                .sum()
            )

            thresholds = parse_thresholds(thresholds_raw)
            if not thresholds:
                st.error("Please provide at least one threshold (e.g., 25 or 25,50).")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(c_daily.index, c_daily["rolling"], label="Rolling fatalities")

                summary_frames = []

                for i, thr in enumerate(thresholds):
                    ax.axhline(
                        y=thr,
                        linestyle="--",
                        linewidth=1,
                        label=f"Threshold {i+1}: {thr:g}"
                    )

                    starts = compute_escalation_starts(
                        c_daily["rolling"],
                        thr,
                        int(persistence_days)
                    )

                    # Normalize starts into labels that match c_daily.index
                    if starts is None:
                        starts = []
                    elif isinstance(starts, pd.Series):
                        starts = starts[starts].index if starts.dtype == bool else starts.tolist()
                    elif isinstance(starts, pd.Index):
                        starts = list(starts)
                    else:
                        starts = list(starts)

                    if len(starts) > 0:
                        # If compute_escalation_starts returned integer positions,
                        # convert them to datetime labels. Otherwise assume labels.
                        if all(isinstance(x, (int,)) for x in starts):
                            start_dates = c_daily.index[starts]
                        else:
                            start_dates = pd.Index(starts)

                        start_dates = start_dates.intersection(c_daily.index)

                        if len(start_dates) > 0:
                            ax.scatter(
                                start_dates,
                                c_daily.loc[start_dates, "rolling"],
                                s=40,
                                label=f"Escalation starts (thr={thr:g})"
                            )

                            starts_df = (
                                c_daily.loc[start_dates, ["rolling"]]
                                .reset_index()
                                .rename(columns={"index": "date"})
                                .assign(threshold=thr)
                                .sort_values("date")
                            )
                        else:
                            starts_df = pd.DataFrame(columns=["date", "rolling", "threshold"])
                    else:
                        starts_df = pd.DataFrame(columns=["date", "rolling", "threshold"])

                    summary_frames.append((thr, starts_df))

                ax.set_title(f"AEGIS Escalation Detection — {selected_country} (rolling={window_days}d)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Rolling fatalities")
                ax.grid(True, alpha=0.3)
                ax.legend()

                st.pyplot(fig, clear_figure=True)

                with st.expander("Preview daily input used for the rolling calculation"):
                    preview_df = (
                        c_daily.reset_index()[["date", "fatalities", "rolling"]]
                        .sort_values("date", ascending=False)
                        .head(20)
                    )
                    st.dataframe(preview_df, use_container_width=True)

                st.markdown("### Summary")
                for thr, starts_df in summary_frames:
                    st.write(f"**Threshold {thr:g}: escalation starts detected = {len(starts_df)}**")
                    st.dataframe(starts_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(str(e))

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
# Live map helpers (GDELT GEO 2.0)
# ----------------------------
LIVE_MAP_DEFAULT_QUERY = (
    '(war OR conflict OR invasion OR insurgency OR airstrike OR missile '
    'OR drone OR shelling OR ceasefire OR militia OR rebels OR offensive)'
)

def build_gdelt_live_map_url(
    query: str,
    *,
    timespan: str = "24h",
    maxpoints: int = 250,
    geores: int = 2,
    sortby: str = "Date",
    zoomwheel: bool = False,
) -> str:
    params = {
        "query": query,
        "mode": "point",
        "format": "html",
        "timespan": timespan,
        "maxpoints": max(1, min(int(maxpoints), 1000)),
        "geores": geores,
        "sortby": sortby,
        "zoomwheel": 1 if zoomwheel else 0,
    }
    return "https://api.gdeltproject.org/api/v2/geo/geo?" + urlencode(params)


# ----------------------------
# Live interactive map section
# ----------------------------
if show_map:
    st.markdown("## Live interactive map")
    st.caption(
        "This map uses GDELT GEO 2.0 to display live geocoded global news locations. "
        "It updates as new reporting appears."
    )

    with st.expander("Live map settings", expanded=False):
        live_query = st.text_input(
            "Live event query",
            value=LIVE_MAP_DEFAULT_QUERY,
            help='You can leave this alone or narrow it, for example: "(Ukraine OR Russia)" or "(Israel OR Gaza OR Hamas)".'
        )

        live_timespan = st.selectbox(
            "Lookback window",
            options=["15m", "30m", "1h", "3h", "6h", "12h", "24h", "3d", "7d"],
            index=6,
            help="How far back the live feed should search."
        )

        live_maxpoints = st.slider(
            "Maximum mapped locations",
            min_value=25,
            max_value=500,
            value=250,
            step=25,
            help="More points gives more coverage but also more noise."
        )

        live_geores = st.selectbox(
            "Geographic precision",
            options=[0, 1, 2],
            index=2,
            format_func=lambda x: {
                0: "All mentions (country + region + city)",
                1: "Exclude country-level mentions",
                2: "City / landmark only",
            }[x],
            help="Higher precision keeps the map closer to event-level locations."
        )

        auto_refresh_live_map = st.checkbox(
            "Auto-refresh live map",
            value=True,
            help="Reloads the app automatically so the map stays current."
        )

        refresh_minutes = st.slider(
            "Auto-refresh interval (minutes)",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            disabled=not auto_refresh_live_map,
        )

    live_map_url = build_gdelt_live_map_url(
        live_query,
        timespan=live_timespan,
        maxpoints=live_maxpoints,
        geores=live_geores,
        sortby="Date",
        zoomwheel=False,
    )

    st.markdown(
        f"""
        <div style="
            padding:12px 14px;
            border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;
            margin-bottom:10px;
            background: rgba(255,255,255,0.02);
        ">
            <div style="font-size:14px; opacity:0.85;">
                <strong>Live source:</strong> GDELT GEO 2.0<br>
                <strong>Query:</strong> <code>{live_query}</code><br>
                <strong>Window:</strong> {live_timespan}
                &nbsp;|&nbsp;
                <strong>Max points:</strong> {live_maxpoints}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.iframe(live_map_url, height=760, scrolling=False)

    st.caption(
        "Note: this is a live geocoded news map, not a confirmed casualty database. "
        "It is far more current than UCDP, but also noisier."
    )

    if auto_refresh_live_map:
        refresh_ms = int(refresh_minutes) * 60 * 1000
        components.html(
            f"""
            <script>
            window.setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_ms});
            </script>
            """,
            height=0,
            width=0,
        )
