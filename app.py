import io
import base64
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import feedparser

# --- Map popup HTML template (inline for Streamlit Cloud compatibility) ---
_MAP_POPUP_TEMPLATE = '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n<style>\n  html, body { margin:0; padding:0; background:#020617; height:100%; }\n  #wrap { position:relative; width:100%; height:820px; }\n  #map  { width:100%; height:820px; }\n  #popup {\n    display:none;\n    position:absolute;\n    top:0; right:0;\n    width:310px;\n    height:100%;\n    background:linear-gradient(160deg,rgba(10,18,38,0.98) 0%,rgba(22,35,60,0.98) 100%);\n    border-left:3px solid #f59e0b;\n    overflow-y:auto;\n    z-index:999;\n    box-sizing:border-box;\n    padding:18px 16px 18px 18px;\n    font-family:Arial,sans-serif;\n    color:white;\n  }\n  .close-btn {\n    float:right; cursor:pointer; font-size:16px;\n    color:rgba(255,255,255,0.5); background:none; border:none;\n    padding:0; margin-top:2px;\n  }\n  .close-btn:hover { color:white; }\n  .grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin:14px 0 4px; }\n  .card { border-radius:9px; padding:11px 8px; text-align:center; }\n  .card .num { font-size:18px; font-weight:800; }\n  .card .lbl { font-size:9px; color:rgba(255,255,255,0.5); margin-top:4px; letter-spacing:.8px; }\n  .news-item { margin:10px 0; }\n  .news-item a { color:#93c5fd; text-decoration:none; font-size:13px; line-height:1.4; }\n  .news-item a:hover { color:white; text-decoration:underline; }\n  .news-meta { font-size:11px; color:rgba(255,255,255,0.4); margin-top:2px; }\n  hr.div { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:8px 0; }\n</style>\n</head>\n<body>\n<div id="wrap">\n  <div id="map"></div>\n  <div id="popup">\n    <button class="close-btn" onclick="closePopup()">&#10005;</button>\n    <div id="popup-body"></div>\n  </div>\n</div>\n<script>\nvar FIG    = __FIG__;\nvar LOOKUP = __LOOKUP__;\nvar STATS  = __STATS__;\nvar COORDS = __COORDS__;\n\nPlotly.newPlot(\'map\', FIG.data, FIG.layout, {scrollZoom:true, displayModeBar:true, responsive:true});\n\ndocument.getElementById(\'map\').on(\'plotly_click\', function(evt) {\n  var pt = evt.points[0];\n  var curve = pt.curveNumber, pidx = pt.pointNumber;\n  var country = (LOOKUP[curve] || [])[pidx] || \'\';\n  if (country) openPopup(country);\n});\n\nfunction n(v) { return (v||0).toLocaleString(); }\n\nfunction card(color, bg, border, val, lbl) {\n  return \'<div class="card" style="background:\'+bg+\';border:1px solid \'+border+\';">\' +\n         \'<div class="num" style="color:\'+color+\'">\'+val+\'</div>\' +\n         \'<div class="lbl">\'+lbl+\'</div></div>\';\n}\n\nfunction openPopup(country) {\n  var s = STATS[country] || {};\n  var coords = COORDS[country];\n  if (coords) {\n    Plotly.relayout(\'map\', {\n      \'mapbox.center\': {lat: coords.lat, lon: coords.lon},\n      \'mapbox.zoom\': coords.zoom\n    });\n  }\n  document.getElementById(\'popup-body\').innerHTML =\n    \'<div style="font-size:20px;font-weight:800;margin-bottom:2px;">&#127758; \' + country + \'</div>\' +\n    \'<div style="font-size:10px;color:rgba(255,255,255,0.35);margin-bottom:14px;letter-spacing:1px;">ACLED CONFLICT DATA</div>\' +\n    \'<div class="grid">\' +\n      card(\'#ef4444\',\'rgba(239,68,68,0.13)\',\'rgba(239,68,68,0.3)\',     n(s.fatalities),        \'FATALITIES\')   +\n      card(\'#f59e0b\',\'rgba(245,158,11,0.13)\',\'rgba(245,158,11,0.3)\',   n(s.explosions),         \'EXPLOSIONS\')   +\n      card(\'#f87171\',\'rgba(248,113,113,0.13)\',\'rgba(248,113,113,0.3)\', n(s.battles),            \'BATTLES\')      +\n      card(\'#fde047\',\'rgba(253,224,71,0.13)\',\'rgba(253,224,71,0.3)\',   n(s.civilian_violence),  \'CIV. VIOLENCE\')+\n      card(\'#60a5fa\',\'rgba(96,165,250,0.13)\',\'rgba(96,165,250,0.3)\',   n(s.strategic),          \'STRATEGIC\')    +\n      card(\'#a78bfa\',\'rgba(167,139,250,0.13)\',\'rgba(167,139,250,0.3)\', n(s.protests),           \'PROTESTS\')     +\n      card(\'#f472b6\',\'rgba(244,114,182,0.13)\',\'rgba(244,114,182,0.3)\', n(s.riots),              \'RIOTS\')        +\n      card(\'white\',\'rgba(255,255,255,0.05)\',\'rgba(255,255,255,0.1)\',   n(s.violent_actors),     \'ACTORS\')       +\n    \'</div>\';\n  document.getElementById(\'popup\').style.display = \'block\';\n}\n\nfunction closePopup() {\n  document.getElementById(\'popup\').style.display = \'none\';\n  Plotly.relayout(\'map\', {\'mapbox.center\':{lat:20,lon:10},\'mapbox.zoom\':1});\n}\n</script>\n</body>\n</html>\n'


# Optional: used for the interactive map.
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="ZoomedLogo.png",
    layout="wide",
)

HF_WORLD_CSV_URL = "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")

# Public ArcGIS layer for ACLED monthly subnational indicators
ACLED_ARCGIS_QUERY_URL = (
    "https://services8.arcgis.com/xu983xJB6fIDCjpX/arcgis/rest/services/ACLED/FeatureServer/0/query"
)
ACLED_FIELDS = [
    "country",
    "admin1",
    "event_month",
    "battles",
    "explosions_remote_violence",
    "protests",
    "riots",
    "strategic_developments",
    "violence_against_civilians",
    "violent_actors",
    "fatalities",
    "centroid_longitude",
    "centroid_latitude",
    "ObjectId",
]


# ----------------------------
# Live conflict news feed helpers
# ----------------------------
NEWS_FEED_URL = (
    "https://news.google.com/rss/search?"
    "q=(war%20OR%20conflict%20OR%20airstrike%20OR%20missile%20OR%20battle%20OR%20explosion)"
    "&hl=en-US&gl=US&ceid=US:en"
)


def get_favicon(url: str) -> str:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return f"https://www.google.com/s2/favicons?sz=64&domain={domain}"


@st.cache_data(ttl=900, show_spinner=False)
def load_live_conflict_news(max_items: int = 5):
    feed = feedparser.parse(NEWS_FEED_URL)

    items = []
    for entry in feed.entries[:max_items]:
        published_raw = entry.get("published", "") or entry.get("updated", "")
        published_dt = None

        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            published_dt = None

        source = entry.get("source", {})
        if hasattr(source, 'get'):
            source_title = source.get("title", "Unknown source")
        else:
            source_title = "Unknown source"

        items.append(
            {
                "title": entry.get("title", "Untitled"),
                "link": entry.get("link", ""),
                "source": source_title,
                "published_raw": published_raw,
                "published_dt": published_dt,
                "media_content": entry.get("media_content", []),
                "media_thumbnail": entry.get("media_thumbnail", []),
                "summary": entry.get("summary", ""),
            }
        )

    return items


def format_news_age(dt_obj):
    if dt_obj is None:
        return ""
    now = datetime.now(timezone.utc)
    delta = now - dt_obj

    if delta < timedelta(minutes=1):
        return "just now"
    if delta < timedelta(hours=1):
        mins = int(delta.total_seconds() // 60)
        return f"{mins}m ago"
    if delta < timedelta(days=1):
        hrs = int(delta.total_seconds() // 3600)
        return f"{hrs}h ago"
    days = delta.days
    return f"{days}d ago"

def get_source_logo_url(source_name: str) -> str:
    source_map = {
        "Reuters": "reuters.com",
        "Associated Press": "apnews.com",
        "AP News": "apnews.com",
        "BBC News": "bbc.com",
        "BBC": "bbc.com",
        "CNN": "cnn.com",
        "The New York Times": "nytimes.com",
        "New York Times": "nytimes.com",
        "Financial Times": "ft.com",
        "Politico": "politico.com",
        "Council on Foreign Relations": "cfr.org",
        "Foreign Affairs": "foreignaffairs.com",
        "The Washington Post": "washingtonpost.com",
        "Wall Street Journal": "wsj.com",
        "Al Jazeera": "aljazeera.com",
        "The Guardian": "theguardian.com",
        "Bloomberg": "bloomberg.com",
        "CNBC": "cnbc.com",
        "Fox News": "foxnews.com",
        "NBC News": "nbcnews.com",
        "CBS News": "cbsnews.com",
        "ABC News": "abcnews.go.com",
        "KUOW": "kuow.org",
    }

    domain = source_map.get(source_name, "")
    if not domain:
        cleaned = (
            source_name.lower()
            .replace("the ", "")
            .replace(" news", "")
            .replace(" ", "")
            .replace(".", "")
        )
        domain = f"{cleaned}.com"

    return f"https://www.google.com/s2/favicons?sz=128&domain={domain}"

# ----------------------------
# Robust CSV loading helpers
# ----------------------------
def _read_csv_attempt(data: bytes, *, encoding: str, sep):
    bio = io.BytesIO(data)
    if sep is None:
        return pd.read_csv(bio, encoding=encoding, sep=None, engine="python", on_bad_lines="skip")
    bio.seek(0)
    return pd.read_csv(bio, encoding=encoding, sep=sep, engine="python", on_bad_lines="skip")


@st.cache_data(show_spinner=False)
def read_csv_bytes_robust(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    seps = [None, ",", ";", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = _read_csv_attempt(data, encoding=enc, sep=sep)
                if df.shape[1] >= 2 and df.shape[0] >= 1:
                    return df
            except Exception as e:
                last_err = e

    raise last_err if last_err else ValueError("Could not parse CSV (unknown error).")


@st.cache_data(show_spinner=False)
def read_csv_path_robust(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    return read_csv_bytes_robust(p.read_bytes())


@st.cache_data(show_spinner=False)
def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content


# ----------------------------
# App helpers
# ----------------------------
def require_columns(df: pd.DataFrame, cols: list[str], label: str = "Dataset"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. Available columns: {list(df.columns)[:20]}"
        )


def parse_thresholds(raw: str) -> list[float]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def build_country_daily(df: pd.DataFrame, country_col: str, date_col: str, fatal_col: str) -> pd.DataFrame:
    d = df[[country_col, date_col, fatal_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d[fatal_col] = pd.to_numeric(d[fatal_col], errors="coerce").fillna(0)

    out = (
        d.groupby([country_col, pd.Grouper(key=date_col, freq="D")], as_index=False)[fatal_col]
        .sum()
        .rename(columns={country_col: "country", date_col: "date", fatal_col: "fatalities"})
        .sort_values(["country", "date"])
    )
    return out


def compute_escalation_starts(series: pd.Series, threshold: float, persistence_days: int) -> pd.Series:
    above = series > threshold
    consec = above.groupby((~above).cumsum()).cumcount() + 1
    consec = consec.where(above, 0)
    in_escalation = consec >= persistence_days
    starts = in_escalation & (~in_escalation.shift(1, fill_value=False))
    return starts


@st.cache_data(show_spinner=False)
def load_world_dataset_for_map() -> pd.DataFrame:
    b = download_bytes(HF_WORLD_CSV_URL)
    return read_csv_bytes_robust(b)


def _parse_arcgis_date_col(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", errors="coerce", utc=True).dt.tz_localize(None)
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_acled_arcgis_monthly() -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            "where": "1=1",
            "outFields": ",".join(ACLED_FIELDS),
            "returnGeometry": "false",
            "orderByFields": "ObjectId ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        r = requests.get(
            ACLED_ARCGIS_QUERY_URL,
            params=params,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        payload = r.json()

        if "error" in payload:
            raise RuntimeError(payload["error"])

        features = payload.get("features", [])
        if not features:
            break

        rows.extend(f.get("attributes", {}) for f in features)
        if len(features) < page_size:
            break
        offset += page_size
        if offset > 50000:
            break

    if not rows:
        return pd.DataFrame(columns=ACLED_FIELDS)

    df = pd.DataFrame(rows)
    numeric_cols = [
        "battles",
        "explosions_remote_violence",
        "protests",
        "riots",
        "strategic_developments",
        "violence_against_civilians",
        "violent_actors",
        "fatalities",
        "centroid_longitude",
        "centroid_latitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_month"] = _parse_arcgis_date_col(df["event_month"]).dt.normalize()
    df = df.dropna(subset=["event_month", "centroid_longitude", "centroid_latitude"])
    df = df[df["centroid_latitude"].between(-90, 90) & df["centroid_longitude"].between(-180, 180)]
    return df


# ----------------------------
# Sidebar: branding + inputs
# ----------------------------
st.sidebar.header("AEGIS Control Bar")

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
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.header("Inputs")

use_demo = st.sidebar.checkbox(
    "Use built-in dataset (recommended for demo)",
    value=False,
    help="A small number of countries may not have data.",
)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload any CSV that includes country/date/fatalities columns.",
    )

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "",
    help="Must match the country values in your dataset exactly (e.g., 'Ukraine').",
)

st.write("")
with st.sidebar.expander("Advanced Settings"):
    country_col = st.text_input(
        "Name of Country Column",
        "country",
        help="Column name that contains the country name for each row.",
    )

    date_col = st.text_input(
        "Name of Date Column",
        "date_start",
        help="Column name that contains the event date.",
    )

    fatalities_col = st.text_input(
        "Name of Fatalities Column",
        "best",
        help="Column name that contains fatalities.",
    )

    rolling_window = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Number of days used to compute rolling fatalities.",
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        "25,1000",
        help="One or two thresholds. Example: 25 or 25,50",
    )

    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
        help="How many consecutive days the rolling fatalities must exceed the threshold.",
    )

run_btn = st.sidebar.button("Generate plot")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div style="opacity:0.6; font-size:13px;">
Plot data source: UCDP GED (1989–present) via HuggingFace.<br>
Map data source: public ACLED ArcGIS monthly indicators.
</div>
""",
    unsafe_allow_html=True,
)

show_map = st.sidebar.checkbox(
    "Show interactive map",
    value=True,
    help="Turn the map section on/off.",
)

override_map_dates = st.sidebar.checkbox(
    "Override map date range",
    value=False,
    help="If off, the map automatically uses the latest month available in the map dataset.",
)

st.sidebar.markdown("---")

with st.sidebar.expander("Purpose"):
    st.markdown(
        """
**Made as Demo for Palantir© Valley Forge Grants**

AEGIS is designed to identify and visualize patterns of conflict escalation using structured event data.

By aggregating fatalities and applying rolling thresholds, the system highlights periods where violence intensifies beyond normal levels.

The goal is to provide analysts with an intuitive tool for exploring global conflict dynamics and detecting potential escalation signals early.
"""
    )

with st.sidebar.expander("Limitations"):
    st.markdown(
        """
**Current limitations of AEGIS**

- Fatality totals aggregate all events since 1989.
- Some conflicts may be overcounted due to event duplication in the datasets.
- Escalation detection currently uses simple rolling thresholds.
- Public ACLED map data is monthly and subnational, not individual strike-level event data.
- Dataset upload currently limited to 200MB / file.

**Planned improvements**

- Subnational geolocation mapping for user-uploaded files
- Actor-level escalation detection
- Higher-frequency conflict ingestion
- Improved fatality normalization across datasets
"""
    )


# ----------------------------
# Live news feed
# ----------------------------
with st.expander("Live conflict news", expanded=False):
    try:
        news_items = load_live_conflict_news(max_items=5)

        if not news_items:
            st.info("No live news items available right now.")
        else:
            for item in news_items:
                image_url = None

                media_content = item.get("media_content") or []
                media_thumbnail = item.get("media_thumbnail") or []

                if media_content and isinstance(media_content[0], dict):
                    image_url = media_content[0].get("url")
                elif media_thumbnail and isinstance(media_thumbnail[0], dict):
                    image_url = media_thumbnail[0].get("url")

                age_txt = format_news_age(item.get("published_dt"))
                meta_parts = [p for p in [item.get("source"), age_txt] if p]
                meta = " • ".join(meta_parts)

                col1, col2 = st.columns([1, 4])

                with col1:
                    if image_url:
                        st.image(image_url, use_container_width=True)
                    else:
                        st.image(get_source_logo_url(item["source"]), width=500)

                with col2:
                    st.markdown(
                        f"**[{item['title']}]({item['link']})**  \n"
                        f"<span style='opacity:0.75'>{meta}</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

            st.caption("Refreshes automatically every 15 minutes.")
    except Exception as e:
        st.warning(f"Could not load live news feed: {e}")

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
        import matplotlib.pyplot as plt

        df_plot = df_raw_plot[[country_col, date_col, fatalities_col]].copy()
        df_plot = df_plot.rename(
            columns={
                country_col: "country",
                date_col: "date",
                fatalities_col: "fatalities",
            }
        )

        df_plot["country"] = df_plot["country"].astype(str).str.strip()
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        df_plot["fatalities"] = pd.to_numeric(df_plot["fatalities"], errors="coerce")
        df_plot = df_plot.dropna(subset=["country", "date", "fatalities"])
        df_plot = df_plot[df_plot["fatalities"] >= 0]

        daily = (
            df_plot.groupby(["country", "date"], as_index=False)["fatalities"]
            .sum()
            .sort_values(["country", "date"])
        )

        selected_country = str(country_name).strip()
        c_daily = daily[daily["country"] == selected_country].copy()

        if c_daily.empty:
            st.warning(
                f"No rows found for country='{selected_country}'. Check spelling/case or rewrite in the following format: [Current Name] ([Previous Name])."
            )
        else:
            c_daily = c_daily.sort_values("date").set_index("date")
            full_index = pd.date_range(start=c_daily.index.min(), end=c_daily.index.max(), freq="D")
            c_daily = c_daily.reindex(full_index)
            c_daily.index.name = "date"
            c_daily["country"] = selected_country
            c_daily["fatalities"] = pd.to_numeric(c_daily["fatalities"], errors="coerce").fillna(0.0)

            window_days = int(rolling_window)
            c_daily["rolling"] = c_daily["fatalities"].rolling(window=window_days, min_periods=1).sum()

            thresholds = parse_thresholds(thresholds_raw)
            if not thresholds:
                st.error("Please provide at least one threshold (e.g., 25 or 25,50).")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(c_daily.index, c_daily["rolling"], label="Rolling fatalities")
                summary_frames = []

                for i, thr in enumerate(thresholds):
                    ax.axhline(y=thr, linestyle="--", linewidth=1, label=f"Threshold {i+1}: {thr:g}")
                    starts = compute_escalation_starts(c_daily["rolling"], thr, int(persistence_days))

                    if starts is None:
                        starts = []
                    elif isinstance(starts, pd.Series):
                        starts = starts[starts].index if starts.dtype == bool else starts.tolist()
                    elif isinstance(starts, pd.Index):
                        starts = list(starts)
                    else:
                        starts = list(starts)

                    if len(starts) > 0:
                        if all(isinstance(x, int) for x in starts):
                            start_dates = c_daily.index[starts]
                        else:
                            start_dates = pd.Index(starts)

                        start_dates = start_dates.intersection(c_daily.index)

                        if len(start_dates) > 0:
                            ax.scatter(
                                start_dates,
                                c_daily.loc[start_dates, "rolling"],
                                s=40,
                                label=f"Escalation starts (thr={thr:g})",
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
# Interactive map section
# ----------------------------
def _build_dominant_category(row: pd.Series) -> str:
    category_map = {
        "battles": "Battles",
        "explosions_remote_violence": "Explosions / remote violence",
        "violence_against_civilians": "Violence against civilians",
        "strategic_developments": "Strategic developments",
        "protests": "Protests",
        "riots": "Riots",
    }
    vals = {k: float(row.get(k, 0) or 0) for k in category_map}
    best_key = max(vals, key=vals.get)
    return category_map[best_key]


if show_map:
    st.markdown("## Interactive map")

    if not _HAS_PLOTLY:
        st.info("Interactive map requires Plotly. Add `plotly` to requirements.txt to enable it.")
    else:
        try:
            df_map = fetch_acled_arcgis_monthly()
            if df_map.empty:
                st.warning("No ACLED map data was returned from the public layer.")
            else:
                violent_cols = [
                    "battles",
                    "explosions_remote_violence",
                    "violence_against_civilians",
                    "strategic_developments",
                ]
                latest_month = df_map["event_month"].max()
                earliest_month = df_map["event_month"].min()

                metric_labels = {
                    "battles": "Battles",
                    "explosions_remote_violence": "Explosions / remote violence",
                    "violence_against_civilians": "Violence against civilians",
                    "strategic_developments": "Strategic developments",
                    "fatalities": "Fatalities",
                    "violent_actors": "Violent actors",
                    "protests": "Protests",
                    "riots": "Riots",
                }

                with st.expander("Conflict map settings", expanded=False):
                    selected_metric = st.selectbox(
                        "Map metric",
                        options=list(metric_labels.keys()),
                        index=1,
                        format_func=lambda x: metric_labels[x],
                        help="Explosions / remote violence is the closest built-in category to airstrikes, missile strikes, and shelling.",
                    )
                    show_only_violent = st.checkbox(
                        "Hide rows with no violent activity",
                        value=True,
                        help="Filters out rows where battles, explosions/remote violence, violence against civilians, and strategic developments are all zero.",
                    )
                    only_selected_country = st.checkbox(
                        "Only show the country entered above",
                        value=False,
                    )
                    size_max = st.slider(
                        "Maximum marker size",
                        min_value=10,
                        max_value=45,
                        value=24,
                        step=1,
                    )
                    auto_refresh_map = st.checkbox(
                        "Auto-refresh map",
                        value=True,
                        help="Reload the app on a timer so the map picks up any new public layer updates.",
                    )
                    refresh_minutes = st.slider(
                        "Auto-refresh interval (minutes)",
                        min_value=15,
                        max_value=180,
                        value=60,
                        step=15,
                        disabled=not auto_refresh_map,
                    )

                if override_map_dates:
                    default_start = max(earliest_month.date(), (latest_month - pd.DateOffset(months=2)).date())
                    start_dt, end_dt = st.sidebar.date_input(
                        "Map date range",
                        value=(default_start, latest_month.date()),
                        min_value=earliest_month.date(),
                        max_value=latest_month.date(),
                        key="map_date_range",
                    )
                    if isinstance(start_dt, date) and isinstance(end_dt, date) and start_dt <= end_dt:
                        df_map = df_map[
                            (df_map["event_month"].dt.date >= start_dt)
                            & (df_map["event_month"].dt.date <= end_dt)
                        ]
                    else:
                        start_dt, end_dt = latest_month.date(), latest_month.date()
                        df_map = df_map[df_map["event_month"].dt.date == latest_month.date()]
                else:
                    start_dt, end_dt = latest_month.date(), latest_month.date()
                    df_map = df_map[df_map["event_month"].dt.date == latest_month.date()]

                if only_selected_country:
                    df_map = df_map[df_map["country"].astype(str).str.strip() == str(country_name).strip()]

                if show_only_violent:
                    df_map = df_map[df_map[violent_cols].fillna(0).sum(axis=1) > 0]

                if df_map.empty:
                    st.info("No rows matched the current map filters.")
                else:
                    grouped = (
                        df_map.groupby(
                            ["country", "admin1", "centroid_latitude", "centroid_longitude"],
                            as_index=False,
                        )
                        [[
                            "battles",
                            "explosions_remote_violence",
                            "protests",
                            "riots",
                            "strategic_developments",
                            "violence_against_civilians",
                            "violent_actors",
                            "fatalities",
                        ]]
                        .sum()
                    )

                    grouped["dominant_category"] = grouped.apply(_build_dominant_category, axis=1)
                    grouped["metric_value"] = pd.to_numeric(grouped[selected_metric], errors="coerce").fillna(0)
                    grouped = grouped[grouped["metric_value"] > 0].copy()

                    if grouped.empty:
                        st.info("No positive values were found for the selected map metric.")
                    else:
                        grouped["admin1"] = grouped["admin1"].fillna("Unknown")
                        grouped["bubble_size"] = grouped["metric_value"].clip(lower=1)
                        grouped["hover_location"] = grouped["admin1"] + ", " + grouped["country"]

                        st.caption(
                            f"Source: public ACLED ArcGIS monthly indicators. Showing {metric_labels[selected_metric]} from {start_dt} to {end_dt}."
                        )
                        st.caption(
                            "This layer is monthly aggregated at the subnational level. Working towards individual strike-by-strike live telemetry."
                        )

                        fig = px.scatter_mapbox(
                            grouped,
                            lat="centroid_latitude",
                            lon="centroid_longitude",
                            color="dominant_category",
                            size="bubble_size",
                            size_max=size_max,
                            hover_name="hover_location",
                            hover_data={
                                "metric_value": ":,",
                                "fatalities": ":,",
                                "battles": ":,",
                                "explosions_remote_violence": ":,",
                                "violence_against_civilians": ":,",
                                "strategic_developments": ":,",
                                "protests": ":,",
                                "riots": ":,",
                                "violent_actors": ":,",
                                "centroid_latitude": False,
                                "centroid_longitude": False,
                                "admin1": False,
                                "country": False,
                                "bubble_size": False,
                            },
                            mapbox_style="carto-darkmatter",
                            center={"lat": 20, "lon": 10},
                            zoom=1,
                            title="Current conflict-related hotspots",
                            color_discrete_map={
                                "Battles": "#ef4444",
                                "Explosions / remote violence": "#f59e0b",
                                "Violence against civilians": "#fde047",
                                "Strategic developments": "#60a5fa",
                                "Protests": "#a78bfa",
                                "Riots": "#f472b6",
                            },
                        )

                        fig.update_traces(
                            marker=dict(opacity=0.85),
                            hovertemplate=(
                                "<b style='font-size:16px'>%{hovertext}</b><br><br>"
                                + f"{metric_labels[selected_metric]}: %{{customdata[0]:,}}<br>"
                                + "Fatalities: %{customdata[1]:,}"
                                + "<extra></extra>"
                            ),
                        )

                        fig.update_layout(
                            paper_bgcolor="#020617",
                            plot_bgcolor="#020617",
                            font=dict(color="white"),
                            title=dict(
                                text="Current Conflict-Related Hotspots",
                                x=0.5, xanchor="center",
                                y=0.98, yanchor="top",
                                font=dict(color="white", size=22),
                            ),
                            legend=dict(
                                title=dict(text="<b>Categories (Click to Isolate):</b>", side="top", font=dict(color="white", size=13)),
                                orientation="h",
                                yanchor="bottom", y=-0.08,
                                xanchor="left", x=0,
                                bgcolor="rgba(2,6,23,0)",
                                font=dict(color="white", size=13),
                            ),
                            margin=dict(l=0, r=0, t=60, b=80),
                            height=780,
                            hoverlabel=dict(bgcolor="rgba(20,20,20,0.95)", font_size=14, font_family="Arial"),
                        )

                        # Build per-trace country lookup so JS knows which country each point belongs to
                        point_lookup = {}
                        for ci, trace in enumerate(fig.data):
                            cat_name = trace.name
                            cat_rows = grouped[grouped["dominant_category"] == cat_name].copy()
                            countries_ordered = []
                            if hasattr(trace, "lat") and trace.lat is not None:
                                for tlat, tlon in zip(trace.lat, trace.lon):
                                    m = cat_rows[
                                        (cat_rows["centroid_latitude"] == tlat) &
                                        (cat_rows["centroid_longitude"] == tlon)
                                    ]
                                    countries_ordered.append(str(m.iloc[0]["country"]) if not m.empty else "")
                            point_lookup[ci] = countries_ordered

                        # Per-country aggregated stats for popup cards
                        country_stats = {}
                        for ctry, rows in grouped.groupby("country"):
                            country_stats[str(ctry)] = {
                                "fatalities":        int(rows["fatalities"].sum()),
                                "explosions":        int(rows["explosions_remote_violence"].sum()),
                                "battles":           int(rows["battles"].sum()),
                                "civilian_violence": int(rows["violence_against_civilians"].sum()),
                                "strategic":         int(rows["strategic_developments"].sum()),
                                "protests":          int(rows["protests"].sum()),
                                "riots":             int(rows["riots"].sum()),
                                "violent_actors":    int(rows["violent_actors"].sum()),
                            }

                        # Per-country zoom coordinates
                        country_coords = {}
                        for ctry, rows in grouped.groupby("country"):
                            clat = (rows["centroid_latitude"].min() + rows["centroid_latitude"].max()) / 2
                            clon = (rows["centroid_longitude"].min() + rows["centroid_longitude"].max()) / 2
                            span = max(
                                rows["centroid_latitude"].max() - rows["centroid_latitude"].min(),
                                rows["centroid_longitude"].max() - rows["centroid_longitude"].min(),
                                0.5,
                            )
                            country_coords[str(ctry)] = {
                                "lat":  round(float(clat), 4),
                                "lon":  round(float(clon), 4),
                                "zoom": round(max(1.5, min(6.0, 5.8 - float(np.log2(span + 1)))), 2),
                            }

                        # Inject data into the HTML template and render as a component
                        html_template = _MAP_POPUP_TEMPLATE
                        map_html = (
                            html_template
                            .replace("__FIG__",    fig.to_json())
                            .replace("__LOOKUP__", json.dumps(point_lookup))
                            .replace("__STATS__",  json.dumps(country_stats))
                            .replace("__COORDS__", json.dumps(country_coords))
                            .replace("__METRIC_LABEL__", metric_labels[selected_metric])
                        )
                        st.components.v1.html(map_html, height=825, scrolling=False)

                        summary_cols = [
                            "country",
                            "admin1",
                            "metric_value",
                            "fatalities",
                            "battles",
                            "explosions_remote_violence",
                            "violence_against_civilians",
                        ]
                        if selected_metric in {"battles", "explosions_remote_violence", "violence_against_civilians", "fatalities"}:
                            summary_cols = [c for c in summary_cols if c != selected_metric]

                        top_hotspots = grouped.sort_values("metric_value", ascending=False)[summary_cols].head(25).copy()
                        top_hotspots = top_hotspots.rename(
                            columns={
                                "country": "Country",
                                "admin1": "Admin1",
                                "metric_value": f"Selected metric ({metric_labels[selected_metric]})",
                                "fatalities": "Fatalities",
                                "battles": "Battles",
                                "explosions_remote_violence": "Explosions / remote violence",
                                "violence_against_civilians": "Violence against civilians",
                            }
                        )

                        with st.expander("Top hotspots in the current view"):
                            st.dataframe(top_hotspots, use_container_width=True)

                        if auto_refresh_map:
                            refresh_ms = int(refresh_minutes) * 60 * 1000
                            st.components.v1.html(
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

        except Exception as e:
            st.error(f"Map error: {e}")
