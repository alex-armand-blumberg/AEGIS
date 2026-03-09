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

# ── Page navigation state ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

# ── Landing page ──────────────────────────────────────────────────────────────
if st.session_state["page"] == "landing":
    # Hide sidebar entirely on landing page
    st.markdown(
        """<style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stSidebarCollapsedControl"] { display: none; }
        .block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 100% !important; }
        header { display: none; }
        </style>""",
        unsafe_allow_html=True,
    )

    LANDING_VIDEO = Path("landing.mp4")
    if LANDING_VIDEO.exists():
        v64 = base64.b64encode(open(LANDING_VIDEO, "rb").read()).decode()
        video_tag = f'<source src="data:video/mp4;base64,{v64}" type="video/mp4">'
    else:
        video_tag = ""

    st.components.v1.html(
        f"""
        <style>
          * {{ margin: 0; padding: 0; box-sizing: border-box; }}
          body {{ background: #000; overflow: hidden; }}
          .wrap {{
            position: relative; width: 100vw; height: 100vh;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            font-family: 'Inter', sans-serif;
          }}
          video {{
            position: absolute; inset: 0; width: 100%; height: 100%;
            object-fit: cover; opacity: 0.35; filter: grayscale(60%);
          }}
          .overlay {{
            position: absolute; inset: 0;
            background: linear-gradient(180deg, rgba(2,6,23,0.5) 0%, rgba(2,6,23,0.7) 100%);
          }}
          .content {{
            position: relative; z-index: 10; text-align: center; padding: 0 24px;
          }}
          .tag {{
            font-size: 11px; letter-spacing: 0.2em; color: #ef4444;
            font-weight: 700; text-transform: uppercase; margin-bottom: 18px;
          }}
          h1 {{
            font-size: clamp(28px, 5vw, 64px); font-weight: 800;
            color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 10px;
            text-shadow: 0 2px 40px rgba(0,0,0,0.8);
          }}
          .sub {{
            font-size: clamp(11px, 1.5vw, 15px); color: #94a3b8;
            letter-spacing: 0.12em; text-transform: uppercase;
            margin-bottom: 48px;
          }}
          .nodash {{ color: #475569; margin: 0 8px; }}
        </style>
        <div class="wrap">
          {"<video autoplay loop muted playsinline>" + video_tag + "</video>" if video_tag else "<div style='position:absolute;inset:0;background:radial-gradient(ellipse at 50% 40%,#0f1e3a 0%,#020617 70%)'></div>"}
          <div class="overlay"></div>
          <div class="content">
            <div class="tag">&#9632; Palantir Valley Forge Grant Demo</div>
            <h1>AEGIS</h1>
            <div class="sub">Advanced Early-warning
              <span class="nodash">&amp;</span>
              Geostrategic Intelligence System</div>
          </div>
        </div>
        """,
        height=520,
        scrolling=False,
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([2, 1, 1, 2])
    with c1:
        if st.button("📊  Escalation Index", use_container_width=True, type="primary"):
            st.session_state["page"] = "index"
            st.rerun()
    with c2:
        if st.button("🗺️  Interactive Map", use_container_width=True):
            st.session_state["page"] = "map"
            st.rerun()

    st.markdown(
        "<div style='text-align:center;color:#334155;font-size:11px;margin-top:12px;'>"
        "© 2026 Alexander Armand-Blumberg · AEGIS</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Public ArcGIS layer for ACLED monthly subnational indicators
ACLED_ARCGIS_QUERY_URL = (
    "https://services8.arcgis.com/xu983xJB6fIDCjpX/arcgis/rest/services/ACLED/FeatureServer/0/query"
)
ACLED_FIELDS = [
    "country", "admin1", "event_month",
    "battles", "explosions_remote_violence", "protests", "riots",
    "strategic_developments", "violence_against_civilians",
    "violent_actors", "fatalities",
    "centroid_longitude", "centroid_latitude", "ObjectId",
]


# ----------------------------
# Live conflict news helpers
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
        published_dt = None
        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
        source = entry.get("source", {})
        source_title = source.get("title", "Unknown source") if hasattr(source, "get") else "Unknown source"
        items.append({
            "title": entry.get("title", "Untitled"),
            "link": entry.get("link", ""),
            "source": source_title,
            "published_raw": entry.get("published", ""),
            "published_dt": published_dt,
            "media_content": entry.get("media_content", []),
            "media_thumbnail": entry.get("media_thumbnail", []),
            "summary": entry.get("summary", ""),
        })
    return items


def format_news_age(dt_obj):
    if dt_obj is None:
        return ""
    now = datetime.now(timezone.utc)
    delta = now - dt_obj
    if delta < timedelta(minutes=1):
        return "just now"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)}m ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)}h ago"
    return f"{delta.days}d ago"


def get_source_logo_url(source_name: str) -> str:
    source_map = {
        "Reuters": "reuters.com", "Associated Press": "apnews.com", "AP News": "apnews.com",
        "BBC News": "bbc.com", "BBC": "bbc.com", "CNN": "cnn.com",
        "The New York Times": "nytimes.com", "New York Times": "nytimes.com",
        "Financial Times": "ft.com", "Politico": "politico.com",
        "Council on Foreign Relations": "cfr.org", "Foreign Affairs": "foreignaffairs.com",
        "The Washington Post": "washingtonpost.com", "Wall Street Journal": "wsj.com",
        "Al Jazeera": "aljazeera.com", "The Guardian": "theguardian.com",
        "Bloomberg": "bloomberg.com", "CNBC": "cnbc.com", "Fox News": "foxnews.com",
        "NBC News": "nbcnews.com", "CBS News": "cbsnews.com", "ABC News": "abcnews.go.com",
    }
    domain = source_map.get(source_name, "")
    if not domain:
        cleaned = (
            source_name.lower()
            .replace("the ", "").replace(" news", "")
            .replace(" ", "").replace(".", "")
        )
        domain = f"{cleaned}.com"
    return f"https://www.google.com/s2/favicons?sz=128&domain={domain}"


@st.cache_data(ttl=900, show_spinner=False)
def load_country_news(country: str, max_items: int = 5):
    query = requests.utils.quote(f"{country} conflict war military")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        published_dt = None
        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
        source = entry.get("source", {})
        source_title = source.get("title", "Unknown source") if hasattr(source, "get") else "Unknown source"
        items.append({
            "title": entry.get("title", "Untitled"),
            "link": entry.get("link", ""),
            "source": source_title,
            "published_dt": published_dt,
        })
    return items


# ----------------------------
# ACLED data fetcher
# ----------------------------
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
        "battles", "explosions_remote_violence", "protests", "riots",
        "strategic_developments", "violence_against_civilians",
        "violent_actors", "fatalities", "centroid_longitude", "centroid_latitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_month"] = _parse_arcgis_date_col(df["event_month"]).dt.normalize()
    df = df.dropna(subset=["event_month", "centroid_longitude", "centroid_latitude"])
    df = df[df["centroid_latitude"].between(-90, 90) & df["centroid_longitude"].between(-180, 180)]
    # Drop any future placeholder months the layer may contain
    today = pd.Timestamp.now().normalize()
    df = df[df["event_month"] <= today]
    return df


# ----------------------------
# ACLED API full-history fetch (new OAuth system — uses myACLED email + password)
# No API key needed. Register free at acleddata.com/user/register
# ----------------------------
ACLED_OAUTH_URL  = "https://acleddata.com/oauth/token"
ACLED_API_URL    = "https://acleddata.com/api/acled/read"

ACLED_EVENT_TYPE_MAP = {
    "Battles":                      "battles",
    "Explosions/Remote violence":   "explosions_remote_violence",
    "Violence against civilians":   "violence_against_civilians",
    "Strategic developments":       "strategic_developments",
    "Protests":                     "protests",
    "Riots":                        "riots",
}


def _get_acled_oauth_token(email: str, password: str) -> str:
    """Exchange myACLED credentials for a 24-hour bearer token."""
    r = requests.post(
        ACLED_OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username":   email.strip(),
            "password":   password.strip(),
            "grant_type": "password",
            "client_id":  "acled",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"ACLED login failed ({r.status_code}). Check your email and password."
        )
    return r.json()["access_token"]


@st.cache_data(ttl=3600, show_spinner=False)
def _get_acled_bearer_token(email: str, password: str) -> str:
    """Cached OAuth token — valid 24h, re-fetched every hour."""
    return _get_acled_oauth_token(email, password)


def _process_acled_rows(all_rows: list, country: str) -> pd.DataFrame:
    """Convert raw API row-list → aggregated country×month DataFrame."""
    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    raw.columns = [c.lower().strip() for c in raw.columns]

    date_col = next((c for c in raw.columns if "date" in c), None)
    if date_col is None:
        return pd.DataFrame()

    raw["event_date"] = pd.to_datetime(raw[date_col], errors="coerce")
    raw["fatalities"] = pd.to_numeric(raw.get("fatalities", 0), errors="coerce").fillna(0)
    raw["latitude"]   = pd.to_numeric(raw.get("latitude",  pd.Series(0, index=raw.index)), errors="coerce")
    raw["longitude"]  = pd.to_numeric(raw.get("longitude", pd.Series(0, index=raw.index)), errors="coerce")

    type_col = next((c for c in raw.columns if "event_type" in c), None)
    raw["event_type"] = raw[type_col].astype(str).str.strip() if type_col else ""

    if "country" not in raw.columns:
        raw["country"] = country

    raw = raw.dropna(subset=["event_date"])
    raw["event_month"] = raw["event_date"].dt.to_period("M").dt.to_timestamp()
    raw = raw[raw["event_month"] <= pd.Timestamp.now().normalize()]

    for col in ACLED_EVENT_TYPE_MAP.values():
        raw[col] = 0
    for api_type, col in ACLED_EVENT_TYPE_MAP.items():
        raw.loc[raw["event_type"] == api_type, col] = 1

    agg = (
        raw.groupby(["country", "event_month"], as_index=False)
        .agg(
            battles=("battles", "sum"),
            explosions_remote_violence=("explosions_remote_violence", "sum"),
            protests=("protests", "sum"),
            riots=("riots", "sum"),
            strategic_developments=("strategic_developments", "sum"),
            violence_against_civilians=("violence_against_civilians", "sum"),
            fatalities=("fatalities", "sum"),
            centroid_latitude=("latitude",  "median"),
            centroid_longitude=("longitude", "median"),
        )
    )
    agg["admin1"] = ""
    agg["violent_actors"] = 0
    return agg


# ----------------------------
# Ticker bar — top escalating countries (ArcGIS, no login needed)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data() -> list:
    """
    Pull the latest month from the ArcGIS layer, compute a simple
    intensity score per country, and return the top 10 as ticker items.
    """
    try:
        df = fetch_acled_arcgis_monthly()
        if df.empty:
            return []

        latest = df["event_month"].max()
        prev   = latest - pd.DateOffset(months=1)

        cur  = df[df["event_month"] == latest].copy()
        prev_df = df[df["event_month"] == prev].copy()

        event_cols = ["battles", "explosions_remote_violence",
                      "violence_against_civilians", "protests",
                      "riots", "strategic_developments"]

        cur["score"] = cur[event_cols].sum(axis=1)
        prev_df["score"] = prev_df[event_cols].sum(axis=1)

        cur_agg  = cur.groupby("country")["score"].sum().reset_index()
        prev_agg = prev_df.groupby("country")["score"].sum().reset_index()

        merged = cur_agg.merge(prev_agg, on="country", suffixes=("_cur", "_prev"), how="left")
        merged["prev_score"] = merged["score_prev"].fillna(0)
        merged["trend"] = merged["score_cur"] - merged["prev_score"]

        top = merged.nlargest(10, "score_cur")

        items = []
        for _, row in top.iterrows():
            score = int(row["score_cur"])
            if row["trend"] > 5:
                arrow, color = "▲", "#ef4444"
            elif row["trend"] < -5:
                arrow, color = "▼", "#60a5fa"
            else:
                arrow, color = "→", "#f59e0b"
            items.append({
                "country": row["country"],
                "score":   score,
                "arrow":   arrow,
                "color":   color,
            })
        return items
    except Exception:
        return []


def render_ticker(items: list, month_label: str = "") -> None:
    """Render a scrolling ticker bar using st.components HTML."""
    if not items:
        return

    # Build the inner text spans
    parts = []
    for it in items:
        parts.append(
            f'<span style="color:{it["color"]};font-weight:600;">'
            f'&#11044;&nbsp;{it["country"]}</span>'
            f'&nbsp;<span style="opacity:0.7;">{it["score"]} events</span>'
            f'&nbsp;<span style="color:{it["color"]};">{it["arrow"]}</span>'
        )
    # Join with a separator
    sep = '&nbsp;&nbsp;<span style="opacity:0.3;">|</span>&nbsp;&nbsp;'
    content = sep.join(parts)
    # Duplicate for seamless loop
    content_double = content + sep + content

    label = f"&nbsp;&nbsp;🌐 LIVE CONFLICT TICKER — Top 10 active conflict zones · {month_label}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

    html = f"""
<div style="
    background: linear-gradient(90deg,#0f172a 0%,#1e293b 100%);
    border:1px solid #334155;
    border-radius:6px;
    padding:0;
    overflow:hidden;
    display:flex;
    align-items:center;
    height:36px;
    margin-bottom:8px;
    font-family:'Inter',sans-serif;
    font-size:13px;
    color:#e2e8f0;
">
  <div style="
    flex-shrink:0;
    background:#ef4444;
    color:white;
    font-weight:700;
    font-size:11px;
    letter-spacing:0.05em;
    padding:0 10px;
    height:100%;
    display:flex;
    align-items:center;
    white-space:nowrap;
  ">LIVE</div>
  <div style="overflow:hidden;flex:1;height:100%;position:relative;">
    <div style="
      display:inline-block;
      white-space:nowrap;
      animation:ticker 35s linear infinite;
      height:100%;
      line-height:36px;
    ">{content_double}</div>
  </div>
</div>
<style>
@keyframes ticker {{
  0%   {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
</style>
"""
    st.components.v1.html(html, height=44, scrolling=False)


# ----------------------------
# Escalation Index computation
# ----------------------------
def compute_escalation_index(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Monthly Escalation Index (0-100) built from six ACLED components.

    The key design principle: separate INTENSITY (how bad is it right now in
    absolute terms) from ACCELERATION (is it getting worse). A steady-state
    war like Ukraine scores high on intensity; a country entering conflict
    scores high on acceleration. Both are dangerous for different reasons.

      Component                        Weight   Signal type
      ──────────────────────────────────────────────────────
      Raw conflict intensity            30%     Lagging anchor — battles +
        (battles + explosions)                  explosions normalised globally.
                                                Ensures sustained wars like Ukraine
                                                score high even with flat MoM change.
      Event frequency acceleration      20%     Leading — MoM % change in total
                                                events. Catches countries entering
                                                or re-escalating conflict.
      Explosions / remote violence      20%     Leading — shelling/airstrikes
                                                precede ground battle fatalities.
      Strategic developments            15%     Leading — troop moves, HQ changes,
                                                peace deal collapses.
      Civil unrest (protests + riots)   10%     Leading — social unrest precedes
                                                armed conflict escalation.
      Civilian targeting ratio           5%     Leading — shift to civilians signals
                                                strategic deterioration.

    All components normalised globally (all countries, all months in dataset).
    """
    event_cols = [
        "battles", "explosions_remote_violence", "protests", "riots",
        "strategic_developments", "violence_against_civilians", "violent_actors",
    ]
    agg = (
        df.groupby(["country", "event_month"], as_index=False)[event_cols + ["fatalities"]]
        .sum()
        .sort_values(["country", "event_month"])
    )

    agg["total_events"] = agg[event_cols].sum(axis=1)
    agg["violent_events"] = (
        agg["battles"] + agg["explosions_remote_violence"] + agg["violence_against_civilians"]
    )

    # Component 1: raw conflict intensity (battles + explosions — absolute scale)
    agg["intensity_raw"] = agg["battles"] + agg["explosions_remote_violence"]

    # Component 2: event frequency acceleration (MoM % change, clipped to [-2, 10])
    agg["event_accel_raw"] = (
        agg.groupby("country")["total_events"]
        .pct_change()
        .clip(-2, 10)
        .fillna(0)
    )

    # Component 3–5: raw counts
    agg["explosions_raw"] = agg["explosions_remote_violence"]
    agg["strategic_raw"]  = agg["strategic_developments"]
    agg["unrest_raw"]     = agg["protests"] + agg["riots"]

    # Component 6: civilian targeting ratio
    agg["civ_ratio_raw"] = np.where(
        agg["violent_events"] > 0,
        agg["violence_against_civilians"] / agg["violent_events"],
        0.0,
    )

    # Normalise every component using percentile rank (0→1) across all country-months.
    # This prevents one extreme outlier country (e.g. Ukraine) from crushing every
    # other country's score under min-max scaling. A country in the 80th percentile
    # globally scores 0.8 regardless of absolute event volume.
    def pct_rank(col):
        return agg[col].rank(pct=True, method="average").fillna(0.0)

    agg["c_intensity"]  = pct_rank("intensity_raw")
    agg["c_accel"]      = pct_rank("event_accel_raw")
    agg["c_explosion"]  = pct_rank("explosions_raw")
    agg["c_strategic"]  = pct_rank("strategic_raw")
    agg["c_unrest"]     = pct_rank("unrest_raw")
    agg["c_civilian"]   = pct_rank("civ_ratio_raw")

    # Weighted composite → 0–100
    agg["escalation_index"] = (
        0.30 * agg["c_intensity"]
        + 0.20 * agg["c_accel"]
        + 0.20 * agg["c_explosion"]
        + 0.15 * agg["c_strategic"]
        + 0.10 * agg["c_unrest"]
        + 0.05 * agg["c_civilian"]
    ) * 100

    return agg[agg["country"] == country].copy().reset_index(drop=True)


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

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "",
    help="Must match the country name in ACLED exactly (e.g. 'Ukraine', 'Sudan', 'Myanmar').",
)

# Load ACLED credentials silently from Streamlit secrets
try:
    acled_api_email = st.secrets["acled"]["email"]
    acled_api_key   = st.secrets["acled"]["password"]
    use_api = bool(acled_api_email and acled_api_key)
except Exception:
    acled_api_email = ""
    acled_api_key   = ""
    use_api = False

with st.sidebar.expander("Advanced Settings"):
    escalation_threshold = st.slider(
        "Escalation alert threshold (0–100)",
        min_value=0,
        max_value=100,
        value=45,
        step=1,
        help="Recommended: 45 for major conflicts. Lower to 35–40 for smaller or less-covered conflicts.",
    )
    smooth_window = st.number_input(
        "Smoothing window (months)",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
        help="Rolling average applied to reduce month-to-month noise.",
    )
    show_components = st.checkbox(
        "Show component breakdown chart",
        value=False,
        help="Stacked bar chart showing each sub-index contribution.",
    )
    st.markdown("**Plot date range**")
    st.caption("Data from Jan 2018 to Jan 2026.")
    help="Data from Jan 2018 to Jan 2025."
    plot_date_col1, plot_date_col2 = st.columns(2)
    with plot_date_col1:
        plot_start_date = st.date_input(
            "From",
            value=date(2018, 1, 1),
            min_value=date(2018, 1, 1),
            max_value=date.today(),
            key="plot_start",
        )
    with plot_date_col2:
        plot_end_date = st.date_input(
            "To",
            value=date.today(),
            min_value=date(2018, 1, 1),
            max_value=date.today(),
            key="plot_end",
        )

run_btn = st.sidebar.button("Generate plot")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div style="opacity:0.6; font-size:13px;">
Plot data source: ACLED (2018-2025 history via Researcher-Tier API).<br>
Map data source: Public ACLED ArcGIS layer.
</div>
""",
    unsafe_allow_html=True,
)

show_map = st.sidebar.checkbox(
    "Show interactive map",
    value=(st.session_state.get("page") != "index"),
    help="Turn the map section on/off.",
)

override_map_dates = st.sidebar.checkbox(
    "Override map date range",
    value=False,
    help="If off, the map automatically uses the latest month available.",
)

st.sidebar.markdown("---")

with st.sidebar.expander("Purpose"):
    st.markdown(
        """
**Made as Demo for Palantir© Valley Forge Grants**

AEGIS identifies and visualizes conflict escalation patterns using ACLED event data.

The Escalation Index combines five **leading indicators** — signals that tend to
precede kinetic violence rather than confirm it after the fact:
event frequency acceleration, explosions/remote violence, strategic developments,
civil unrest, and civilian targeting ratio.

Unlike fatality counts (a lagging indicator), these signals surface escalation
pressure before it peaks.
"""
    )

with st.sidebar.expander("Limitations"):
    st.markdown(
        """
**Current limitations of AEGIS**

- Only have access to data from Jan 2018 to exactly One Year Ago for Escalation Index, as I currently only have Researcher Tier ACLED access.
- ACLED public ArcGIS layer for the map is monthly aggregated at subnational level, not individual events.
- Some countries may have sparse data in earlier months.
- Public map data is monthly and subnational, not individual strike-level event data.

**Planned improvements**

- Get a higher ACLED Tier, giving me access to more data for Escalation Index.
- Direct ACLED API for weekly/event-level granularity
- Actor-level escalation detection
- ML-based index calibration against historical escalation outcomes
- Subnational index breakdown
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
                meta = " • ".join(p for p in [item.get("source"), age_txt] if p)

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
if st.button("← Back to AEGIS", key="back_btn"):
    st.session_state["page"] = "landing"
    st.rerun()

col1, col2 = st.columns([1, 12])
with col1:
    st.image("logo.png", width=2000)
with col2:
    st.title("AEGIS — Escalation Detection Demo")
st.caption(
    "Enter a country name and click Generate plot to see the ACLED-based Escalation Index. "
    "The index combines five leading indicators — event frequency acceleration, explosions, "
    "strategic developments, civil unrest, and civilian targeting — into a single 0–100 score."
)

# ── Ticker bar ───────────────────────────────────────────────────────────────
_ticker_items = fetch_ticker_data()
if _ticker_items:
    try:
        _ticker_month = fetch_acled_arcgis_monthly()["event_month"].max().strftime("%b %Y")
    except Exception:
        _ticker_month = ""
    render_ticker(_ticker_items, _ticker_month)


# ----------------------------
# Escalation Index plot section
# ----------------------------
st.subheader("Escalation Index")

st.caption(
    "Depending on the date range selected, index plotting time may range from a couple seconds to a couple minutes. "
    "One page of events (5,000 events) takes ~4 seconds to load."
)

if run_btn:
    selected_country = str(country_name).strip()
    if not selected_country:
        st.warning("Please enter a country name in the sidebar.")
    else:
        try:
            _prog = st.progress(0, text=f"Connecting to ACLED…")

            if use_api:
                # ── Live-paging fetch so bar advances each page ───────────────
                _cache_key = f"acled_{selected_country}_{plot_start_date.year}_{plot_end_date.year}"
                if _cache_key not in st.session_state:
                    token   = _get_acled_bearer_token(acled_api_email, acled_api_key)
                    headers = {"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
                    end_yr  = min(plot_end_date.year, pd.Timestamp.now().year)
                    all_rows, page, page_size = [], 1, 5000
                    PAGE_CAP = 60  # hard cap; each page = 5,000 events

                    while True:
                        pct = min(5 + int((page - 1) / PAGE_CAP * 65), 70)
                        _prog.progress(pct, text=f"Fetching page {page} of ~{PAGE_CAP} ({len(all_rows):,} events so far)…")
                        r = requests.get(
                            ACLED_API_URL,
                            params={"country": selected_country,
                                    "year": f"{plot_start_date.year}|{end_yr}",
                                    "year_where": "BETWEEN",
                                    "fields": "event_date|country|event_type|fatalities|latitude|longitude",
                                    "limit": page_size, "page": page},
                            headers=headers, timeout=120,
                        )
                        r.raise_for_status()
                        data = r.json().get("data", [])
                        if not data:
                            break
                        all_rows.extend(data)
                        if len(data) < page_size:
                            break
                        page += 1
                        if page > PAGE_CAP:
                            break

                    _prog.progress(70, text="Processing events…")
                    st.session_state[_cache_key] = _process_acled_rows(all_rows, selected_country)

                df_acled   = st.session_state[_cache_key]
                data_label = "ACLED API (full history)"
            else:
                _prog.progress(30, text="Loading ArcGIS layer…")
                df_acled   = fetch_acled_arcgis_monthly()
                data_label = "ACLED ArcGIS layer (~13 months)"

            _prog.progress(80, text="Computing Escalation Index…")

            if df_acled.empty:
                _prog.empty()
                st.error(
                    f"No data found for **{selected_country}**. "
                    "Check spelling and capitalisation (e.g. 'Ukraine', not 'ukraine'). "
                    "Country names must match ACLED exactly."
                )
            else:
                st.caption(f"Data source: {data_label} · {len(df_acled):,} country-month rows loaded ({df_acled['event_month'].min().strftime('%b %Y') if not df_acled.empty else '?'} – {df_acled['event_month'].max().strftime('%b %Y') if not df_acled.empty else '?'})")
                idx_df = compute_escalation_index(df_acled, selected_country)
                _prog.progress(95, text="Rendering chart…")

                if idx_df.empty:
                    _prog.empty()
                    st.warning(f"No monthly data could be computed for {selected_country}.")
                else:
                        import matplotlib.pyplot as plt
                        import matplotlib.dates as mdates

                        w = int(smooth_window)
                        idx_df["index_smoothed"] = (
                            idx_df["escalation_index"]
                            .rolling(window=w, min_periods=1)
                            .mean()
                        )

                        # Apply date range filter
                        if plot_start_date <= plot_end_date:
                            idx_df = idx_df[
                                (idx_df["event_month"].dt.date >= plot_start_date)
                                & (idx_df["event_month"].dt.date <= plot_end_date)
                            ].copy()

                        if idx_df.empty:
                            st.warning(
                                f"No data for {selected_country} between "
                                f"{plot_start_date} and {plot_end_date}. "
                                "Try widening the date range in Advanced Settings."
                            )
                        else:
                            # Save to session_state so widget reruns don't wipe the chart
                            st.session_state["aegis_plot"] = {
                                "idx_df":               idx_df,
                                "selected_country":     selected_country,
                                "escalation_threshold": escalation_threshold,
                                "w":                    w,
                                "show_components":      show_components,
                                "data_label":           data_label,
                                "n_rows":               len(df_acled),
                                "date_min":             df_acled["event_month"].min().strftime("%b %Y"),
                                "date_max":             df_acled["event_month"].max().strftime("%b %Y"),
                            }
                            st.session_state["aegis_prog"] = _prog
        except Exception as e:
            _prog.empty()
            st.error(f"Error computing escalation index: {e}")


# ----------------------------
# Escalation plot render — separate from compute so widget
# interactions don't collapse the chart on rerun
# ----------------------------
if "aegis_plot" in st.session_state:
    _ps = st.session_state["aegis_plot"]
    idx_df               = _ps["idx_df"]
    selected_country     = _ps["selected_country"]
    escalation_threshold = _ps["escalation_threshold"]
    w                    = _ps["w"]
    show_components      = _ps["show_components"]
    data_label           = _ps["data_label"]

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


    dates     = pd.to_datetime(idx_df["event_month"])
    esc_rows  = idx_df[idx_df["index_smoothed"] > escalation_threshold]

    # ── Pre-escalation warning signal ─────────────
    # Fires when index is BELOW threshold but:
    #   - strategic + explosion components both elevated (sum > 0.9)
    #   - index has been rising for 2+ consecutive months
    import numpy as np
    idx_df = idx_df.copy()
    idx_df["_rising"] = (idx_df["index_smoothed"].diff() > 0).astype(int)
    idx_df["_lead_signal"] = idx_df["c_strategic"] + idx_df["c_explosion"]
    # Fire when index is below threshold but approaching it (within 20pts),
    # at least one leading component is elevated (>50th pct), and index rising
    warn_rows = idx_df[
        (idx_df["index_smoothed"] < escalation_threshold)
        & (idx_df["index_smoothed"] > escalation_threshold - 20)
        & (
            (idx_df["c_strategic"] > 0.25)
            | (idx_df["c_explosion"] > 0.25)
        )
        & (idx_df["_rising"] == 1)
    ]

    # ── 3-month forecast (linear trend on last 6 months) ──
    forecast_dates, forecast_vals, forecast_lo, forecast_hi = [], [], [], []
    if len(idx_df) >= 6:
        tail   = idx_df.tail(6).copy()
        tail_x = np.arange(len(tail))
        tail_y = tail["index_smoothed"].values
        coeffs = np.polyfit(tail_x, tail_y, 1)
        resid  = tail_y - np.polyval(coeffs, tail_x)
        std_err = resid.std()
        last_date = pd.to_datetime(idx_df["event_month"].iloc[-1])
        for i in range(1, 4):
            fv = float(np.clip(np.polyval(coeffs, len(tail) - 1 + i), 0, 100))
            forecast_dates.append(last_date + pd.DateOffset(months=i))
            forecast_vals.append(fv)
            forecast_lo.append(max(0,   fv - 1.5 * std_err))
            forecast_hi.append(min(100, fv + 1.5 * std_err))

    # ── Main escalation index chart (Plotly interactive) ──
    import plotly.graph_objects as go

    def _drill_hover(row):
        return (
            f"<b>{pd.to_datetime(row['event_month']).strftime('%b %Y')}</b><br>"
            f"Index: {row['escalation_index']:.1f} &nbsp;|&nbsp; Smoothed: {row['index_smoothed']:.1f}<br>"
            f"<br>"
            f"Battles: <b>{int(row.get('battles',0)):,}</b><br>"
            f"Explosions: <b>{int(row.get('explosions_remote_violence',0)):,}</b><br>"
            f"Strategic devs: <b>{int(row.get('strategic_developments',0)):,}</b><br>"
            f"Protests: <b>{int(row.get('protests',0)):,}</b><br>"
            f"Riots: <b>{int(row.get('riots',0)):,}</b><br>"
            f"Civ. violence: <b>{int(row.get('violence_against_civilians',0)):,}</b><br>"
            f"<br>Fatalities: <b>{int(row.get('fatalities',0)):,}</b>"
        )

    pfig = go.Figure()

    # Threshold shading
    pfig.add_hrect(
        y0=escalation_threshold, y1=105,
        fillcolor="#ef4444", opacity=0.06,
        layer="below", line_width=0,
    )

    # Threshold line
    pfig.add_hline(
        y=escalation_threshold,
        line_dash="dash", line_color="#ef4444", line_width=1.5,
        annotation_text=f"Alert threshold ({escalation_threshold})",
        annotation_font_color="#ef4444",
        annotation_position="bottom right",
    )

    # Raw index (faint)
    pfig.add_trace(go.Scatter(
        x=dates, y=idx_df["escalation_index"],
        mode="lines",
        line=dict(color="#60a5fa", width=1),
        opacity=0.25,
        name="Raw ACLED Index",
        hoverinfo="skip",
    ))

    # Smoothed index line with hover
    pfig.add_trace(go.Scatter(
        x=dates, y=idx_df["index_smoothed"],
        mode="lines",
        line=dict(color="#60a5fa", width=2.5),
        name=f"Escalation Index ({w}-mo smoothed)",
        customdata=idx_df.index,
        hovertemplate=[_drill_hover(r) for _, r in idx_df.iterrows()],
    ))

    # Escalation flagged dots
    if not esc_rows.empty:
        pfig.add_trace(go.Scatter(
            x=pd.to_datetime(esc_rows["event_month"]),
            y=esc_rows["index_smoothed"],
            mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="circle"),
            name=f"Escalation flagged ({len(esc_rows)} months)",
            hovertemplate=[
                "<b>🔴 ESCALATION FLAGGED</b><br>" + _drill_hover(r)
                for _, r in esc_rows.iterrows()
            ],
        ))

    # Pre-escalation warning diamonds
    if not warn_rows.empty:
        pfig.add_trace(go.Scatter(
            x=pd.to_datetime(warn_rows["event_month"]),
            y=warn_rows["index_smoothed"],
            mode="markers",
            marker=dict(color="#f97316", size=13, symbol="diamond"),
            name=f"Pre-escalation warning ({len(warn_rows)} months)",
            hovertemplate=[
                "<b>🟠 PRE-ESCALATION WARNING</b><br>"
                "Index below threshold but leading indicators elevated.<br>"
                + _drill_hover(r)
                for _, r in warn_rows.iterrows()
            ],
        ))

    # Forecast line + band
    if forecast_dates:
        last_hist_val = float(idx_df["index_smoothed"].iloc[-1])
        fc_x = [pd.to_datetime(idx_df["event_month"].iloc[-1])] + forecast_dates
        fc_y = [last_hist_val] + forecast_vals
        fc_lo_all = [last_hist_val] + forecast_lo
        fc_hi_all = [last_hist_val] + forecast_hi

        pfig.add_trace(go.Scatter(
            x=fc_x + fc_x[::-1],
            y=fc_hi_all + fc_lo_all[::-1],
            fill="toself", fillcolor="rgba(167,139,250,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        pfig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines+markers",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            marker=dict(color="#a78bfa", size=7),
            name="3-month forecast",
            hovertemplate=[
                f"<b>Forecast: {d.strftime('%b %Y')}</b><br>"
                f"Projected index: <b>{v:.1f}</b><br>"
                f"Range: {lo:.1f} – {hi:.1f}<extra></extra>"
                for d, v, lo, hi in zip(fc_x, fc_y, fc_lo_all, fc_hi_all)
            ],
        ))

    pfig.update_layout(
        title=dict(
            text=f"AEGIS Escalation Index — {selected_country}",
            font=dict(color="white", size=16),
        ),
        paper_bgcolor="#020617",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        xaxis=dict(
            gridcolor="#1e293b", showgrid=True,
            tickformat="%b %Y", tickangle=-35,
        ),
        yaxis=dict(
            gridcolor="#1e293b", showgrid=True,
            range=[0, 105], title="Escalation Index (0–100)",
        ),
        legend=dict(
            bgcolor="#1e293b", bordercolor="#334155",
            borderwidth=1, font=dict(color="white", size=11),
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1e293b", bordercolor="#334155",
            font=dict(color="white", size=12),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        height=500,
    )

    st.plotly_chart(pfig, use_container_width=True)
    if "aegis_prog" in st.session_state:
        try:
            st.session_state["aegis_prog"].progress(100, text="Complete.")
            import time; time.sleep(0.6)
            st.session_state["aegis_prog"].empty()
            del st.session_state["aegis_prog"]
        except Exception:
            pass

    # ── How to read the signals ───────────────────
    with st.expander("How to read the signals", expanded=False):
        st.markdown(
            "**Blue line** \u2014 smoothed Escalation Index (0\u2013100). "
            "Combines 6 conflict indicators into one score.\n\n"
            "**Red dots** \u2014 months where the smoothed index exceeded "
            "your alert threshold. Indicates sustained elevated conflict.\n\n"
            "**Orange diamonds** \u2014 pre-escalation warning. The index is "
            "still *below* the threshold, but strategic developments and "
            "explosions/remote violence are both elevated, and the index "
            "has been rising for 2+ consecutive months. This signal fires "
            "*before* red dots appear \u2014 it is the early warning.\n\n"
            "**Purple dotted line** \u2014 3-month linear forecast based on "
            "the last 6 months of the smoothed index. The shaded band shows "
            "the uncertainty range. Treat as a directional signal, not a "
            "precise prediction."
        )

    # ── Month drill-down ─────────────────────────
    with st.expander("Drill down — what drove a specific month?", expanded=True):
        st.markdown("### Drill down — what drove a specific month?")

        # Build option list: flagged months first, then all months
        all_month_labels = idx_df["event_month"].dt.strftime("%b %Y").tolist()
        flagged_labels   = esc_rows["event_month"].dt.strftime("%b %Y").tolist()
        warn_labels      = warn_rows["event_month"].dt.strftime("%b %Y").tolist()

        # Tag each option so user knows its status
        def _tag(m):
            if m in flagged_labels:   return f"🔴 {m} — escalation flagged"
            if m in warn_labels:      return f"🟠 {m} — pre-escalation warning"
            return f"⬜ {m}"

        options = [_tag(m) for m in all_month_labels]
        # Default to highest-scoring flagged month
        if flagged_labels:
            best_month = (
                esc_rows.loc[esc_rows["index_smoothed"].idxmax(), "event_month"]
                .strftime("%b %Y")
            )
            default_idx = all_month_labels.index(best_month)
        else:
            default_idx = len(options) - 1

        selected_opt = st.selectbox(
            "Select a month to inspect:",
            options=options,
            index=default_idx,
            key="drilldown_month",
        )
        # Extract the plain month label back out
        selected_label = selected_opt.split("—")[0].strip().lstrip("🔴🟠⬜ ")

        drill_row = idx_df[
            idx_df["event_month"].dt.strftime("%b %Y") == selected_label
        ]

        if not drill_row.empty:
            dr = drill_row.iloc[0]

            # Determine status tag
            if selected_label in flagged_labels:
                status_html = '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700;">ESCALATION FLAGGED</span>'
            elif selected_label in warn_labels:
                status_html = '<span style="background:#f97316;color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700;">PRE-ESCALATION WARNING</span>'
            else:
                status_html = '<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:12px;">BELOW THRESHOLD</span>'

            st.markdown(
                f"**{selected_label}** &nbsp; {status_html} &nbsp; "
                f"Index: **{dr['escalation_index']:.1f}** (smoothed: **{dr['index_smoothed']:.1f}**)",
                unsafe_allow_html=True,
            )

            # 6 event-type counts with context sentences
            event_types = [
                ("battles",                    "Battles",                    "#ef4444", 30,
                 "Direct armed confrontations between organised forces."),
                ("explosions_remote_violence", "Explosions / Remote violence","#f97316", 20,
                 "Shelling, airstrikes, IEDs, drone strikes. Often precedes ground battles."),
                ("strategic_developments",     "Strategic developments",      "#60a5fa", 15,
                 "Troop movements, HQ changes, peace deal collapses, ceasefires."),
                ("protests",                   "Protests",                    "#a78bfa", 10,
                 "Non-violent demonstrations. Social unrest often precedes armed conflict."),
                ("riots",                      "Riots",                       "#fde047", 10,
                 "Violent but non-armed demonstrations and looting."),
                ("violence_against_civilians", "Violence vs. civilians",      "#f59e0b",  5,
                 "Targeted attacks on non-combatants. Signals strategic deterioration."),
            ]

            # Find max count for bar scaling
            max_count = max(int(dr.get(col, 0)) for col, *_ in event_types) or 1

            cols_drill = st.columns(2)
            for i, (col, label, color, weight, desc) in enumerate(event_types):
                count = int(dr.get(col, 0))
                pct   = int(count / max_count * 100)
                with cols_drill[i % 2]:
                    st.markdown(
                        f"<div style='margin-bottom:12px;'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                        f"<span style='color:{color};font-weight:600;font-size:13px;'>{label}</span>"
                        f"<span style='color:#94a3b8;font-size:13px;'><b style='color:white;'>{count:,}</b> events &nbsp;·&nbsp; {weight}% weight</span>"
                        f"</div>"
                        f"<div style='background:#1e293b;border-radius:4px;height:8px;'>"
                        f"<div style='background:{color};width:{pct}%;height:8px;border-radius:4px;'></div>"
                        f"</div>"
                        f"<div style='color:#64748b;font-size:11px;margin-top:3px;'>{desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Fatalities callout
            fat = int(dr.get("fatalities", 0))
            total = int(dr.get("total_events", 0))
            st.markdown(
                f"<div style='background:#1e293b;border:1px solid #334155;border-radius:6px;"
                f"padding:10px 16px;margin-top:4px;display:flex;gap:32px;'>"
                f"<span style='color:#94a3b8;font-size:13px;'>Total events: "
                f"<b style='color:white;font-size:16px;'>{total:,}</b></span>"
                f"<span style='color:#94a3b8;font-size:13px;'>Recorded fatalities: "
                f"<b style='color:#ef4444;font-size:16px;'>{fat:,}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    # ── Component breakdown stacked bar ───────────
    if show_components:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("#020617")
        ax2.set_facecolor("#0f172a")

        component_data = {
            "Conflict intensity (30%)":  idx_df["c_intensity"] * 30,
            "Event accel. (20%)":        idx_df["c_accel"]     * 20,
            "Explosions (20%)":          idx_df["c_explosion"] * 20,
            "Strategic devs (15%)":      idx_df["c_strategic"] * 15,
            "Unrest / Protests (10%)":   idx_df["c_unrest"]    * 10,
            "Civilian targeting (5%)":   idx_df["c_civilian"]  * 5,
        }
        colors = ["#ef4444", "#f59e0b", "#f97316", "#60a5fa", "#a78bfa", "#fde047"]

        bottom = np.zeros(len(idx_df))
        for (label, values), color in zip(component_data.items(), colors):
            ax2.bar(
                dates, values.values, bottom=bottom,
                color=color, alpha=0.85, label=label, width=20,
            )
            bottom += values.values

        ax2.set_title(
            "Index Component Breakdown (weighted contribution per month)",
            color="white", fontsize=13, pad=10,
        )
        ax2.set_xlabel("Month", color="#94a3b8")
        ax2.set_ylabel("Weighted score", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=35, ha="right")
        ax2.grid(True, alpha=0.12, color="#334155", axis="y")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#334155")
        ax2.legend(
            facecolor="#1e293b", edgecolor="#334155",
            labelcolor="white", fontsize=9, loc="upper left",
        )
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    # ── Flagged months table ──────────────────────
    _flag_label = (
        f"Flagged escalation months ({len(esc_rows)})"
        if not esc_rows.empty else
        "Flagged escalation months (0)"
    )
    with st.expander(_flag_label, expanded=False):
        if esc_rows.empty:
            st.info(
                f"No months exceeded the threshold of {escalation_threshold}. "
                "Try lowering the threshold in Advanced Settings."
            )
        else:
            disp = esc_rows[[
                "event_month", "escalation_index", "index_smoothed",
                "battles", "explosions_remote_violence",
                "strategic_developments", "protests", "riots",
                "violence_against_civilians", "fatalities",
            ]].copy()
            disp["event_month"]      = disp["event_month"].dt.strftime("%b %Y")
            disp["escalation_index"] = disp["escalation_index"].round(1)
            disp["index_smoothed"]   = disp["index_smoothed"].round(1)
            disp = disp.rename(columns={
                "event_month":                "Month",
                "escalation_index":           "Raw Index",
                "index_smoothed":             "Smoothed Index",
                "battles":                    "Battles",
                "explosions_remote_violence": "Explosions",
                "strategic_developments":     "Strategic Devs",
                "protests":                   "Protests",
                "riots":                      "Riots",
                "violence_against_civilians": "Civ. Violence",
                "fatalities":                 "Fatalities",
            })
            st.dataframe(
                disp.sort_values("Smoothed Index", ascending=False),
                use_container_width=True,
            )

    with st.expander("Full monthly index data"):
        full = idx_df[[
            "event_month", "escalation_index", "index_smoothed",
            "total_events", "battles", "explosions_remote_violence",
            "strategic_developments", "protests", "riots",
            "violence_against_civilians", "fatalities",
        ]].copy()
        full["event_month"]      = full["event_month"].dt.strftime("%b %Y")
        full["escalation_index"] = full["escalation_index"].round(1)
        full["index_smoothed"]   = full["index_smoothed"].round(1)
        st.dataframe(full, use_container_width=True)

    st.caption(
        "Index formula: 30% Raw Conflict Intensity (Battles + Explosions) + "
        "20% Event Frequency Acceleration + "
        "20% Explosions/Remote Violence + "
        "15% Strategic Developments + "
        "10% Civil Unrest (Protests + Riots) + "
        "5% Civilian Targeting Ratio. "
        "All components normalised globally (0–1) across all ACLED countries and months. "
        "Source: ACLED via public ArcGIS layer (updated weekly)."
    )
elif not run_btn:
    st.info(
        "Enter a country name in the sidebar (e.g. **Ukraine**, **Sudan**, **Myanmar**) "
        "and click **Generate plot**. The interactive map appears below."
    )



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
    return category_map[max(vals, key=vals.get)]


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
                    "battles", "explosions_remote_violence",
                    "violence_against_civilians", "strategic_developments",
                ]
                latest_month   = df_map["event_month"].max()
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
                        help="Explosions / remote violence is closest to airstrikes, missile strikes, and shelling.",
                    )
                    show_only_violent = st.checkbox(
                        "Hide rows with no violent activity",
                        value=True,
                        help="Filters rows where battles, explosions, violence against civilians, and strategic developments are all zero.",
                    )
                    only_selected_country = st.checkbox(
                        "Only show the country entered above",
                        value=False,
                    )
                    size_max = st.slider("Maximum marker size", min_value=10, max_value=45, value=24, step=1)
                    auto_refresh_map = st.checkbox(
                        "Auto-refresh map", value=True,
                        help="Reload the app on a timer so the map picks up new layer updates.",
                    )
                    refresh_minutes = st.slider(
                        "Auto-refresh interval (minutes)",
                        min_value=15, max_value=180, value=60, step=15,
                        disabled=not auto_refresh_map,
                    )

                if override_map_dates:
                    default_start = max(
                        earliest_month.date(),
                        (latest_month - pd.DateOffset(months=2)).date(),
                    )
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
                        start_dt = end_dt = latest_month.date()
                        df_map = df_map[df_map["event_month"].dt.date == latest_month.date()]
                else:
                    start_dt = end_dt = latest_month.date()
                    df_map = df_map[df_map["event_month"].dt.date == latest_month.date()]

                if only_selected_country:
                    df_map = df_map[
                        df_map["country"].astype(str).str.strip() == str(country_name).strip()
                    ]

                if show_only_violent:
                    df_map = df_map[df_map[violent_cols].fillna(0).sum(axis=1) > 0]

                if df_map.empty:
                    st.info("No rows matched the current map filters.")
                else:
                    grouped = (
                        df_map.groupby(
                            ["country", "admin1", "centroid_latitude", "centroid_longitude"],
                            as_index=False,
                        )[[
                            "battles", "explosions_remote_violence", "protests", "riots",
                            "strategic_developments", "violence_against_civilians",
                            "violent_actors", "fatalities",
                        ]]
                        .sum()
                    )

                    grouped["dominant_category"] = grouped.apply(_build_dominant_category, axis=1)
                    grouped["metric_value"] = pd.to_numeric(
                        grouped[selected_metric], errors="coerce"
                    ).fillna(0)
                    grouped = grouped[grouped["metric_value"] > 0].copy()

                    if grouped.empty:
                        st.info("No positive values for the selected map metric.")
                    else:
                        grouped["admin1"] = grouped["admin1"].fillna("Unknown")
                        raw = grouped["metric_value"].clip(lower=1)
                        grouped["bubble_size"] = raw.clip(lower=raw.max() * 0.01)
                        grouped["hover_location"] = grouped["admin1"] + ", " + grouped["country"]

                        st.caption(
                            f"Source: public ACLED ArcGIS monthly indicators. "
                            f"Showing {metric_labels[selected_metric]} from {start_dt} to {end_dt}."
                        )
                        st.caption(
                            "This layer is monthly aggregated at the subnational level. "
                            "Working towards individual strike-by-strike live telemetry."
                        )
                        st.caption(
                            "🗺️ **How to read this map:** Bubble **color** shows the *dominant conflict category* "
                            "for each region — the event type that occurred most frequently. Bubble **size** reflects "
                            "the *selected map metric*. These two can differ: a region colored red for Battles may still "
                            "show a high Explosions count if that's the metric you've selected — both occurred in that "
                            "region during the same period. Use the metric selector to surface different dimensions of "
                            "the same underlying conflict data."
                        )

                        # ── Session state for focused country ─────────────
                        if "map_focused_country" not in st.session_state:
                            st.session_state["map_focused_country"] = None
                        focused = st.session_state["map_focused_country"]

                        # ── Country picker ────────────────────────────────
                        country_list = sorted(grouped["country"].dropna().unique().tolist())
                        col_pick, col_reset = st.columns([5, 1])
                        with col_pick:
                            sel = st.selectbox(
                                "Select a country to zoom and see details",
                                ["— World view —"] + country_list,
                                index=(
                                    0 if focused is None
                                    else (country_list.index(focused) + 1 if focused in country_list else 0)
                                ),
                                label_visibility="collapsed",
                                key=f"cpick_{selected_metric}_{start_dt}",
                            )
                            new_focus = None if sel == "— World view —" else sel
                            if new_focus != focused:
                                st.session_state["map_focused_country"] = new_focus
                                focused = new_focus
                                st.rerun()
                        with col_reset:
                            if focused and st.button("✕ Reset", use_container_width=True):
                                st.session_state["map_focused_country"] = None
                                focused = None
                                st.rerun()

                        # ── Map center / zoom ─────────────────────────────
                        if focused:
                            c_rows = grouped[grouped["country"] == focused]
                            if not c_rows.empty:
                                clat = (c_rows["centroid_latitude"].min() + c_rows["centroid_latitude"].max()) / 2
                                clon = (c_rows["centroid_longitude"].min() + c_rows["centroid_longitude"].max()) / 2
                                span = max(
                                    c_rows["centroid_latitude"].max() - c_rows["centroid_latitude"].min(),
                                    c_rows["centroid_longitude"].max() - c_rows["centroid_longitude"].min(),
                                    0.5,
                                )
                                map_center = {"lat": float(clat), "lon": float(clon)}
                                map_zoom   = float(max(1.5, min(6.0, 5.8 - np.log2(span + 1))))
                            else:
                                map_center, map_zoom = {"lat": 20, "lon": 10}, 1.0
                        else:
                            map_center, map_zoom = {"lat": 20, "lon": 10}, 1.0

                        # ── Layout ────────────────────────────────────────
                        if focused:
                            map_col, panel_col = st.columns([3, 1], gap="small")
                        else:
                            map_col  = st.container()
                            panel_col = None

                        with map_col:
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
                                    "dominant_category": False,
                                },
                                mapbox_style="carto-darkmatter",
                                center=map_center,
                                zoom=map_zoom,
                                title="Current Conflict-Related Hotspots",
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
                                    "<b style='font-size:15px'>%{hovertext}</b><br>"
                                    + "<span style='color:rgba(255,255,255,0.55);font-size:12px;'>"
                                    + "Dominant category: %{customdata[14]}</span><br><br>"
                                    + f"{metric_labels[selected_metric]}: %{{customdata[0]:,}}<br>"
                                    + "Fatalities: %{customdata[1]:,}"
                                    + "<extra></extra>"
                                ),
                            )
                            map_h = 760 if focused else 790
                            fig.update_layout(
                                paper_bgcolor="#020617",
                                plot_bgcolor="#020617",
                                font=dict(color="white"),
                                title=dict(
                                    text="Current Conflict-Related Hotspots",
                                    x=0.5, xanchor="center",
                                    y=0.98, yanchor="top",
                                    font=dict(color="white", size=20),
                                ),
                                legend=dict(
                                    title=dict(text="<b>Categories:</b>", side="top", font=dict(color="white", size=12)),
                                    orientation="h",
                                    yanchor="bottom", y=-0.10,
                                    xanchor="left", x=0,
                                    bgcolor="rgba(2,6,23,0)",
                                    font=dict(color="white", size=12),
                                ),
                                margin=dict(l=0, r=0, t=55, b=75),
                                height=map_h,
                                hoverlabel=dict(bgcolor="rgba(20,20,20,0.95)", font_size=13, font_family="Arial"),
                            )
                            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

                        # ── Country info panel ────────────────────────────
                        if focused and panel_col is not None:
                            with panel_col:
                                country_rows = grouped[grouped["country"] == focused]
                                if not country_rows.empty:
                                    t = country_rows[[
                                        "battles", "explosions_remote_violence", "protests",
                                        "riots", "strategic_developments", "violence_against_civilians",
                                        "violent_actors", "fatalities",
                                    ]].sum()

                                    def _card(color, bg, border, val, lbl):
                                        return (
                                            f'<div style="background:{bg};border:1px solid {border};'
                                            f'border-radius:8px;padding:10px 6px;text-align:center;margin-bottom:8px;">'
                                            f'<div style="font-size:20px;font-weight:800;color:{color};">{val:,}</div>'
                                            f'<div style="font-size:9px;color:rgba(255,255,255,0.5);margin-top:3px;letter-spacing:.8px;">{lbl}</div>'
                                            f'</div>'
                                        )

                                    panel_html = (
                                        f'<div style="background:linear-gradient(160deg,rgba(10,18,38,0.97),rgba(22,35,60,0.97));'
                                        f'border-left:3px solid #f59e0b;border-radius:10px;padding:16px 14px;'
                                        f'font-family:Arial,sans-serif;color:white;">'
                                        f'<div style="font-size:17px;font-weight:800;margin-bottom:2px;">&#127758; {focused}</div>'
                                        f'<div style="font-size:9px;color:rgba(255,255,255,0.35);margin-bottom:14px;letter-spacing:1px;">ACLED CONFLICT DATA</div>'
                                        + _card("#ef4444","rgba(239,68,68,0.13)","rgba(239,68,68,0.3)",    int(t["fatalities"]),               "FATALITIES")
                                        + _card("#f59e0b","rgba(245,158,11,0.13)","rgba(245,158,11,0.3)",  int(t["explosions_remote_violence"]), "EXPLOSIONS")
                                        + _card("#f87171","rgba(248,113,113,0.13)","rgba(248,113,113,0.3)",int(t["battles"]),                   "BATTLES")
                                        + _card("#fde047","rgba(253,224,71,0.13)","rgba(253,224,71,0.3)",  int(t["violence_against_civilians"]), "CIV. VIOLENCE")
                                        + _card("#60a5fa","rgba(96,165,250,0.13)","rgba(96,165,250,0.3)",  int(t["strategic_developments"]),    "STRATEGIC")
                                        + _card("#a78bfa","rgba(167,139,250,0.13)","rgba(167,139,250,0.3)",int(t["protests"]),                  "PROTESTS")
                                        + _card("#f472b6","rgba(244,114,182,0.13)","rgba(244,114,182,0.3)",int(t["riots"]),                     "RIOTS")
                                        + _card("white","rgba(255,255,255,0.05)","rgba(255,255,255,0.1)",  int(t["violent_actors"]),            "ACTORS")
                                        + '</div>'
                                    )
                                    st.markdown(panel_html, unsafe_allow_html=True)

                                    st.markdown(
                                        '<div style="font-size:13px;font-weight:700;color:white;margin:14px 0 8px;">&#128240; Recent News</div>',
                                        unsafe_allow_html=True,
                                    )
                                    try:
                                        news = load_country_news(focused, max_items=5)
                                        if news:
                                            for article in news:
                                                age = format_news_age(article["published_dt"])
                                                meta = " · ".join(p for p in [article["source"], age] if p)
                                                title_safe = article["title"]
                                                link_safe  = article["link"]
                                                st.markdown(
                                                    f"[{title_safe}]({link_safe})  \n"
                                                    f"<span style='opacity:0.45;font-size:10px;'>{meta}</span>",
                                                    unsafe_allow_html=True,
                                                )
                                                st.markdown(
                                                    "<hr style='border-color:rgba(255,255,255,0.06);margin:6px 0;'>",
                                                    unsafe_allow_html=True,
                                                )
                                        else:
                                            st.caption("No recent news found.")
                                    except Exception:
                                        st.caption("Could not load news.")

                        summary_cols = [
                            "country", "admin1", "metric_value", "fatalities",
                            "battles", "explosions_remote_violence", "violence_against_civilians",
                        ]
                        if selected_metric in {
                            "battles", "explosions_remote_violence",
                            "violence_against_civilians", "fatalities",
                        }:
                            summary_cols = [c for c in summary_cols if c != selected_metric]

                        top_hotspots = (
                            grouped.sort_values("metric_value", ascending=False)[summary_cols]
                            .head(25)
                            .copy()
                        )
                        top_hotspots = top_hotspots.rename(columns={
                            "country": "Country",
                            "admin1": "Admin1",
                            "metric_value": f"Selected metric ({metric_labels[selected_metric]})",
                            "fatalities": "Fatalities",
                            "battles": "Battles",
                            "explosions_remote_violence": "Explosions / remote violence",
                            "violence_against_civilians": "Violence against civilians",
                        })

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


# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
<div style="text-align:center; color:#e2e8f0; font-size:12px; padding: 8px 0 16px 0;">
    © 2026 Alexander Armand-Blumberg · AEGIS
</div>
""",
    unsafe_allow_html=True,
)
