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

# ----------------------------
# Claude AI helper
# ----------------------------
def _call_claude(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    """Call Groq (Llama 3) API. Returns text or error string."""
    try:
        api_key = st.secrets["groq"]["api_key"]
    except Exception:
        return "⚠️ Groq API key not configured. Add it to .streamlit/secrets.toml as [groq] api_key = '...'"
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": "llama-3.1-8b-instant",
            "max_tokens": max_tokens,
            "messages": messages,
        }
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json=body,
            timeout=30,
        )
        if not r.ok:
            return f"⚠️ API error {r.status_code}: {r.text}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ AI analysis unavailable: {e}"

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
    # Inject video as fixed background + style the real Streamlit buttons
    LANDING_VIDEO = Path("landing.mp4")
    if LANDING_VIDEO.exists():
        v64 = base64.b64encode(open(LANDING_VIDEO, "rb").read()).decode()
        video_tag = f"""
        <video id="aegis-bg-video" autoplay loop muted playsinline
          style="position:fixed;inset:0;width:100%;height:100%;
                 object-fit:cover;opacity:0.42;
                 filter:grayscale(55%) contrast(1.1);z-index:0;pointer-events:none;">
          <source src="data:video/mp4;base64,{v64}" type="video/mp4">
        </video>"""
        has_video = True
    else:
        video_tag = ""
        has_video = False

    st.markdown(
        f"""
        <style>
        /* Hide all Streamlit chrome */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        header, footer, .stDeployButton {{ display: none !important; }}

        /* Full viewport, no scroll */
        html, body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main, section.main, .block-container {{
            overflow: hidden !important;
            padding: 0 !important; margin: 0 !important;
            max-width: 100vw !important; width: 100vw !important;
            height: 100vh !important; max-height: 100vh !important;
            background: #000 !important;
        }}

        /* Dark gradient overlay */
        [data-testid="stApp"]::after {{
            content: '';
            position: fixed; inset: 0; z-index: 1; pointer-events: none;
            background: linear-gradient(180deg,
              rgba(2,6,23,0.38) 0%, rgba(2,6,23,0.55) 50%, rgba(2,6,23,0.82) 100%);
        }}

        /* Push all landing content above overlay */
        [data-testid="stMain"] > div {{ position: relative; z-index: 2; }}

        /* Center everything vertically */
        .block-container {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100vh !important;
        }}

        /* Landing text */
        .landing-tag {{
            font-size: 10.5px; letter-spacing: 0.22em; color: #ef4444;
            font-weight: 700; text-transform: uppercase;
            font-family: -apple-system, 'Inter', sans-serif;
        }}
        .landing-title {{
            font-size: clamp(56px, 10vw, 118px); font-weight: 900; color: #fff;
            letter-spacing: -0.02em; line-height: 1;
            text-shadow: 0 0 80px rgba(0,0,0,0.9), 0 2px 6px rgba(0,0,0,0.8);
            font-family: -apple-system, 'Inter', sans-serif;
        }}
        .landing-sub {{
            font-size: clamp(11px, 1.4vw, 15px); color: #e2e8f0;
            letter-spacing: 0.18em; text-transform: uppercase;
            text-shadow: 0 1px 10px rgba(0,0,0,0.9);
            font-family: -apple-system, 'Inter', sans-serif;
        }}
        .landing-copy {{
            font-size: 11px; color: #94a3b8;
            font-family: -apple-system, 'Inter', sans-serif;
        }}

        /* Style the Streamlit buttons to look like the custom ones */
        [data-testid="stButton"] button {{
            font-size: 15px !important; font-weight: 600 !important;
            padding: 13px 34px !important; border-radius: 7px !important;
            letter-spacing: 0.03em !important; height: auto !important;
            transition: transform .15s, opacity .15s !important;
        }}
        [data-testid="stButton"] button:hover {{
            transform: translateY(-2px) !important; opacity: 0.90 !important;
        }}

        /* Page fade-in */
        @keyframes aegis-fadein {{
            from {{ opacity: 0; }}
            to   {{ opacity: 1; }}
        }}
        [data-testid="stMain"] > div {{
            animation: aegis-fadein 0.35s ease forwards;
        }}
        </style>
        {video_tag}
        """,
        unsafe_allow_html=True,
    )

    # Landing page content using real Streamlit widgets
    st.markdown(
        """
        <div style="text-align:center; padding: 20vh 24px 8px;">
          <div class="landing-tag">&#9632;&nbsp; Palantir Valley Forge Grant Demo</div>
          <div class="landing-title">AEGIS</div>
          <div class="landing-sub" style="margin-top:12px; margin-bottom:36px;">
            Advanced Early-Warning &nbsp;&amp;&nbsp; Geostrategic Intelligence System
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c1, c2, _ = st.columns([2, 1, 1, 2])
    with c1:
        if st.button("📊  Escalation Index", use_container_width=True, type="primary", key="land_index"):
            st.session_state["page"] = "index"
            st.rerun()
    with c2:
        if st.button("🗺️  Interactive Map", use_container_width=True, key="land_map"):
            st.session_state["page"] = "map"
            st.rerun()

    st.markdown(
        '<div style="text-align:center;" class="landing-copy">'
        '<a href="https://www.linkedin.com/in/alexanderbab/" target="_blank" style="color:inherit;">Alexander Armand-Blumberg</a> &middot; AEGIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="position:fixed;bottom:36px;left:16px;z-index:100;display:flex;align-items:center;gap:10px;">'
        '<button id="aegis-pause-btn" style="'
        'background:rgba(2,8,20,0.70);border:1px solid rgba(255,255,255,0.18);'
        'color:rgba(255,255,255,0.55);font-size:11px;letter-spacing:0.08em;'
        'padding:5px 11px;border-radius:5px;cursor:pointer;'
        'font-family:-apple-system,Inter,sans-serif;">⏸ PAUSE</button>'
        '<div style="width:90px;height:3px;background:rgba(255,255,255,0.12);border-radius:2px;overflow:hidden;">'
        '<div id="aegis-progress-bar" style="height:100%;width:0%;background:rgba(255,255,255,0.35);border-radius:2px;"></div>'
        '</div></div>'
        '<div style="position:fixed;bottom:14px;left:16px;font-size:9px;color:rgba(255,255,255,0.25);'
        'font-family:-apple-system,Inter,sans-serif;letter-spacing:0.05em;z-index:10;">'
        'Background footage: Public Domain (CC0)</div>',
        unsafe_allow_html=True,
    )

    if has_video:
        st.components.v1.html("""
<script>
(function poll() {
  const vid = window.parent.document.getElementById('aegis-bg-video');
  const bar = window.parent.document.getElementById('aegis-progress-bar');
  const btn = window.parent.document.getElementById('aegis-pause-btn');
  if (!vid || !bar || !btn) { setTimeout(poll, 100); return; }
  let paused = false;
  function updateBar() {
    if (vid.duration) bar.style.width = (vid.currentTime / vid.duration * 100) + '%';
    requestAnimationFrame(updateBar);
  }
  requestAnimationFrame(updateBar);
  btn.onclick = function() {
    paused = !paused;
    if (paused) { vid.pause(); btn.textContent = '▶ PLAY'; }
    else { vid.play(); btn.textContent = '⏸ PAUSE'; }
  };
  btn.onmouseenter = function() {
    btn.style.borderColor = 'rgba(255,255,255,0.45)';
    btn.style.color = 'rgba(255,255,255,0.85)';
  };
  btn.onmouseleave = function() {
    btn.style.borderColor = 'rgba(255,255,255,0.18)';
    btn.style.color = 'rgba(255,255,255,0.55)';
  };
})();
</script>
""", height=0, scrolling=False)
    
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

- Only have access to data from Jan 2018 to exactly one year ago for the Escalation Index, as I currently only have Researcher Tier ACLED access.
- Interactive map data has a 1-2 month lag due to my ACLED access tier.
- ACLED public ArcGIS layer for the map is monthly aggregated at subnational level, not individual events.
- Some countries may have sparse data in earlier months.
- Using Streamlit for Python, not an actual domain.

**Planned improvements**

- Get a higher ACLED Tier, giving me access to more data for Escalation Index. The code works the same with more up-to-date data.
- Direct ACLED API for weekly/event-level granularity
- Actor-level escalation detection
- ML-based index calibration against historical escalation outcomes
- Subnational index breakdown
"""
    )

with st.sidebar.expander("Background"):
    st.markdown(
        """
**Why AEGIS?**

I've always had a deep love of Greek mythology — so when naming this project, the choice felt natural. In myth, the Aegis was the divine shield of Zeus and Athena: a symbol of protection, foreknowledge, and strategic power. Athena carried it into battle not just as armor, but as an instrument of clarity — it was said to inspire terror in enemies and confidence in allies.

That felt like exactly the right metaphor. AEGIS the program is designed to do the same thing in the modern context: cut through the noise of global events, surface what matters, and give decision-makers the clarity to act before crises escalate. The shield doesn't fight the war — it tells you where the threat is coming from.

**Why conflict data?**

I've been fascinated by current affairs and geopolitics for as long as I can remember. Watching conflicts unfold in the news and wondering what the underlying patterns were — what signals preceded an escalation, what data existed but wasn't being surfaced — is what pushed me to build something that tries to answer those questions systematically.
"""
    )

st.sidebar.markdown("---")
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

if st.session_state.get("page") != "map":
  _adv_exp = st.sidebar.expander("Advanced Settings")
else:
  _adv_exp = None
if _adv_exp:
  with _adv_exp:
    escalation_threshold = st.slider(
        "Escalation alert threshold (0–100)",
        min_value=0,
        max_value=100,
        value=45,
        step=1,
        help="Recommended: Leave at 45",
    )
    smooth_window = st.number_input(
        "Smoothing window (months)",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
        help="Rolling average applied to reduce month-to-month noise. Recommended not to change.",
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

if st.session_state.get("page") != "map":
    run_btn = st.sidebar.button("Generate plot")
else:
    run_btn = False

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

if st.session_state.get("page") != "index":
    show_map = st.sidebar.checkbox(
        "Show interactive map",
        value=True,
        help="Turn the map section on/off.",
    )
    override_map_dates = st.sidebar.checkbox(
        "Override map date range",
        value=False,
        help="If off, the map automatically uses the latest month available.",
    )
else:
    show_map = False
    override_map_dates = False



# ----------------------------
# Main header
# ----------------------------
st.markdown("""
<style>
@keyframes aegis-fadein {
    from { opacity: 0; }
    to   { opacity: 1; }
}
[data-testid="stMain"] > div {
    animation: aegis-fadein 0.35s ease forwards;
}
</style>
""", unsafe_allow_html=True)

if st.button("← Back to Home", key="back_btn"):
    st.session_state["page"] = "landing"
    st.rerun()

# ----------------------------
# Live news feed
# ----------------------------
with st.expander("Recent conflict news", expanded=False):
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

col1, col2 = st.columns([1, 12])
with col1:
    st.image("logo.png", width=2000)
with col2:
    st.title("AEGIS — Escalation Detection Demo")
if st.session_state.get("page") == "map":
    st.caption(
        "Explore the interactive global conflict map powered by the ACLED ArcGIS public layer. "
        "Bubble color shows the dominant conflict category per region; bubble size reflects the selected metric. "
        "Updated weekly."
    )
else:
    st.caption(
        "WAIT A COUPLE SECONDS FOR THE SOFTWARE TO LOAD. Enter a country name and click Generate plot to see the ACLED-based Escalation Index. "
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
if st.session_state.get("page") != "map":
    st.subheader("Escalation Index")

    st.caption(
        "Depending on the date range selected, index plotting time may range from a couple seconds to a couple minutes. "
        "One page of events (5,000 events) takes ~4 seconds to load."
    )

if st.session_state.get("page") != "map" and run_btn:
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
if "aegis_plot" in st.session_state and st.session_state.get("page") != "map":
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

    # ── AI Analysis ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Intelligence Analysis")

    # Build a compact data summary to pass to Claude
    recent = idx_df.tail(6)
    latest = idx_df.iloc[-1]
    prev3  = idx_df.tail(4).iloc[0]
    trend_dir = "rising" if latest["index_smoothed"] > prev3["index_smoothed"] else "falling"
    peak_month = idx_df.loc[idx_df["index_smoothed"].idxmax(), "event_month"].strftime("%b %Y")
    peak_val   = idx_df["index_smoothed"].max()
    num_flagged = len(esc_rows)
    num_warned  = len(warn_rows)
    recent_summary = "\n".join(
        f"  {row['event_month'].strftime('%b %Y')}: index={row['index_smoothed']:.1f}, "
        f"battles={int(row.get('battles',0))}, explosions={int(row.get('explosions_remote_violence',0))}, "
        f"strategic={int(row.get('strategic_developments',0))}, protests={int(row.get('protests',0))}, "
        f"riots={int(row.get('riots',0))}, civ_violence={int(row.get('violence_against_civilians',0))}, "
        f"fatalities={int(row.get('fatalities',0))}"
        for _, row in recent.iterrows()
    )
    _data_latest = latest["event_month"].strftime("%b %Y")
    _ai_system = (
        "You are a concise geopolitical intelligence analyst. "
        "Write in plain English. Be specific and data-driven. No bullet lists — flowing prose only. "
        "Do not mention ACLED by name. Do not use phrases like 'the data shows' — just state the finding directly. "
        f"IMPORTANT: Your first sentence must always be a brief disclaimer that the data only extends to {_data_latest} "
        "due to access limitations, so the analysis reflects that period, not the present day. "
        "Keep this disclaimer to one short sentence, then continue with the analysis."
    )

    ai_tabs = st.tabs(["📊 Country Insight", "📈 Trend Interpretation", "⚖️ Comparative Analysis"])

    # ── Tab 1: Country Insight ─────────────────────
    with ai_tabs[0]:
        st.caption(f"AI-generated summary of {selected_country}'s escalation profile.")
        if st.button("Generate Country Insight", key="ai_country_btn"):
            with st.spinner("Analyzing..."):
                prompt = (
                    f"Analyze the conflict escalation situation in {selected_country} based on this index data.\n\n"
                    f"Overall trend: {trend_dir}. Peak index: {peak_val:.1f} in {peak_month}. "
                    f"Escalation flagged in {num_flagged} months. Pre-escalation warnings in {num_warned} months.\n\n"
                    f"Last 6 months of data:\n{recent_summary}\n\n"
                    f"Write 3 sentences after the disclaimer: "
                    f"(1) State the escalation trend clearly and connect it to real-world events you know happened in {selected_country} "
                    f"during this period — for example, a specific battle, ceasefire, peace deal, election, coup, or offensive "
                    f"that explains why the numbers moved the way they did. Be specific with event names and dates where possible. "
                    f"(2) Identify which event types (battles, explosions, protests, etc.) are driving the index and what that signals. "
                    f"(3) Based on where the index stood at {_data_latest}, what was the near-term risk outlook at that time."
                )
                result = _call_claude(prompt, system=_ai_system, max_tokens=300)
                st.markdown(
                    f"<div style='background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;"
                    f"padding:16px 20px;color:#e2e8f0;font-size:14px;line-height:1.7;'>{result}</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Trend Interpretation ───────────────
    with ai_tabs[1]:
        st.caption("Is this country escalating or de-escalating, and why?")
        if st.button("Interpret Trend", key="ai_trend_btn"):
            with st.spinner("Analyzing..."):
                prompt = (
                    f"Analyze the escalation trend for {selected_country}.\n\n"
                    f"The index is currently {trend_dir}. "
                    f"Current smoothed index: {latest['index_smoothed']:.1f}. "
                    f"3 months ago: {prev3['index_smoothed']:.1f}.\n\n"
                    f"Last 6 months:\n{recent_summary}\n\n"
                    f"Write 2-3 sentences after the disclaimer: (1) clearly state whether the country is escalating or "
                    f"de-escalating and connect it to a specific real-world event or development you know occurred in "
                    f"{selected_country} around this period — name it specifically. "
                    f"(2) identify which event types are driving the change. "
                    f"(3) flag any warning signs or positive signals in the most recent month of data."
                )
                result = _call_claude(prompt, system=_ai_system, max_tokens=300)
                st.markdown(
                    f"<div style='background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;"
                    f"padding:16px 20px;color:#e2e8f0;font-size:14px;line-height:1.7;'>{result}</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Comparative Analysis ───────────────
    with ai_tabs[2]:
        st.caption("Compare this country's escalation profile against another.")
        compare_country = st.text_input(
            "Country to compare against",
            placeholder="e.g. Nigeria, Sudan, Iraq",
            key="ai_compare_input",
        )
        if st.button("Compare Countries", key="ai_compare_btn") and compare_country:
            with st.spinner(f"Loading {compare_country} data and analyzing..."):
                try:
                    df_compare = fetch_acled_arcgis_monthly()
                    df_compare = df_compare[
                        df_compare["country"].str.strip().str.lower() == compare_country.strip().lower()
                    ]
                    if df_compare.empty:
                        st.warning(f"No data found for '{compare_country}'. Check the spelling.")
                    else:
                        idx_compare = compute_escalation_index(df_compare, compare_country)
                        if idx_compare.empty:
                            st.warning(f"Could not compute escalation index for '{compare_country}'.")
                        else:
                            c_latest    = idx_compare.iloc[-1]
                            c_prev3     = idx_compare.tail(4).iloc[0]
                            c_trend     = "rising" if c_latest["index_smoothed"] > c_prev3["index_smoothed"] else "falling"
                            c_peak      = idx_compare["index_smoothed"].max()
                            c_flagged   = int((idx_compare["index_smoothed"] >= escalation_threshold).sum())
                            c_recent    = idx_compare.tail(6)
                            c_summary   = "\n".join(
                                f"  {row['event_month'].strftime('%b %Y')}: index={row['index_smoothed']:.1f}, "
                                f"battles={int(row.get('battles',0))}, explosions={int(row.get('explosions_remote_violence',0))}, "
                                f"fatalities={int(row.get('fatalities',0))}"
                                for _, row in c_recent.iterrows()
                            )
                            prompt = (
                                f"Compare the conflict escalation situations in {selected_country} and {compare_country}.\n\n"
                                f"{selected_country}: index {trend_dir}, current={latest['index_smoothed']:.1f}, "
                                f"peak={peak_val:.1f} ({peak_month}), {num_flagged} flagged months.\n"
                                f"Last 6 months:\n{recent_summary}\n\n"
                                f"{compare_country}: index {c_trend}, current={c_latest['index_smoothed']:.1f}, "
                                f"peak={c_peak:.1f}, {c_flagged} flagged months.\n"
                                f"Last 6 months:\n{c_summary}\n\n"
                                f"In 3 sentences: (1) which country has higher current escalation and by how much, "
                                f"(2) what's driving each country's index — are the causes similar or different, "
                                f"(3) which poses the greater near-term risk and why."
                            )
                            result = _call_claude(prompt, system=_ai_system, max_tokens=400)
                            st.markdown(
                                f"<div style='background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;"
                                f"padding:16px 20px;color:#e2e8f0;font-size:14px;line-height:1.7;'>{result}</div>",
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
elif not run_btn and st.session_state.get("page") != "map":
    st.info(
        "Enter a country name in the sidebar (e.g. **Ukraine**, **Sudan**, **Myanmar**) "
        "and click **Generate plot**."
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


if show_map and st.session_state.get("page") != "index":
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

                        focused = None  # selection handled in-map via JS

                        map_center, map_zoom = {"lat": 20, "lon": 10}, 1.0
                        map_col  = st.container()
                        panel_col = None

                        with map_col:
                            map_mode = st.radio(
                                "Map view",
                                ["🗺️  2D Map (Reccomended)", "🌐  3D Globe"],
                                horizontal=True,
                                label_visibility="collapsed",
                                key="map_mode_toggle",
                            )
                            use_3d = map_mode == "🌐  3D Globe"

                            # ── CesiumJS 3D Globe ─────────────────────────
                            try:
                                cesium_token = st.secrets["cesium"]["token"]
                            except Exception:
                                cesium_token = ""

                            # Build point data for Cesium
                            color_map = {
                                "Battles":                        "#ef4444",
                                "Explosions / remote violence":   "#f59e0b",
                                "Violence against civilians":     "#fde047",
                                "Strategic developments":         "#60a5fa",
                                "Protests":                       "#a78bfa",
                                "Riots":                          "#f472b6",
                            }
                            max_val = float(grouped["metric_value"].max()) or 1.0
                            cesium_points = []
                            for _, row in grouped.iterrows():
                                _lat = float(row["centroid_latitude"])
                                _lon = float(row["centroid_longitude"])
                                if abs(_lat) < 0.5 and abs(_lon) < 0.5:
                                    continue
                                cesium_points.append({
                                    "lat":      _lat,
                                    "lon":      _lon,
                                    "color":    color_map.get(row["dominant_category"], "#ffffff"),
                                    "size":     6 + 22 * float(row["metric_value"]) / max_val,
                                    "label":    f"{row['admin1']}, {row['country']}",
                                    "category": row["dominant_category"],
                                    "metric":   int(row["metric_value"]),
                                    "fatalities": int(row.get("fatalities", 0)),
                                    "metric_name": metric_labels[selected_metric],
                                    "country":    str(row["country"]),
                                })

                            cam_lat, cam_lon = 20, 10

                            # Hardcoded country bounding boxes (lat_min, lat_max, lon_min, lon_max, center_lat, center_lon)
                            # Used instead of computing from ACLED centroids, which can have bad coordinates.
                            _COUNTRY_BBOX = {
                                "Afghanistan": (29.4, 38.5, 60.5, 74.9, 33.9, 67.7),
                                "Algeria": (18.9, 37.1, -8.7, 12.0, 28.0, 2.6),
                                "Angola": (-18.0, -4.4, 11.7, 24.1, -11.2, 17.9),
                                "Armenia": (38.8, 41.3, 43.4, 46.6, 40.1, 45.0),
                                "Azerbaijan": (38.3, 41.9, 44.8, 50.4, 40.1, 47.6),
                                "Bangladesh": (20.7, 26.6, 88.0, 92.7, 23.7, 90.4),
                                "Belarus": (51.3, 56.2, 23.2, 32.8, 53.7, 28.0),
                                "Benin": (6.2, 12.4, 0.8, 3.8, 9.3, 2.3),
                                "Bolivia": (-22.9, -9.7, -69.6, -57.5, -16.3, -63.6),
                                "Bosnia and Herzegovina": (42.6, 45.3, 15.7, 19.6, 44.2, 17.7),
                                "Burkina Faso": (9.4, 15.1, -5.5, 2.4, 12.4, -1.6),
                                "Burundi": (-4.5, -2.3, 29.0, 30.9, -3.4, 29.9),
                                "Cambodia": (10.4, 14.7, 102.3, 107.6, 12.6, 104.9),
                                "Cameroon": (1.7, 13.1, 8.4, 16.2, 5.7, 12.4),
                                "Central African Republic": (2.2, 11.0, 14.4, 27.5, 6.6, 20.9),
                                "Chad": (7.4, 23.5, 13.5, 24.0, 15.5, 18.7),
                                "Colombia": (-4.2, 13.4, -79.0, -66.9, 4.1, -72.9),
                                "Congo": (-5.0, 3.7, 11.2, 18.6, -0.7, 15.2),
                                "Democratic Republic of Congo": (-13.5, 5.4, 12.2, 31.3, -4.0, 23.7),
                                "DRC": (-13.5, 5.4, 12.2, 31.3, -4.0, 23.7),
                                "DR Congo": (-13.5, 5.4, 12.2, 31.3, -4.0, 23.7),
                                "Ecuador": (-5.0, 1.4, -80.9, -75.2, -1.8, -78.2),
                                "Egypt": (22.0, 31.7, 24.7, 36.9, 26.8, 30.8),
                                "El Salvador": (13.1, 14.5, -90.1, -87.7, 13.8, -88.9),
                                "Eritrea": (12.4, 18.0, 36.4, 43.1, 15.2, 39.8),
                                "Ethiopia": (3.4, 15.0, 33.0, 48.0, 9.1, 40.5),
                                "Georgia": (41.1, 43.6, 40.0, 46.7, 42.3, 43.4),
                                "Guatemala": (13.7, 17.8, -92.2, -88.2, 15.8, -90.2),
                                "Guinea": (7.2, 12.7, -15.1, -7.6, 10.9, -11.4),
                                "Guinea-Bissau": (11.0, 12.7, -16.7, -13.6, 11.9, -15.2),
                                "Haiti": (18.0, 20.1, -74.5, -71.6, 18.9, -72.3),
                                "Honduras": (13.0, 16.5, -89.4, -83.1, 14.8, -86.2),
                                "India": (8.1, 35.5, 68.1, 97.4, 22.5, 80.0),
                                "Indonesia": (-11.0, 5.9, 95.0, 141.0, -2.5, 118.0),
                                "Iran": (25.1, 39.8, 44.0, 63.3, 32.4, 53.7),
                                "Iraq": (29.1, 37.4, 38.8, 48.6, 33.2, 43.7),
                                "Israel": (29.5, 33.3, 34.3, 35.9, 31.5, 35.0),
                                "Jordan": (29.2, 33.4, 34.9, 39.3, 31.2, 36.5),
                                "Kazakhstan": (40.6, 55.4, 50.3, 87.4, 48.0, 67.5),
                                "Kenya": (-4.7, 5.0, 33.9, 41.9, 0.2, 37.9),
                                "Kosovo": (41.9, 43.3, 20.0, 21.8, 42.6, 20.9),
                                "Kyrgyzstan": (39.2, 43.2, 69.3, 80.3, 41.2, 74.8),
                                "Lebanon": (33.1, 34.7, 35.1, 36.6, 33.9, 35.9),
                                "Liberia": (4.4, 8.6, -11.5, -7.4, 6.5, -9.4),
                                "Libya": (19.5, 33.2, 9.3, 25.2, 26.3, 17.2),
                                "Madagascar": (-25.6, -12.0, 43.2, 50.5, -18.8, 46.9),
                                "Malawi": (-17.1, -9.4, 32.7, 35.9, -13.3, 34.3),
                                "Mali": (10.1, 25.0, -4.2, 4.2, 17.6, -2.0),
                                "Mauritania": (14.7, 27.3, -17.1, -4.8, 20.3, -10.9),
                                "Mexico": (14.5, 32.7, -117.1, -86.7, 23.6, -102.6),
                                "Morocco": (27.7, 35.9, -13.2, -1.0, 31.8, -7.1),
                                "Mozambique": (-26.9, -10.5, 30.2, 40.8, -18.7, 35.5),
                                "Myanmar": (9.8, 28.5, 92.2, 101.2, 19.2, 96.7),
                                "Namibia": (-29.0, -17.0, 11.7, 25.3, -22.0, 18.5),
                                "Nicaragua": (10.7, 15.0, -87.7, -83.1, 12.9, -85.2),
                                "Niger": (11.7, 23.5, 0.2, 15.9, 17.6, 8.1),
                                "Nigeria": (4.3, 13.9, 2.7, 14.7, 9.1, 8.7),
                                "North Korea": (37.7, 43.0, 124.2, 130.7, 40.3, 127.5),
                                "Pakistan": (23.7, 37.1, 60.9, 77.8, 30.4, 69.3),
                                "Palestine": (31.2, 33.3, 34.2, 35.6, 31.9, 35.2),
                                "Peru": (-18.4, -0.0, -81.3, -68.7, -9.2, -75.0),
                                "Philippines": (5.0, 20.6, 117.2, 126.5, 12.9, 121.8),
                                "Rwanda": (-2.9, -1.1, 28.9, 30.9, -2.0, 29.9),
                                "Senegal": (12.3, 16.7, -17.5, -11.4, 14.5, -14.5),
                                "Sierra Leone": (6.9, 10.0, -13.3, -10.3, 8.5, -11.8),
                                "Somalia": (-1.7, 11.9, 40.9, 51.4, 5.2, 46.2),
                                "South Sudan": (3.5, 12.2, 24.1, 35.9, 7.9, 30.2),
                                "Sri Lanka": (5.9, 9.8, 79.7, 81.9, 7.9, 80.8),
                                "Sudan": (8.7, 22.2, 21.8, 38.6, 15.6, 30.2),
                                "Syria": (32.3, 37.3, 35.7, 42.4, 34.8, 39.1),
                                "Tajikistan": (36.7, 41.0, 67.4, 75.2, 38.9, 71.3),
                                "Tanzania": (-11.7, -0.9, 29.3, 40.4, -6.4, 34.9),
                                "Togo": (6.1, 11.1, -0.1, 1.8, 8.6, 0.8),
                                "Tunisia": (30.2, 37.5, 7.5, 11.6, 33.9, 9.6),
                                "Turkey": (35.8, 42.1, 26.0, 44.8, 39.0, 35.4),
                                "Turkmenistan": (35.1, 42.8, 52.4, 66.7, 39.0, 59.6),
                                "Uganda": (-1.5, 4.2, 29.6, 35.0, 1.4, 32.3),
                                "Ukraine": (44.4, 52.4, 22.1, 40.2, 48.4, 31.2),
                                "Uzbekistan": (37.2, 45.6, 55.9, 73.1, 41.4, 63.9),
                                "Venezuela": (0.6, 12.2, -73.4, -59.8, 7.1, -66.6),
                                "Vietnam": (8.6, 23.4, 102.1, 109.5, 16.0, 107.8),
                                "Yemen": (12.1, 19.0, 42.5, 54.5, 15.6, 48.5),
                                "Zambia": (-18.1, -8.2, 22.0, 33.7, -13.1, 27.8),
                                "Zimbabwe": (-22.4, -15.6, 25.2, 33.1, -19.0, 29.4),
                            }

                            # Per-country aggregates for in-map info panels
                            _country_data = {}
                            for _c, _cg in grouped.groupby("country"):
                                _lats = _cg["centroid_latitude"]
                                _lons = _cg["centroid_longitude"]
                                # Prefer hardcoded bbox — ACLED centroid coords can be corrupted
                                if _c in _COUNTRY_BBOX:
                                    _bb = _COUNTRY_BBOX[_c]
                                    _clat, _clon = _bb[4], _bb[5]
                                    _minlat, _maxlat, _minlon, _maxlon = _bb[0], _bb[1], _bb[2], _bb[3]
                                else:
                                    # Fallback: filter obvious outliers then use median
                                    _clat = float(_lats.median())
                                    _clon = float(_lons.median())
                                    _ok = ((_lats - _clat).abs() <= 15) & ((_lons - _clon).abs() <= 20)
                                    _lats_c = _lats[_ok] if _ok.any() else _lats
                                    _lons_c = _lons[_ok] if _ok.any() else _lons
                                    _minlat = float(_lats_c.min())
                                    _maxlat = float(_lats_c.max())
                                    _minlon = float(_lons_c.min())
                                    _maxlon = float(_lons_c.max())
                                _span = float(max(_maxlat - _minlat, _maxlon - _minlon, 0.5))
                                _zoom = float(max(1.5, min(6.0, 5.8 - np.log2(_span + 1))))
                                _country_data[_c] = {
                                    "lat": _clat, "lon": _clon, "zoom": _zoom,
                                    "minlat": _minlat, "maxlat": _maxlat,
                                    "minlon": _minlon, "maxlon": _maxlon,
                                    "fatalities":   int(_cg["fatalities"].sum()),
                                    "battles":      int(_cg["battles"].sum()),
                                    "explosions":   int(_cg["explosions_remote_violence"].sum()),
                                    "civ_violence": int(_cg["violence_against_civilians"].sum()),
                                    "strategic":    int(_cg["strategic_developments"].sum()),
                                    "protests":     int(_cg["protests"].sum()),
                                    "riots":        int(_cg["riots"].sum()),
                                    "actors":       int(_cg["violent_actors"].sum()),
                                    "metric_total": int(_cg["metric_value"].sum()),
                                    "metric_name":  metric_labels[selected_metric],
                                }

                            import json as _json
                            points_json = _json.dumps(cesium_points)
                            country_data_json = _json.dumps(_country_data)
                            map_h = 790
                            _s = start_dt.strftime('%b %Y')
                            _e = end_dt.strftime('%b %Y')
                            date_label = _s if _s == _e else f"{_s} \u2013 {_e}"

                            if use_3d:
                                globe_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  html,body{{width:100%;height:{map_h}px;background:#020617;overflow:hidden;font-family:'Inter',Arial,sans-serif;}}
  canvas{{display:block;}}
  #tooltip{{
    position:absolute;pointer-events:none;
    background:rgba(2,8,20,0.97);color:#e2e8f0;
    padding:12px 16px;border-radius:8px;font-size:13px;
    border:1px solid rgba(96,165,250,0.3);max-width:260px;display:none;
    line-height:1.8;box-shadow:0 0 24px rgba(96,165,250,0.15);z-index:99;
  }}
  #tooltip b{{color:#fff;font-size:14px;}}
  #tooltip .cat{{color:rgba(255,255,255,0.45);font-size:11px;letter-spacing:.05em;}}
  #legend{{
    position:absolute;bottom:16px;left:16px;
    background:rgba(2,8,20,0.88);color:#e2e8f0;
    padding:12px 16px;border-radius:8px;font-size:12px;
    border:1px solid rgba(96,165,250,0.2);
    box-shadow:0 0 20px rgba(0,0,0,0.5);
  }}
  #legend .ltitle{{font-weight:700;font-size:13px;margin-bottom:8px;color:#fff;letter-spacing:.06em;}}
  #legend .row{{display:flex;align-items:center;gap:9px;margin:4px 0;}}
  #legend .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  #title{{
    position:absolute;top:14px;left:50%;transform:translateX(-50%);
    color:#e2e8f0;font-size:16px;font-weight:600;letter-spacing:.08em;
    text-transform:uppercase;z-index:10;white-space:nowrap;
    text-shadow:0 0 20px rgba(96,165,250,0.6);
  }}
  #hint{{position:absolute;bottom:16px;right:16px;color:rgba(255,255,255,0.25);font-size:11px;letter-spacing:.04em;}}
  #rotatebtn{{
    position:absolute;top:14px;right:16px;z-index:20;
    background:rgba(2,8,20,0.80);border:1px solid rgba(96,165,250,0.35);
    color:#a0c4ff;font-size:12px;font-family:Inter,Arial,sans-serif;
    letter-spacing:.06em;padding:6px 13px;border-radius:6px;cursor:pointer;
    display:flex;align-items:center;gap:7px;user-select:none;
    transition:border-color .2s,color .2s;
  }}
  #rotatebtn:hover{{border-color:rgba(96,165,250,0.7);color:#fff;}}
  #rotatebtn .indicator{{
    width:8px;height:8px;border-radius:50%;background:#60a5fa;
    box-shadow:0 0 6px #60a5fa;transition:background .2s,box-shadow .2s;
  }}
  #rotatebtn.off .indicator{{background:#334;box-shadow:none;}}
  #rotatebtn.off{{color:rgba(255,255,255,0.35);}}
  #recenterbtn{{
    position:absolute;top:50px;right:16px;z-index:20;
    background:rgba(2,8,20,0.80);border:1px solid rgba(96,165,250,0.35);
    color:#a0c4ff;font-size:12px;font-family:Inter,Arial,sans-serif;
    letter-spacing:.06em;padding:6px 13px;border-radius:6px;cursor:pointer;
    display:flex;align-items:center;gap:7px;user-select:none;
    transition:border-color .2s,color .2s;
  }}
  #recenterbtn:hover{{border-color:rgba(96,165,250,0.7);color:#fff;}}
  #infopanel{{
    position:absolute;top:56px;right:16px;width:220px;
    background:linear-gradient(160deg,rgba(2,8,25,0.97),rgba(8,18,45,0.97));
    border:1px solid rgba(96,165,250,0.3);border-radius:10px;
    padding:14px;color:white;display:none;z-index:50;
    box-shadow:0 0 30px rgba(96,165,250,0.12);
    animation:slideIn .2s ease;font-family:Inter,Arial,sans-serif;
  }}
  @keyframes slideIn{{from{{opacity:0;transform:translateX(10px)}}to{{opacity:1;transform:translateX(0)}}}}
  #infopanel-close{{
    position:absolute;top:9px;right:11px;cursor:pointer;
    color:rgba(255,255,255,0.35);font-size:15px;line-height:1;transition:color .15s;
  }}
  #infopanel-close:hover{{color:white;}}
  #fullscreenbtn{{
    position:absolute;bottom:55px;right:16px;z-index:20;
    background:rgba(2,8,20,0.80);border:1px solid rgba(96,165,250,0.35);
    color:#a0c4ff;font-size:12px;font-family:Inter,Arial,sans-serif;
    letter-spacing:.06em;padding:6px 13px;border-radius:6px;cursor:pointer;
    display:flex;align-items:center;gap:7px;user-select:none;
    transition:border-color .2s,color .2s;
  }}
  #fullscreenbtn:hover{{border-color:rgba(96,165,250,0.7);color:#fff;}}
</style>
</head><body>
<div id="title">&#9632;&nbsp; {date_label} Conflict Hotspots</div>
<div id="rotatebtn" id="rotatebtn" onclick="toggleRotate()">
  <div class="indicator"></div><span id="rotatelabel">AUTO-ROTATE ON</span>
</div>
<div id="recenterbtn" onclick="recenter()">&#8859;&nbsp; RECENTER</div>
<div id="fullscreenbtn" onclick="toggleFullscreen()">&#x26F6;&nbsp; FULLSCREEN</div>
<div id="tooltip"></div>
<div id="legend">
  <div class="ltitle">CATEGORIES</div>
  <div class="row"><div class="dot" style="background:#ef4444"></div>Battles</div>
  <div class="row"><div class="dot" style="background:#f59e0b"></div>Explosions / Remote Violence</div>
  <div class="row"><div class="dot" style="background:#fde047"></div>Violence Against Civilians</div>
  <div class="row"><div class="dot" style="background:#60a5fa"></div>Strategic Developments</div>
  <div class="row"><div class="dot" style="background:#a78bfa"></div>Protests</div>
  <div class="row"><div class="dot" style="background:#f472b6"></div>Riots</div>
</div>
<div id="hint">DRAG TO ROTATE &nbsp;·&nbsp; SCROLL TO ZOOM &nbsp;·&nbsp; CLICK FOR DETAILS</div>
<div id="infopanel">
  <div id="infopanel-close" onclick="closePanel()">&#10005;</div>
  <div id="infopanel-content"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
const W = window.innerWidth, H = {map_h};
const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:false}});
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x020617);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, W/H, 0.1, 1000);
camera.position.z = 3.2;

// Stars
const sv = [];
for(let i=0;i<16000;i++){{
  const r=400, t=2*Math.PI*Math.random(), p=Math.acos(2*Math.random()-1);
  sv.push(r*Math.sin(p)*Math.cos(t), r*Math.sin(p)*Math.sin(t), r*Math.cos(p));
}}
const sg = new THREE.BufferGeometry();
sg.setAttribute('position', new THREE.Float32BufferAttribute(sv, 3));
scene.add(new THREE.Points(sg, new THREE.PointsMaterial({{color:0xffffff, size:0.5}})));

// Globe group
const globe = new THREE.Group();
scene.add(globe);

// Earth sphere
const earthMat = new THREE.MeshPhongMaterial({{
  color: 0x0a1628, specular: 0x1a3a6a, shininess: 40,
}});
const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), earthMat);
globe.add(earth);

// Glow
globe.add(new THREE.Mesh(
  new THREE.SphereGeometry(1.06, 64, 64),
  new THREE.MeshPhongMaterial({{color:0x1a4aaa, transparent:true, opacity:0.08, side:THREE.BackSide}})
));

// Lighting
scene.add(new THREE.AmbientLight(0x334466, 1.5));
const rim = new THREE.DirectionalLight(0x4488ff, 0.4);
rim.position.set(-3, 2, -3); scene.add(rim);
const key = new THREE.DirectionalLight(0x8abaff, 0.25);
key.position.set(4, 2, 4); scene.add(key);

// lat/lon → 3D
function ll(lat, lon, r){{
  const phi = (90-lat)*Math.PI/180, th = (lon+180)*Math.PI/180;
  return new THREE.Vector3(
    -r*Math.sin(phi)*Math.cos(th),
     r*Math.cos(phi),
     r*Math.sin(phi)*Math.sin(th)
  );
}}

// Lat/lon grid
(function(){{
  const mat = new THREE.LineBasicMaterial({{color:0x1a3060, transparent:true, opacity:0.3}});
  for(let lat=-80;lat<=80;lat+=20){{
    const pts=[];
    for(let lon=-180;lon<=180;lon+=2) pts.push(ll(lat,lon,1.001));
    globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }}
  for(let lon=-180;lon<180;lon+=20){{
    const pts=[];
    for(let lat=-90;lat<=90;lat+=2) pts.push(ll(lat,lon,1.001));
    globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }}
}})();

// Country + land borders
const borderMat = new THREE.LineBasicMaterial({{color:0x3a6ab0, transparent:true, opacity:0.75}});
fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
  .then(r=>r.json()).then(world=>{{
    function addLine(coords, r){{
      if(!coords||coords.length<2) return;
      globe.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(coords.map(c=>ll(c[1],c[0],r))),
        borderMat
      ));
    }}
    function procGeom(g, r){{
      if(!g) return;
      if(g.type==='LineString') addLine(g.coordinates,r);
      else if(g.type==='MultiLineString') g.coordinates.forEach(c=>addLine(c,r));
      else if(g.type==='Polygon') g.coordinates.forEach(c=>addLine(c,r));
      else if(g.type==='MultiPolygon') g.coordinates.forEach(p=>p.forEach(c=>addLine(c,r)));
    }}
    procGeom(topojson.mesh(world, world.objects.countries, (a,b)=>a!==b), 1.002);
    const land = topojson.feature(world, world.objects.land);
    if(land.type==='Feature') procGeom(land.geometry,1.003);
    else if(land.type==='FeatureCollection') land.features.forEach(f=>procGeom(f.geometry,1.003));
  }}).catch(()=>{{}});

// ── Geographic labels ─────────────────────────────────────────────
// Uses PlaneGeometry oriented to face outward from globe center
// so labels appear painted onto the surface
function makeLabel(text, color, fontSize, canvasW){{
  canvasW = canvasW || 512;
  const canvasH = Math.round(canvasW * 0.25);
  const c = document.createElement('canvas');
  c.width = canvasW; c.height = canvasH;
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,canvasW,canvasH);
  ctx.font = '700 '+fontSize+'px Inter,Arial,sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // dark halo for readability
  ctx.shadowColor = 'rgba(0,0,0,0.95)';
  ctx.shadowBlur = 10;
  ctx.strokeStyle = 'rgba(0,0,10,0.9)';
  ctx.lineWidth = Math.max(3, fontSize*0.18);
  ctx.lineJoin = 'round';
  ctx.strokeText(text, canvasW/2, canvasH/2);
  ctx.shadowBlur = 6;
  ctx.fillStyle = color;
  ctx.fillText(text, canvasW/2, canvasH/2);
  return new THREE.CanvasTexture(c);
}}

function addLabel(text, lat, lon, worldWidth, color, fontSize){{
  const tex = makeLabel(text, color, fontSize);
  const mat = new THREE.MeshBasicMaterial({{
    map: tex,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
  }});
  const aspect = 0.25; // canvasH/canvasW
  const geo = new THREE.PlaneGeometry(worldWidth, worldWidth * aspect);
  const mesh = new THREE.Mesh(geo, mat);
  // Position on surface
  const pos = ll(lat, lon, 1.001);
  mesh.position.copy(pos);

  // Orient: face outward (normal = radial direction)
  mesh.lookAt(new THREE.Vector3(0,0,0));
  mesh.rotateY(Math.PI); // flip so text faces outward

  globe.add(mesh);
}}

// Continents — gold, wide
const continents = [
  ['AFRICA',         5,   22,  0.52, 'rgba(255,210,80,0.90)',  56],
  ['EUROPE',        54,   15,  0.52, 'rgba(255,210,80,0.90)',  52],
  ['ASIA',          47,   90,  0.52, 'rgba(255,210,80,0.90)',  56],
  ['NORTH AMERICA', 50, -100,  0.58, 'rgba(255,210,80,0.90)',  52],
  ['SOUTH AMERICA',-15,  -58,  0.52, 'rgba(255,210,80,0.90)',  52],
  ['OCEANIA',      -25,  140,  0.45, 'rgba(255,210,80,0.90)',  46],
  ['ANTARCTICA',   -82,    0,  0.45, 'rgba(255,210,80,0.80)',  44],
];
continents.forEach(l=>addLabel(l[0],l[1],l[2],l[3],l[4],l[5]));

// Countries with zoom-based visibility
// Format: [name, lat, lon, worldWidth, color, fontSize, maxCamZ]
// maxCamZ: label shows when camera.position.z <= this (5=always, 2=close only)
const countryDefs = [
  // Giant nations — always visible
  ['RUSSIA',           61,  100,  0.44,'rgba(255,255,255,0.82)',42, 5.0],
  ['CANADA',           60,  -96,  0.40,'rgba(255,255,255,0.82)',40, 5.0],
  ['UNITED STATES',    38,  -97,  0.48,'rgba(255,255,255,0.82)',42, 5.0],
  ['BRAZIL',           -9,  -53,  0.38,'rgba(255,255,255,0.82)',40, 5.0],
  ['AUSTRALIA',       -24,  134,  0.38,'rgba(255,255,255,0.82)',40, 5.0],
  ['CHINA',            35,  103,  0.36,'rgba(255,255,255,0.82)',40, 5.0],
  ['INDIA',            22,   80,  0.32,'rgba(255,255,255,0.80)',38, 5.0],
  // Large nations
  ['ALGERIA',          28,    2,  0.28,'rgba(255,255,255,0.76)',32, 3.8],
  ['MEXICO',           24, -102,  0.26,'rgba(255,255,255,0.76)',32, 3.8],
  ['KAZAKHSTAN',       49,   68,  0.30,'rgba(255,255,255,0.76)',32, 3.8],
  ['SUDAN',            16,   30,  0.24,'rgba(255,255,255,0.76)',32, 3.8],
  ['UKRAINE',          49,   31,  0.26,'rgba(255,255,255,0.76)',32, 3.8],
  ['NIGERIA',           9,    8,  0.24,'rgba(255,255,255,0.76)',32, 3.8],
  ['ETHIOPIA',          9,   40,  0.26,'rgba(255,255,255,0.76)',32, 3.8],
  ['MYANMAR',          20,   96,  0.24,'rgba(255,255,255,0.76)',30, 3.8],
  ['AFGHANISTAN',      33,   66,  0.28,'rgba(255,255,255,0.76)',30, 3.8],
  ['MALI',             18,   -2,  0.22,'rgba(255,255,255,0.76)',30, 3.8],
  ['ANGOLA',          -12,   18,  0.26,'rgba(255,255,255,0.76)',30, 3.8],
  ['IRAN',             32,   54,  0.22,'rgba(255,255,255,0.76)',30, 3.8],
  ['CHAD',             15,   18,  0.20,'rgba(255,255,255,0.76)',30, 3.8],
  ['NIGER',            17,    8,  0.20,'rgba(255,255,255,0.76)',30, 3.8],
  ['PERU',            -10,  -76,  0.26,'rgba(255,255,255,0.76)',30, 3.8],
  ['COLOMBIA',          4,  -74,  0.24,'rgba(255,255,255,0.76)',30, 3.8],
  ['BOLIVIA',         -17,  -65,  0.24,'rgba(255,255,255,0.76)',30, 3.8],
  ['ARGENTINA',       -35,  -65,  0.26,'rgba(255,255,255,0.76)',30, 3.8],
  ['SOUTH AFRICA',    -29,   25,  0.28,'rgba(255,255,255,0.76)',30, 3.8],
  ['EGYPT',            27,   30,  0.22,'rgba(255,255,255,0.76)',30, 3.8],
  ['TURKEY',           39,   35,  0.22,'rgba(255,255,255,0.76)',30, 3.8],
  ['PAKISTAN',         30,   70,  0.24,'rgba(255,255,255,0.76)',30, 3.8],
  ['INDONESIA',        -5,  118,  0.26,'rgba(255,255,255,0.76)',30, 3.8],
  ['SAUDI ARABIA',     24,   45,  0.28,'rgba(255,255,255,0.76)',30, 3.8],
  ['LIBYA',            26,   17,  0.20,'rgba(255,255,255,0.76)',30, 3.8],
  ['D.R. CONGO',       -4,   24,  0.26,'rgba(255,255,255,0.76)',30, 3.8],
  ['MOZAMBIQUE',      -17,   35,  0.26,'rgba(255,255,255,0.76)',28, 3.8],
  ['SOMALIA',           6,   46,  0.22,'rgba(255,255,255,0.76)',28, 3.8],
  ['VENEZUELA',         7,  -66,  0.24,'rgba(255,255,255,0.76)',28, 3.8],
  ['NAMIBIA',         -22,   18,  0.22,'rgba(255,255,255,0.76)',28, 3.8],
  ['MAURITANIA',       20,  -11,  0.22,'rgba(255,255,255,0.76)',28, 3.8],
  ['ZAMBIA',          -14,   27,  0.22,'rgba(255,255,255,0.76)',28, 3.8],
  ['SOUTH SUDAN',       7,   31,  0.22,'rgba(255,255,255,0.76)',28, 3.8],
  ['CENT. AFR. REP.',   7,   21,  0.26,'rgba(255,255,255,0.76)',26, 3.8],
  ['MONGOLIA',         47,  104,  0.28,'rgba(255,255,255,0.76)',28, 3.8],
  ['MOROCCO',          32,   -6,  0.18,'rgba(255,255,255,0.76)',28, 3.8],
  ['ITALY',            43,   12,  0.16,'rgba(255,255,255,0.76)',26, 3.8],
  ['SPAIN',            40,   -4,  0.18,'rgba(255,255,255,0.76)',26, 3.8],
  ['FRANCE',           47,    2,  0.18,'rgba(255,255,255,0.76)',26, 3.8],
  ['GERMANY',          51,   10,  0.16,'rgba(255,255,255,0.76)',26, 3.8],
  ['POLAND',           52,   20,  0.18,'rgba(255,255,255,0.76)',26, 3.8],
  ['SWEDEN',           62,   16,  0.16,'rgba(255,255,255,0.76)',24, 3.8],
  ['NORWAY',           65,   14,  0.16,'rgba(255,255,255,0.76)',24, 3.8],
  ['FINLAND',          63,   26,  0.16,'rgba(255,255,255,0.76)',24, 3.8],
  ['CHILE',           -35,  -71,  0.20,'rgba(255,255,255,0.76)',26, 3.8],
  // Medium nations
  ['IRAQ',             33,   44,  0.20,'rgba(255,255,255,0.74)',26, 2.8],
  ['SYRIA',            35,   38,  0.20,'rgba(255,255,255,0.74)',26, 2.8],
  ['YEMEN',            15,   48,  0.20,'rgba(255,255,255,0.74)',26, 2.8],
  ['BURKINA FASO',     12,   -2,  0.22,'rgba(255,255,255,0.74)',24, 2.8],
  ['CAMEROON',          6,   12,  0.22,'rgba(255,255,255,0.74)',24, 2.8],
  ['KENYA',            -1,   38,  0.18,'rgba(255,255,255,0.74)',24, 2.8],
  ['TANZANIA',         -6,   35,  0.20,'rgba(255,255,255,0.74)',24, 2.8],
  ['ZIMBABWE',        -19,   29,  0.18,'rgba(255,255,255,0.74)',24, 2.8],
  ['GHANA',             8,   -2,  0.16,'rgba(255,255,255,0.74)',24, 2.8],
  ["COTE D'IVOIRE",    7,   -6,  0.20,'rgba(255,255,255,0.74)',22, 2.8],
  ['SENEGAL',          14,  -14,  0.16,'rgba(255,255,255,0.74)',22, 2.8],
  ['GUINEA',           11,  -12,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['MADAGASCAR',      -20,   47,  0.18,'rgba(255,255,255,0.74)',22, 2.8],
  ['MALAYSIA',          2,  112,  0.20,'rgba(255,255,255,0.74)',22, 2.8],
  ['THAILAND',         15,  101,  0.18,'rgba(255,255,255,0.74)',22, 2.8],
  ['VIETNAM',          16,  107,  0.18,'rgba(255,255,255,0.74)',22, 2.8],
  ['PHILIPPINES',      12,  122,  0.18,'rgba(255,255,255,0.74)',22, 2.8],
  ['JAPAN',            36,  138,  0.20,'rgba(255,255,255,0.74)',24, 2.8],
  ['SOUTH KOREA',      36,  128,  0.16,'rgba(255,255,255,0.74)',22, 2.8],
  ['NORTH KOREA',      40,  127,  0.16,'rgba(255,255,255,0.74)',22, 2.8],
  ['UZBEKISTAN',       41,   64,  0.18,'rgba(255,255,255,0.74)',22, 2.8],
  ['ECUADOR',          -2,  -77,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['PARAGUAY',        -23,  -58,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['URUGUAY',         -33,  -56,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['UGANDA',            2,   32,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['MALAWI',          -13,   34,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['BENIN',             9,    2,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['BANGLADESH',       24,   90,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['JORDAN',           31,   36,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['UAE',              24,   54,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['OMAN',             22,   57,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['ROMANIA',          46,   25,  0.14,'rgba(255,255,255,0.74)',22, 2.8],
  ['UK',               54,   -2,  0.12,'rgba(255,255,255,0.74)',22, 2.8],
  ['IRELAND',          53,   -8,  0.10,'rgba(255,255,255,0.74)',20, 2.8],
  ['GREECE',           39,   22,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['PORTUGAL',         39,   -8,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['TUNISIA',          34,    9,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['HONDURAS',         15,  -86,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['NICARAGUA',        13,  -85,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['GUATEMALA',        15,  -90,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['CUBA',             22,  -79,  0.14,'rgba(255,255,255,0.74)',20, 2.8],
  ['HAITI',            19,  -73,  0.10,'rgba(255,255,255,0.74)',20, 2.8],
  ['PAPUA NEW GUINEA', -6,  145,  0.18,'rgba(255,255,255,0.74)',20, 2.8],
  ['NEW ZEALAND',     -42,  173,  0.14,'rgba(255,255,255,0.74)',20, 2.8],
  ['TAJIKISTAN',       39,   71,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['KYRGYZSTAN',       42,   75,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['TURKMENISTAN',     40,   59,  0.16,'rgba(255,255,255,0.74)',20, 2.8],
  ['BELARUS',          53,   28,  0.14,'rgba(255,255,255,0.74)',20, 2.8],
  ['AZERBAIJAN',       40,   48,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['GEORGIA',          42,   43,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['ARMENIA',          40,   45,  0.10,'rgba(255,255,255,0.74)',20, 2.8],
  ['CAMBODIA',         12,  105,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['LAOS',             18,  103,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['ERITREA',          15,   39,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['GABON',            -1,   12,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['CONGO',            -1,   15,  0.10,'rgba(255,255,255,0.74)',20, 2.8],
  ['CZECHIA',          50,   16,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['HUNGARY',          47,   19,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['AUSTRIA',          47,   14,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['SWITZERLAND',      47,    8,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['SERBIA',           44,   21,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  ['CROATIA',          45,   16,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  ['SLOVAKIA',         48,   19,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  ['ICELAND',          65,  -18,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['GUYANA',            5,  -59,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['SURINAME',          4,  -56,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['COSTA RICA',        9,  -84,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['PANAMA',            9,  -80,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['SRI LANKA',         7,   81,  0.10,'rgba(255,255,255,0.74)',20, 2.8],
  ['NEPAL',            28,   84,  0.12,'rgba(255,255,255,0.74)',20, 2.8],
  ['TOGO',              8,    1,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  ['SIERRA LEONE',      8,  -12,  0.12,'rgba(255,255,255,0.74)',18, 2.8],
  ['LIBERIA',           6,   -9,  0.12,'rgba(255,255,255,0.74)',18, 2.8],
  ['BURUNDI',          -3,   30,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  ['RWANDA',           -2,   30,  0.10,'rgba(255,255,255,0.74)',18, 2.8],
  // Small nations — only appear when zoomed in
  ['ISRAEL',           31,   35,  0.12,'rgba(255,255,255,0.82)',22, 2.0],
  ['LEBANON',          34,   36,  0.10,'rgba(255,255,255,0.82)',20, 2.0],
  ['GAZA',             31,   34,  0.09,'rgba(255,255,255,0.92)',18, 1.8],
  ['TAIWAN',           24,  121,  0.10,'rgba(255,255,255,0.74)',20, 2.0],
  ['DJIBOUTI',         12,   43,  0.09,'rgba(255,255,255,0.74)',18, 1.9],
  ['QATAR',            25,   51,  0.08,'rgba(255,255,255,0.74)',16, 1.8],
  ['KUWAIT',           29,   48,  0.08,'rgba(255,255,255,0.74)',16, 1.8],
  ['BAHRAIN',          26,   50,  0.06,'rgba(255,255,255,0.74)',14, 1.7],
  ['MOLDOVA',          47,   29,  0.10,'rgba(255,255,255,0.74)',18, 1.9],
  ['BOSNIA',           44,   17,  0.10,'rgba(255,255,255,0.74)',18, 1.9],
  ['ALBANIA',          41,   20,  0.08,'rgba(255,255,255,0.74)',18, 1.9],
  ['N. MACEDONIA',     41,   22,  0.10,'rgba(255,255,255,0.74)',16, 1.9],
  ['MONTENEGRO',       43,   19,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['KOSOVO',           42,   21,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['SLOVENIA',         46,   15,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['ESTONIA',          59,   25,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['LATVIA',           57,   25,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['LITHUANIA',        56,   24,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['DENMARK',          56,   10,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['NETHERLANDS',      52,    5,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['BELGIUM',          51,    4,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['DOM. REPUBLIC',    19,  -70,  0.12,'rgba(255,255,255,0.74)',18, 1.9],
  ['JAMAICA',          18,  -77,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['ESWATINI',        -26,   32,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['LESOTHO',         -29,   28,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['BELIZE',           17,  -89,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['EL SALVADOR',      14,  -89,  0.09,'rgba(255,255,255,0.74)',16, 1.9],
  ['GUINEA-BISSAU',    12,  -15,  0.12,'rgba(255,255,255,0.74)',16, 1.9],
  ['TIMOR-LESTE',      -9,  126,  0.09,'rgba(255,255,255,0.74)',16, 1.9],
  ['BRUNEI',            4,  115,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['SINGAPORE',         1,  104,  0.06,'rgba(255,255,255,0.74)',14, 1.7],
  ['TRINIDAD',         11,  -61,  0.07,'rgba(255,255,255,0.74)',14, 1.8],
  ['LUXEMBOURG',       50,    6,  0.06,'rgba(255,255,255,0.74)',14, 1.7],
  ['MALTA',            36,   14,  0.05,'rgba(255,255,255,0.74)',12, 1.6],
  ['CYPRUS',           35,   33,  0.08,'rgba(255,255,255,0.74)',16, 1.9],
  ['ICELAND',          65,  -18,  0.12,'rgba(255,255,255,0.74)',20, 2.2],
  ['CABO VERDE',       16,  -24,  0.06,'rgba(255,255,255,0.74)',12, 1.7],
  ['COMOROS',         -12,   44,  0.06,'rgba(255,255,255,0.74)',12, 1.6],
  ['MALDIVES',          4,   73,  0.05,'rgba(255,255,255,0.74)',12, 1.6],
];

const countryLabelMeshes = [];
countryDefs.forEach(function(d) {{
  const tex = makeLabel(d[0], d[4], d[5]);
  const mat = new THREE.MeshBasicMaterial({{map:tex,transparent:true,depthWrite:false,depthTest:true,side:THREE.DoubleSide}});
  const w = d[3], h = w * 0.25;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  mesh.position.copy(ll(d[1], d[2], 1.001));
  mesh.lookAt(new THREE.Vector3(0,0,0));
  mesh.rotateY(Math.PI);
  mesh.userData.maxCamZ = d[6];
  mesh.visible = false; // set in animate loop
  globe.add(mesh);
  countryLabelMeshes.push(mesh);
}});

// Oceans — blue
const waters = [
  ['PACIFIC OCEAN',    5, -155, 0.55, 'rgba(120,185,255,0.65)', 50],
  ['ATLANTIC OCEAN',  10,  -30, 0.50, 'rgba(120,185,255,0.65)', 48],
  ['INDIAN OCEAN',   -22,   80, 0.48, 'rgba(120,185,255,0.65)', 46],
  ['ARCTIC OCEAN',    87,    0, 0.40, 'rgba(120,185,255,0.60)', 38],
  ['SOUTHERN OCEAN', -62,    0, 0.44, 'rgba(120,185,255,0.60)', 38],
  ['MEDITERRANEAN',   36,   18, 0.28, 'rgba(120,185,255,0.60)', 28],
  ['RED SEA',         20,   38, 0.20, 'rgba(120,185,255,0.60)', 26],
  ['PERSIAN GULF',    26,   52, 0.20, 'rgba(120,185,255,0.60)', 24],
  ['CARIBBEAN SEA',   16,  -75, 0.26, 'rgba(120,185,255,0.60)', 28],
  ['BLACK SEA',       43,   34, 0.20, 'rgba(120,185,255,0.60)', 24],
  ['CASPIAN SEA',     42,   51, 0.20, 'rgba(120,185,255,0.60)', 24],
  ['SOUTH CHINA SEA', 12,  114, 0.28, 'rgba(120,185,255,0.60)', 28],
];
waters.forEach(l=>addLabel(l[0],l[1],l[2],l[3],l[4],l[5]));

// ── ACLED conflict dots ───────────────────────────────────────────
const points = {points_json};
const dotMeshes = [], dotData = [], dotBaseSizes = [];
const BASE_CAM_Z = 2.6;
points.forEach(function(p){{
  const sz = 0.003 + 0.009 * (p.size / 28);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(sz, 8, 8),
    new THREE.MeshBasicMaterial({{color: new THREE.Color(p.color)}})
  );
  mesh.position.copy(ll(p.lat, p.lon, 1.014));
  globe.add(mesh);
  dotMeshes.push(mesh);
  dotData.push(p);
  dotBaseSizes.push(sz);
}});

// Orient globe
const id = ll({cam_lat}, {cam_lon}, 1);
globe.rotation.y = -Math.atan2(id.x, id.z);
const initRotY = globe.rotation.y;

// Country data for info panels
const countryData = {country_data_json};

// ── Info panel ────────────────────────────────────────────────
function infoRow(color, label, val){{
  return '<div style="display:flex;justify-content:space-between;align-items:center;'
    +'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
    +'<span style="color:rgba(255,255,255,0.55);font-size:11px;letter-spacing:.04em;">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;font-size:13px;">'+val.toLocaleString()+'</span>'
    +'</div>';
}}

function showInfoPanel(name){{
  const c = countryData[name];
  if(!c) return;
  document.getElementById('infopanel-content').innerHTML =
    '<div style="font-size:15px;font-weight:800;margin-bottom:1px;padding-right:18px;">'+name+'</div>'
    +'<div style="font-size:9px;color:rgba(255,255,255,0.3);letter-spacing:1.2px;margin-bottom:10px;">ACLED CONFLICT DATA</div>'
    +infoRow('#ef4444','FATALITIES',    c.fatalities)
    +infoRow('#f87171','BATTLES',       c.battles)
    +infoRow('#f59e0b','EXPLOSIONS',    c.explosions)
    +infoRow('#fde047','CIV. VIOLENCE', c.civ_violence)
    +infoRow('#60a5fa','STRATEGIC',     c.strategic)
    +infoRow('#a78bfa','PROTESTS',      c.protests)
    +infoRow('#f472b6','RIOTS',         c.riots)
    +'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0 0;">'
    +'<span style="color:rgba(255,255,255,0.45);font-size:10px;">'+c.metric_name.toUpperCase()+'</span>'
    +'<span style="color:white;font-weight:700;font-size:13px;">'+c.metric_total.toLocaleString()+'</span>'
    +'</div>';
  document.getElementById('infopanel').style.display='block';
}}

function closePanel(){{
  document.getElementById('infopanel').style.display='none';
  tz = 3.2;
  flyTarget = null;
}}
function toggleFullscreen(){{
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  if(!isFs && req) {{
    req.call(el);
  }} else if(isFs && exit) {{
    exit.call(document);
  }}
}}
function _updateFsBtn3d(){{
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  const btn = document.getElementById('fullscreenbtn');
  if(btn) btn.innerHTML = isFs ? '&#x26F6;&nbsp; EXIT FULL' : '&#x26F6;&nbsp; FULLSCREEN';
  if(isFs){{
    renderer.setSize(window.screen.width, window.screen.height);
    camera.aspect = window.screen.width/window.screen.height;
    camera.updateProjectionMatrix();
  }} else {{
    renderer.setSize(W, H);
    camera.aspect = W/H;
    camera.updateProjectionMatrix();
  }}
}}
document.addEventListener('fullscreenchange', _updateFsBtn3d);
document.addEventListener('webkitfullscreenchange', _updateFsBtn3d);
function recenter(){{
  flyTarget = {{ rotY: initRotY, rotX: 0, z: 3.2 }};
  closePanel();
}}

// ── Hardcoded country centroids for fly-to (bypasses bad ACLED coords) ──────
const GEO_CENTER = {{
  "Afghanistan":[33.9,67.7],"Algeria":[28.0,2.6],"Angola":[-11.2,17.9],
  "Armenia":[40.1,45.0],"Azerbaijan":[40.1,47.6],"Bangladesh":[23.7,90.4],
  "Belarus":[53.7,28.0],"Benin":[9.3,2.3],"Bolivia":[-16.3,-63.6],
  "Bosnia and Herzegovina":[44.2,17.7],"Burkina Faso":[12.4,-1.6],
  "Burundi":[-3.4,29.9],"Cambodia":[12.6,104.9],"Cameroon":[5.7,12.4],
  "Central African Republic":[6.6,20.9],"Chad":[15.5,18.7],
  "Colombia":[4.1,-72.9],"Congo":[-0.7,15.2],"Democratic Republic of Congo":[-4.0,23.7],
  "DR Congo":[-4.0,23.7],"DRC":[-4.0,23.7],"Ecuador":[-1.8,-78.2],
  "Egypt":[26.8,30.8],"El Salvador":[13.8,-88.9],"Eritrea":[15.2,39.8],
  "Ethiopia":[9.1,40.5],"Georgia":[42.3,43.4],"Guatemala":[15.8,-90.2],
  "Guinea":[10.9,-11.4],"Guinea-Bissau":[11.9,-15.2],"Haiti":[18.9,-72.3],
  "Honduras":[14.8,-86.2],"India":[22.5,80.0],"Indonesia":[-2.5,118.0],
  "Iran":[32.4,53.7],"Iraq":[33.2,43.7],"Israel":[31.5,35.0],
  "Jordan":[31.2,36.5],"Kazakhstan":[48.0,67.5],"Kenya":[0.2,37.9],
  "Kosovo":[42.6,20.9],"Kyrgyzstan":[41.2,74.8],"Lebanon":[33.9,35.9],
  "Liberia":[6.5,-9.4],"Libya":[26.3,17.2],"Madagascar":[-18.8,46.9],
  "Malawi":[-13.3,34.3],"Mali":[17.6,-2.0],"Mauritania":[20.3,-10.9],
  "Mexico":[23.6,-102.6],"Morocco":[31.8,-7.1],"Mozambique":[-18.7,35.5],
  "Myanmar":[19.2,96.7],"Namibia":[-22.0,18.5],"Nicaragua":[12.9,-85.2],
  "Niger":[17.6,8.1],"Nigeria":[9.1,8.7],"North Korea":[40.3,127.5],
  "Pakistan":[30.4,69.3],"Palestine":[31.9,35.2],"Peru":[-9.2,-75.0],
  "Philippines":[12.9,121.8],"Rwanda":[-2.0,29.9],"Senegal":[14.5,-14.5],
  "Sierra Leone":[8.5,-11.8],"Somalia":[5.2,46.2],"South Sudan":[7.9,30.2],
  "Sri Lanka":[7.9,80.8],"Sudan":[15.6,30.2],"Syria":[34.8,39.1],
  "Tajikistan":[38.9,71.3],"Tanzania":[-6.4,34.9],"Togo":[8.6,0.8],
  "Tunisia":[33.9,9.6],"Turkey":[39.0,35.4],"Turkmenistan":[39.0,59.6],
  "Uganda":[1.4,32.3],"Ukraine":[48.4,31.2],"Uzbekistan":[41.4,63.9],
  "Venezuela":[7.1,-66.6],"Vietnam":[16.0,107.8],"Yemen":[15.6,48.5],
  "Zambia":[-13.1,27.8],"Zimbabwe":[-19.0,29.4],
  "Russia":[61.0,100.0],"China":[35.0,103.0],"United States":[38.0,-97.0],
  "Brazil":[-9.0,-53.0],"Australia":[-24.0,134.0],"Canada":[60.0,-96.0],
  "Argentina":[-35.0,-65.0],"South Africa":[-29.0,25.0],
}};

// ── Fly-to ────────────────────────────────────────────────────
let flyTarget = null;

function selectCountry(name){{
  const c = countryData[name];
  if(!c){{ closePanel(); return; }}
  // Use hardcoded centroid if available — ACLED centroids can be corrupted
  const geo = GEO_CENTER[name];
  const flyLat = geo ? geo[0] : c.lat;
  const flyLon = geo ? geo[1] : c.lon;
  const pos = ll(flyLat, flyLon, 1);
  let rotY = -Math.atan2(pos.x, pos.z);
  const rotX = Math.asin(Math.max(-0.99,Math.min(0.99,pos.y)));
  let dy = rotY - globe.rotation.y;
  while(dy>Math.PI) dy-=2*Math.PI;
  while(dy<-Math.PI) dy+=2*Math.PI;
  flyTarget = {{
    rotY: globe.rotation.y+dy,
    rotX: rotX,
    z: Math.max(0.85, Math.min(1.4, 2.6-(c.zoom-1.0)*0.38)),
  }};
  showInfoPanel(name);
  autoRotate=false;
  document.getElementById('rotatebtn').classList.add('off');
  document.getElementById('rotatelabel').textContent='AUTO-ROTATE OFF';
}}

// Auto-rotate state
let autoRotate = true;
function toggleRotate(){{
  autoRotate = !autoRotate;
  const btn = document.getElementById('rotatebtn');
  const lbl = document.getElementById('rotatelabel');
  if(autoRotate){{
    btn.classList.remove('off');
    lbl.textContent = 'AUTO-ROTATE ON';
  }} else {{
    btn.classList.add('off');
    lbl.textContent = 'AUTO-ROTATE OFF';
  }}
}}

// Drag + scroll + inertia + click detection
let drag=false, px=0, py=0, vx=0, vy=0, tz=3.2;
let clickStartX=0, clickStartY=0;
const cvs = renderer.domElement;
const ray=new THREE.Raycaster(), mouse=new THREE.Vector2();

cvs.addEventListener('mousedown', e=>{{
  drag=true; px=e.clientX; py=e.clientY; vx=0; vy=0;
  clickStartX=e.clientX; clickStartY=e.clientY;
}});

window.addEventListener('mouseup', e=>{{
  drag=false;
  const dx=e.clientX-clickStartX, dy2=e.clientY-clickStartY;
  if(Math.sqrt(dx*dx+dy2*dy2)<6){{
    const r=cvs.getBoundingClientRect();
    mouse.x=((e.clientX-r.left)/r.width)*2-1;
    mouse.y=-((e.clientY-r.top)/r.height)*2+1;
    ray.setFromCamera(mouse,camera);
    // Click on a dot → use its country
    const dotHits=ray.intersectObjects(dotMeshes);
    if(dotHits.length){{
      selectCountry(dotData[dotMeshes.indexOf(dotHits[0].object)].country);
    }} else {{
      // Click on globe surface → only match if inside a country's bounding box
      const earthHits=ray.intersectObject(earth);
      if(earthHits.length){{
        const pt = earthHits[0].point.clone();
        // Transform world-space hit point into globe's local space
        // so lat/lon match actual geographic coordinates
        globe.worldToLocal(pt);
        pt.normalize();
        const lat=Math.asin(Math.max(-1,Math.min(1,pt.y)))*180/Math.PI;
        let lon=Math.atan2(pt.z,-pt.x)*180/Math.PI-180;
        if(lon < -180) lon += 360;
        // Find countries whose bounding box contains the click point
        const candidates=[];
        for(const [cn,cd] of Object.entries(countryData)){{
          const pad=0.5;
          if(lat>=cd.minlat-pad && lat<=cd.maxlat+pad &&
             lon>=cd.minlon-pad && lon<=cd.maxlon+pad){{
            // Use GEO_CENTER for reliable centroid distance
            const geo=GEO_CENTER[cn];
            const clat=geo?geo[0]:cd.lat, clon=geo?geo[1]:cd.lon;
            const dlat=clat-lat, dlon=(clon-lon)*Math.cos(lat*Math.PI/180);
            candidates.push({{name:cn, d:Math.sqrt(dlat*dlat+dlon*dlon)}});
          }}
        }}
        if(candidates.length>0){{
          candidates.sort((a,b)=>a.d-b.d);
          selectCountry(candidates[0].name);
        }} else {{
          closePanel();
        }}
      }} else {{
        closePanel();
      }}
    }}
  }}
}});

cvs.addEventListener('mousemove', e=>{{
  if(!drag) return;
  vy=(e.clientX-px)*0.004; vx=(e.clientY-py)*0.004;
  globe.rotation.y+=vy; globe.rotation.x+=vx;
  globe.rotation.x=Math.max(-1.4, Math.min(1.4, globe.rotation.x));
  px=e.clientX; py=e.clientY;
}});
cvs.addEventListener('wheel', e=>{{
  tz=Math.max(1.3, Math.min(5.0, tz+e.deltaY*0.003));
  flyTarget=null;
  e.preventDefault();
}}, {{passive:false}});

// Hover tooltip
const tip=document.getElementById('tooltip');
cvs.addEventListener('mousemove', e=>{{
  const r=cvs.getBoundingClientRect();
  mouse.x=((e.clientX-r.left)/r.width)*2-1;
  mouse.y=-((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(mouse,camera);
  const hits=ray.intersectObjects(dotMeshes);
  if(hits.length){{
    const p=dotData[dotMeshes.indexOf(hits[0].object)];
    tip.style.display='block';
    tip.style.left=(e.clientX+16)+'px';
    tip.style.top=(e.clientY-10)+'px';
    tip.innerHTML='<b>'+p.label+'</b><br>'
      +'<span class="cat">'+p.category.toUpperCase()+'</span><br><br>'
      +p.metric_name+': <b>'+p.metric.toLocaleString()+'</b><br>'
      +'Fatalities: <b>'+p.fatalities.toLocaleString()+'</b>';
    cvs.style.cursor='pointer';
  }} else {{
    tip.style.display='none';
    cvs.style.cursor='grab';
  }}
}});

function animate(){{
  requestAnimationFrame(animate);
  if(!drag){{
    globe.rotation.y += vy*0.90; vy*=0.90;
    if(autoRotate) globe.rotation.y += 0.0008;
    if(flyTarget){{
      const spd=0.055;
      globe.rotation.y+=(flyTarget.rotY-globe.rotation.y)*spd;
      globe.rotation.x+=(flyTarget.rotX-globe.rotation.x)*spd;
      tz+=(flyTarget.z-tz)*spd;
      if(Math.abs(flyTarget.rotY-globe.rotation.y)<0.0005&&Math.abs(flyTarget.z-tz)<0.0005)
        flyTarget=null;
    }}
  }}
  camera.position.z += (tz-camera.position.z)*0.08;
  const cz = camera.position.z;
  const zr = cz / BASE_CAM_Z;
  for(let i=0;i<dotMeshes.length;i++) dotMeshes[i].scale.set(zr,zr,zr);
  for(let i=0;i<countryLabelMeshes.length;i++){{
    countryLabelMeshes[i].visible = cz <= countryLabelMeshes[i].userData.maxCamZ;
  }}
  renderer.render(scene, camera);
}}
animate();
</script></body></html>"""
                                st.components.v1.html(globe_html, height=map_h, scrolling=False)

                            else:
                                # ── 2D Leaflet Map ────────────────────────────
                                leaflet_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  html,body{{width:100%;height:{map_h}px;background:#020617;overflow:hidden;font-family:'Inter',Arial,sans-serif;}}
  #mapid{{width:100%;height:100%;}}
  #title2d{{
    position:absolute;top:14px;left:50%;transform:translateX(-50%);
    color:#e2e8f0;font-size:16px;font-weight:600;letter-spacing:.08em;
    text-transform:uppercase;z-index:1000;white-space:nowrap;
    text-shadow:0 0 20px rgba(96,165,250,0.6);pointer-events:none;
  }}
  #legend2d{{
    position:absolute;bottom:30px;left:16px;
    background:rgba(2,8,20,0.88);color:#e2e8f0;
    padding:12px 16px;border-radius:8px;font-size:12px;
    border:1px solid rgba(96,165,250,0.2);z-index:1000;
    box-shadow:0 0 20px rgba(0,0,0,0.5);
  }}
  #legend2d .ltitle{{font-weight:700;font-size:13px;margin-bottom:8px;color:#fff;letter-spacing:.06em;}}
  #legend2d .row{{display:flex;align-items:center;gap:9px;margin:4px 0;}}
  #legend2d .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  #hint2d{{position:absolute;bottom:30px;right:16px;color:rgba(255,255,255,0.25);font-size:11px;letter-spacing:.04em;z-index:1000;}}
  #recenterbtn2d{{
    position:absolute;bottom:55px;right:16px;z-index:1000;
    background:rgba(2,8,20,0.85);border:1px solid rgba(96,165,250,0.35);
    color:#a0c4ff;font-size:12px;font-family:Inter,Arial,sans-serif;
    letter-spacing:.06em;padding:6px 13px;border-radius:6px;cursor:pointer;
    display:flex;align-items:center;gap:7px;user-select:none;
    transition:border-color .2s,color .2s;
  }}
  #recenterbtn2d:hover{{border-color:rgba(96,165,250,0.7);color:#fff;}}
  #fullscreenbtn2d{{
    position:absolute;bottom:90px;right:16px;z-index:1000;
    background:rgba(2,8,20,0.85);border:1px solid rgba(96,165,250,0.35);
    color:#a0c4ff;font-size:12px;font-family:Inter,Arial,sans-serif;
    letter-spacing:.06em;padding:6px 13px;border-radius:6px;cursor:pointer;
    display:flex;align-items:center;gap:7px;user-select:none;
    transition:border-color .2s,color .2s;
  }}
  #fullscreenbtn2d:hover{{border-color:rgba(96,165,250,0.7);color:#fff;}}
  #infopanel2d{{
    position:absolute;top:56px;right:16px;width:220px;
    background:linear-gradient(160deg,rgba(2,8,25,0.97),rgba(8,18,45,0.97));
    border:1px solid rgba(96,165,250,0.3);border-radius:10px;
    padding:14px;color:white;display:none;z-index:1000;
    box-shadow:0 0 30px rgba(96,165,250,0.12);
    animation:slideIn .2s ease;font-family:Inter,Arial,sans-serif;
  }}
  @keyframes slideIn{{from{{opacity:0;transform:translateX(10px)}}to{{opacity:1;transform:translateX(0)}}}}
  #infopanel2d-close{{
    position:absolute;top:9px;right:11px;cursor:pointer;
    color:rgba(255,255,255,0.35);font-size:15px;line-height:1;transition:color .15s;
  }}
  #infopanel2d-close:hover{{color:white;}}
  .leaflet-container{{background:#020617;}}
  .leaflet-tile-pane{{opacity:1;}}
  .dark-tip{{
    background:rgba(2,8,20,0.97)!important;border:1px solid rgba(96,165,250,0.3)!important;
    color:white!important;font-family:Inter,Arial,sans-serif!important;font-size:13px!important;
    border-radius:8px!important;padding:10px 13px!important;
    box-shadow:0 0 20px rgba(96,165,250,0.12)!important;
  }}
  .dark-tip.leaflet-tooltip-right::before{{border-right-color:rgba(96,165,250,0.3)!important;}}
  .dark-tip.leaflet-tooltip-left::before{{border-left-color:rgba(96,165,250,0.3)!important;}}
  .leaflet-control-zoom{{border:1px solid rgba(96,165,250,0.25)!important;}}
  .leaflet-control-zoom a{{background:rgba(2,8,20,0.85)!important;color:#a0c4ff!important;border-bottom:1px solid rgba(96,165,250,0.15)!important;}}
  .leaflet-control-zoom a:hover{{background:rgba(10,20,50,0.95)!important;color:white!important;}}
</style>
</head><body>
<div id="title2d">&#9632;&nbsp; {date_label} Conflict Hotspots</div>
<div id="mapid"></div>
<div id="legend2d">
  <div class="ltitle">CATEGORIES</div>
  <div class="row"><div class="dot" style="background:#ef4444"></div>Battles</div>
  <div class="row"><div class="dot" style="background:#f59e0b"></div>Explosions / Remote Violence</div>
  <div class="row"><div class="dot" style="background:#fde047"></div>Violence Against Civilians</div>
  <div class="row"><div class="dot" style="background:#60a5fa"></div>Strategic Developments</div>
  <div class="row"><div class="dot" style="background:#a78bfa"></div>Protests</div>
  <div class="row"><div class="dot" style="background:#f472b6"></div>Riots</div>
</div>
<div id="hint2d">CLICK FOR DETAILS &nbsp;·&nbsp; SCROLL TO ZOOM</div>
<div id="recenterbtn2d" onclick="recenter2d()">&#8859;&nbsp; RECENTER</div>
<div id="fullscreenbtn2d" onclick="toggleFullscreen2d()">&#x26F6;&nbsp; FULLSCREEN</div>
<div id="infopanel2d">
  <div id="infopanel2d-close" onclick="closePanel2d()">&#10005;</div>
  <div id="infopanel2d-content"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const points2d = {points_json};
const countryData2d = {country_data_json};

const map2d = L.map('mapid', {{
  center: [20, 10], zoom: 2,
  zoomControl: true, attributionControl: false,
  minZoom: 1, maxZoom: 12,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  subdomains: 'abcd', maxZoom: 19,
}}).addTo(map2d);

// Plot markers
points2d.forEach(function(p) {{
  if(Math.abs(p.lat)<0.5 && Math.abs(p.lon)<0.5) return;
  const radius = 0.5 + 10*(p.size/28);
  const circle = L.circleMarker([p.lat, p.lon], {{
    radius: radius,
    fillColor: p.color,
    color: 'transparent',
    weight: 0,
    fillOpacity: 0.85,
  }});
  circle.bindTooltip(
    '<b style="font-size:14px">'+p.label+'</b><br>'
    +'<span style="color:rgba(255,255,255,0.5);font-size:11px">'+p.category+'</span><br><br>'
    +p.metric_name+': <b>'+p.metric.toLocaleString()+'</b><br>'
    +'Fatalities: <b>'+p.fatalities.toLocaleString()+'</b>',
    {{className:'dark-tip', sticky:true}}
  );
  circle.on('click', function(e) {{
    L.DomEvent.stopPropagation(e);
    selectCountry2d(p.country);
  }});
  circle.addTo(map2d);
}});

function infoRow2d(color, label, val){{
  return '<div style="display:flex;justify-content:space-between;align-items:center;'
    +'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
    +'<span style="color:rgba(255,255,255,0.55);font-size:11px;letter-spacing:.04em;">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;font-size:13px;">'+val.toLocaleString()+'</span>'
    +'</div>';
}}

function showInfoPanel2d(name){{
  const c = countryData2d[name];
  if(!c) return;
  document.getElementById('infopanel2d-content').innerHTML =
    '<div style="font-size:15px;font-weight:800;margin-bottom:1px;padding-right:18px;">'+name+'</div>'
    +'<div style="font-size:9px;color:rgba(255,255,255,0.3);letter-spacing:1.2px;margin-bottom:10px;">ACLED CONFLICT DATA</div>'
    +infoRow2d('#ef4444','FATALITIES',    c.fatalities)
    +infoRow2d('#f87171','BATTLES',       c.battles)
    +infoRow2d('#f59e0b','EXPLOSIONS',    c.explosions)
    +infoRow2d('#fde047','CIV. VIOLENCE', c.civ_violence)
    +infoRow2d('#60a5fa','STRATEGIC',     c.strategic)
    +infoRow2d('#a78bfa','PROTESTS',      c.protests)
    +infoRow2d('#f472b6','RIOTS',         c.riots)
    +'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0 0;">'
    +'<span style="color:rgba(255,255,255,0.45);font-size:10px;">'+c.metric_name.toUpperCase()+'</span>'
    +'<span style="color:white;font-weight:700;font-size:13px;">'+c.metric_total.toLocaleString()+'</span>'
    +'</div>';
  document.getElementById('infopanel2d').style.display='block';
}}

function closePanel2d(){{
  document.getElementById('infopanel2d').style.display='none';
}}

function selectCountry2d(name){{
  const c = countryData2d[name];
  if(!c){{ closePanel2d(); return; }}
  const pad = Math.max(0.5, (c.maxlat-c.minlat)*0.15);
  const padLon = Math.max(0.5, (c.maxlon-c.minlon)*0.15);
  map2d.flyToBounds(
    [[c.minlat-pad, c.minlon-padLon],[c.maxlat+pad, c.maxlon+padLon]],
    {{duration:1.0, easeLinearity:0.3, maxZoom:9}}
  );
  showInfoPanel2d(name);
}}

function toggleFullscreen2d(){{
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  if(!isFs && req) {{
    req.call(el);
  }} else if(isFs && exit) {{
    exit.call(document);
  }}
}}
function _updateFsBtn2d(){{
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  const btn = document.getElementById('fullscreenbtn2d');
  if(btn) btn.innerHTML = isFs ? '&#x26F6;&nbsp; EXIT FULL' : '&#x26F6;&nbsp; FULLSCREEN';
  setTimeout(function(){{ map2d.invalidateSize(); }}, 100);
}}
document.addEventListener('fullscreenchange', _updateFsBtn2d);
document.addEventListener('webkitfullscreenchange', _updateFsBtn2d);
function recenter2d(){{
  map2d.flyTo([20, 10], 2, {{duration:1.0, easeLinearity:0.3}});
  closePanel2d();
}}
map2d.on('click', closePanel2d);
</script></body></html>"""
                                st.components.v1.html(leaflet_html, height=map_h, scrolling=False)

                        # Country info panel now rendered in-map via JS

                        with st.expander("📖  What do these categories mean?", expanded=False):
                            st.markdown("""
**🔴 Battles** — Armed clashes between two or more organized groups, where both sides are actively fighting. This includes frontline combat, ambushes, and firefights.
*Examples: a government army engaging rebel fighters in a town, two rival militias clashing over territory, a counter-insurgency offensive.*

**🟠 Explosions / Remote Violence** — Attacks using weapons that can strike from a distance without direct armed confrontation. One side inflicts violence without the other being able to fight back directly.
*Examples: airstrikes on a city district, artillery shelling of a village, a drone strike on a convoy, a roadside bomb (IED) detonating on a military patrol.*

**🟡 Violence Against Civilians** — Deliberate, targeted violence by an organized group directed at unarmed civilians. The victims are not combatants.
*Examples: a militia executing villagers, armed groups abducting aid workers, sexual violence used as a weapon of war, a mob killing members of an ethnic minority.*

**🔵 Strategic Developments** — Significant non-violent political or military actions that change the operational landscape. These events signal shifts in power or posture without direct fighting.
*Examples: a rebel group seizing a government building peacefully, a ceasefire agreement being signed, armed forces withdrawing from a region, a faction announcing a merger or split.*

**🟣 Protests** — Organized, primarily non-violent demonstrations by civilians expressing political or social grievances.
*Examples: crowds marching against a government policy, a sit-in at a parliament building, a student demonstration demanding elections, a labor strike turning confrontational.*

**⚔️ Riots** — Violent, often spontaneous collective action by civilians — typically disorganized and not directed by a single command structure.
*Examples: crowds attacking police after a disputed election, looting and arson following a fuel price hike, inter-communal street fighting between ethnic groups.*
""")

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

                        # ── Map AI Analysis ───────────────────────────
                        st.markdown("---")
                        st.markdown("### 🤖 AI Map Intelligence")
                        st.caption("Enter any country visible on the map to get an AI-generated conflict summary.")
                        map_ai_country = st.text_input(
                            "Country name",
                            placeholder="e.g. Libya, Sudan, Ukraine",
                            key="map_ai_country",
                        )
                        if st.button("Analyze Country", key="map_ai_btn") and map_ai_country:
                            with st.spinner(f"Analyzing {map_ai_country}..."):
                                try:
                                    _map_ai_df = fetch_acled_arcgis_monthly()
                                    _map_ai_df = _map_ai_df[
                                        _map_ai_df["country"].str.strip().str.lower()
                                        == map_ai_country.strip().lower()
                                    ]
                                    if _map_ai_df.empty:
                                        st.warning(f"No data found for '{map_ai_country}'.")
                                    else:
                                        _map_idx = compute_escalation_index(_map_ai_df, map_ai_country)
                                        if _map_idx.empty:
                                            st.warning("Could not compute index for this country.")
                                        else:
                                            _ml = _map_idx.iloc[-1]
                                            _mp = _map_idx.tail(4).iloc[0]
                                            _mt = "rising" if _ml["index_smoothed"] > _mp["index_smoothed"] else "falling"
                                            _mr = _map_idx.tail(3)
                                            _ms = "\n".join(
                                                f"  {r['event_month'].strftime('%b %Y')}: index={r['index_smoothed']:.1f}, "
                                                f"battles={int(r.get('battles',0))}, explosions={int(r.get('explosions_remote_violence',0))}, "
                                                f"fatalities={int(r.get('fatalities',0))}, "
                                                f"strategic={int(r.get('strategic_developments',0))}"
                                                for _, r in _mr.iterrows()
                                            )
                                            _map_prompt = (
                                                f"Give a 2-sentence intelligence briefing on the current conflict situation "
                                                f"in {map_ai_country}. The escalation index is {_mt}, "
                                                f"currently at {_ml['index_smoothed']:.1f}/100.\n\n"
                                                f"Recent data:\n{_ms}\n\n"
                                                f"Sentence 1: current situation and dominant conflict type. "
                                                f"Sentence 2: trajectory and key risk."
                                            )
                                            _map_result = _call_claude(
                                                _map_prompt,
                                                system=(
                                                    "You are a concise geopolitical intelligence analyst. "
                                                    "Plain English, no bullet points, no mention of ACLED. "
                                                    "Be direct and specific."
                                                ),
                                                max_tokens=200,
                                            )
                                            st.markdown(
                                                f"<div style='background:#0f172a;border:1px solid #1e3a5f;"
                                                f"border-radius:8px;padding:16px 20px;color:#e2e8f0;"
                                                f"font-size:14px;line-height:1.7;'>"
                                                f"<b style='color:#60a5fa;font-size:13px;letter-spacing:.06em;'>"
                                                f"{map_ai_country.upper()} — INTELLIGENCE BRIEF</b><br><br>"
                                                f"{_map_result}</div>",
                                                unsafe_allow_html=True,
                                            )
                                except Exception as e:
                                    st.error(f"Map AI analysis failed: {e}")

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
    Alexander Armand-Blumberg · AEGIS
</div>
""",
    unsafe_allow_html=True,
)
