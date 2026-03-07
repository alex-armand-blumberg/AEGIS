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
        value=50,
        step=1,
        help="Months where the Escalation Index exceeds this are flagged as escalation events.",
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
    value=True,
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


# ----------------------------
# Escalation Index plot section
# ----------------------------
st.subheader("Escalation Index")

st.caption(
    "Depending on the date range selected, index plotting time may range from a couple seconds to a couple minutes. "
    "One page of events (5,000 events) takes ~4 seconds to load."
)

if not run_btn:
    st.info(
        "Enter a country name in the sidebar (e.g. **Ukraine**, **Sudan**, **Myanmar**) "
        "and click **Generate plot**. The interactive map appears below."
    )
else:
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
                            dates     = pd.to_datetime(idx_df["event_month"])
                            esc_rows  = idx_df[idx_df["index_smoothed"] > escalation_threshold]

                            # ── Pre-escalation warning signal ────────────
                            # Fires when index is BELOW threshold but leading
                            # components (strategic + explosions) are both
                            # elevated AND index has been rising for 2+ months.
                            idx_df["_rising"] = (
                                idx_df["index_smoothed"].diff() > 0
                            ).astype(int)
                            idx_df["_rising2"] = (
                                idx_df["_rising"] + idx_df["_rising"].shift(1)
                            ).fillna(0)
                            idx_df["_lead_signal"] = (
                                idx_df["c_strategic"] + idx_df["c_explosion"]
                            )
                            warn_rows = idx_df[
                                (idx_df["index_smoothed"] < escalation_threshold)
                                & (idx_df["_lead_signal"] > 1.2)
                                & (idx_df["_rising2"] >= 2)
                            ]

                            # ── 3-month forecast (linear trend on last 6 months)
                            import numpy as np
                            forecast_dates, forecast_vals, forecast_lo, forecast_hi = [], [], [], []
                            if len(idx_df) >= 6:
                                tail = idx_df.tail(6).copy()
                                tail_x = np.arange(len(tail))
                                tail_y = tail["index_smoothed"].values
                                coeffs = np.polyfit(tail_x, tail_y, 1)
                                slope, intercept = coeffs
                                last_date = pd.to_datetime(idx_df["event_month"].iloc[-1])
                                residuals = tail_y - np.polyval(coeffs, tail_x)
                                std_err   = residuals.std()
                                for i in range(1, 4):
                                    fd = last_date + pd.DateOffset(months=i)
                                    fv = np.polyval(coeffs, len(tail) - 1 + i)
                                    fv = float(np.clip(fv, 0, 100))
                                    forecast_dates.append(fd)
                                    forecast_vals.append(fv)
                                    forecast_lo.append(max(0,   fv - 1.5 * std_err))
                                    forecast_hi.append(min(100, fv + 1.5 * std_err))

                            # ── Main escalation index chart ──────────────
                            fig, ax = plt.subplots(figsize=(12, 5))
                            fig.patch.set_facecolor("#020617")
                            ax.set_facecolor("#0f172a")

                            ax.axhspan(escalation_threshold, 100, alpha=0.07, color="#ef4444", zorder=0)
                            ax.axhline(
                                y=escalation_threshold, color="#ef4444",
                                linestyle="--", linewidth=1.2,
                                label=f"Alert threshold ({escalation_threshold})",
                            )
                            ax.plot(dates, idx_df["escalation_index"],
                                    color="#60a5fa", alpha=0.25, linewidth=1)
                            ax.plot(
                                dates, idx_df["index_smoothed"],
                                color="#60a5fa", linewidth=2.5,
                                label=f"Escalation Index ({w}-month smoothed)",
                            )
                            if not esc_rows.empty:
                                ax.scatter(
                                    pd.to_datetime(esc_rows["event_month"]),
                                    esc_rows["index_smoothed"],
                                    color="#ef4444", s=60, zorder=5,
                                    label=f"Months where escalation was flagged ({len(esc_rows)} / {len(df_acled)} months)",
                                )

                            # Pre-escalation warning dots (orange diamonds)
                            if not warn_rows.empty:
                                ax.scatter(
                                    pd.to_datetime(warn_rows["event_month"]),
                                    warn_rows["index_smoothed"],
                                    color="#f97316", s=90, marker="D",
                                    zorder=6,
                                    label=f"Pre-escalation warning ({len(warn_rows)} months)",
                                )

                            # 3-month forecast line + uncertainty band
                            if forecast_dates:
                                last_hist_date = pd.to_datetime(idx_df["event_month"].iloc[-1])
                                last_hist_val  = float(idx_df["index_smoothed"].iloc[-1])
                                fc_dates_plot  = [last_hist_date] + forecast_dates
                                fc_vals_plot   = [last_hist_val]  + forecast_vals
                                fc_lo_plot     = [last_hist_val]  + forecast_lo
                                fc_hi_plot     = [last_hist_val]  + forecast_hi
                                ax.plot(
                                    fc_dates_plot, fc_vals_plot,
                                    color="#a78bfa", linewidth=2,
                                    linestyle=":", zorder=4,
                                    label="3-month forecast (linear trend)",
                                )
                                ax.fill_between(
                                    fc_dates_plot, fc_lo_plot, fc_hi_plot,
                                    color="#a78bfa", alpha=0.15, zorder=3,
                                )

                            ax.set_title(
                                f"AEGIS Escalation Index — {selected_country}",
                                color="white", fontsize=15, pad=12,
                            )
                            ax.set_xlabel("Month", color="#94a3b8")
                            ax.set_ylabel("Escalation Index (0–100)", color="#94a3b8")
                            ax.tick_params(colors="#94a3b8")
                            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                            plt.xticks(rotation=35, ha="right")
                            ax.set_ylim(0, 105)
                            ax.grid(True, alpha=0.15, color="#334155")
                            for spine in ax.spines.values():
                                spine.set_edgecolor("#334155")
                            ax.legend(
                                facecolor="#1e293b", edgecolor="#334155",
                                labelcolor="white", fontsize=10,
                            )
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            _prog.progress(100, text="Done.")
                            _prog.empty()

                            # ── Signal legend expander ───────────────────
                            with st.expander("How to read the signals", expanded=False):
                                st.markdown(
                                    "**Blue line** — smoothed Escalation Index (0-100).\n\n"
                                    "**Red dots** — months above your alert threshold. Sustained elevated conflict.\n\n"
                                    "**Orange diamonds** — pre-escalation warning. Index still below threshold, "
                                    "but strategic developments + explosions both elevated AND index rising 2+ months. "
                                    "Fires *before* red dots appear.\n\n"
                                    "**Purple dotted line** — 3-month linear forecast with uncertainty band. "
                                    "Directional signal only, not a causal model."
                                )

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

        except Exception as e:
            _prog.empty()
            st.error(f"Error computing escalation index: {e}")


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
