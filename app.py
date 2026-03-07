import io
import base64
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from streamlit_plotly_events import plotly_events
    _HAS_PLOTLY_EVENTS = True
except Exception:
    _HAS_PLOTLY_EVENTS = False


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="ZoomedLogo.png",
    layout="wide"
)

HF_WORLD_CSV_URL = "https://huggingface.co/datasets/alex-armand-blumberg/UCDP/resolve/main/GEDEvent_v25_1%203.csv"
UKRAINE_SAMPLE_PATH = Path("ukraine_sample.csv")
VIDEO_PATH = Path("logo1.mp4")


# ----------------------------
# News
# ----------------------------
@st.cache_data(ttl=900)
def fetch_rss_items(rss_url: str, max_items: int = 6):
    r = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = ""
        src = item.find("source")
        if src is not None and src.text:
            source = src.text.strip()

        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                }
            )
    return items


def render_news():
    st.markdown("## Current Conflict News")
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

        st.caption("Updates automatically every ~15 minutes.")
    except Exception as e:
        st.warning(f"Live news feed failed to load: {e}")


# ----------------------------
# Robust CSV loading
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

    raise last_err if last_err else ValueError("Could not parse CSV.")


@st.cache_data(show_spinner=False)
def read_csv_path_robust(path_str: str) -> pd.DataFrame:
    return read_csv_bytes_robust(Path(path_str).read_bytes())


@st.cache_data(show_spinner=False)
def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def load_world_dataset_for_map() -> pd.DataFrame:
    return read_csv_bytes_robust(download_bytes(HF_WORLD_CSV_URL))


# ----------------------------
# Helpers
# ----------------------------
def parse_thresholds(raw: str) -> list[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [float(p.strip()) for p in raw.split(",") if p.strip()][:2]


def require_columns(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: Missing column(s): {missing}. Available columns: {list(df.columns)[:50]}")


def build_country_daily(df: pd.DataFrame, country_col: str, date_col: str, fatal_col: str) -> pd.DataFrame:
    d = df[[country_col, date_col, fatal_col]].copy()
    d[country_col] = d[country_col].astype(str)
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


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("AEGIS Control Bar")

if VIDEO_PATH.exists():
    video_bytes = VIDEO_PATH.read_bytes()
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
    help="Loads ukraine_sample.csv from the repo."
)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Must match the country values in your dataset exactly."
)

with st.sidebar.expander("Advanced Settings"):
    country_col = st.text_input("Name of Country Column", "country")
    date_col = st.text_input("Name of Date Column", "date_start")
    fatalities_col = st.text_input("Name of Fatalities Column", "best")

    rolling_window = st.number_input(
        "Rolling window (days)",
        min_value=1, max_value=365, value=30, step=1
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        "25,1000"
    )

    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1, max_value=60, value=7, step=1
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

st.sidebar.markdown("---")
show_map = st.sidebar.checkbox("Show interactive map", value=True)
override_map_dates = st.sidebar.checkbox("Override map date range", value=False)

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
# Header
# ----------------------------
with st.expander("Current Conflict News", expanded=False):
    render_news()

col1, col2 = st.columns([1, 12])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("AEGIS — Escalation Detection Demo")

st.caption("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")


# ----------------------------
# Plot dataset loader
# ----------------------------
def load_primary_dataset_for_plot():
    if use_demo:
        if not UKRAINE_SAMPLE_PATH.exists():
            raise FileNotFoundError("Demo file not found: ukraine_sample.csv")
        return read_csv_path_robust(str(UKRAINE_SAMPLE_PATH)), "Built-in demo (ukraine_sample.csv)"

    if uploaded is None:
        return None, None

    return read_csv_bytes_robust(uploaded.getvalue()), "Uploaded CSV"


# ----------------------------
# Escalation plot first
# ----------------------------
st.subheader("Escalation plot")

df_raw_plot, plot_source = (None, None)
try:
    df_raw_plot, plot_source = load_primary_dataset_for_plot()
except Exception as e:
    st.error(str(e))

plot_ready = True

if df_raw_plot is None:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**.")
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
        daily = build_country_daily(df_raw_plot, country_col, date_col, fatalities_col)
        c_daily = daily[daily["country"] == country_name].copy()

        if c_daily.empty:
            st.warning(f"No rows found for country='{country_name}'.")
        else:
            c_daily = c_daily.set_index("date").sort_index()
            c_daily["rolling"] = c_daily["fatalities"].rolling(int(rolling_window), min_periods=1).sum()

            thresholds = parse_thresholds(thresholds_raw)
            if not thresholds:
                st.error("Please provide at least one threshold.")
            else:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(c_daily.index, c_daily["rolling"], label="Rolling fatalities", linewidth=2)

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


# ----------------------------
# World-monitor-style map section
# ----------------------------
if show_map:
    st.markdown("## Interactive map")

    if not _HAS_PLOTLY:
        st.info("Interactive map requires Plotly.")
    else:
        try:
            df_world = load_world_dataset_for_map()

            world_country_col = "country"
            world_date_col = "date_start"
            world_fatal_col = "best"

            require_columns(df_world, [world_country_col, world_date_col, world_fatal_col], "World map dataset")

            df_world = df_world[[world_country_col, world_date_col, world_fatal_col]].copy()
            df_world[world_date_col] = pd.to_datetime(df_world[world_date_col], errors="coerce")
            df_world = df_world.dropna(subset=[world_date_col])
            df_world[world_fatal_col] = pd.to_numeric(df_world[world_fatal_col], errors="coerce").fillna(0)

            min_dt = df_world[world_date_col].min().date()
            max_dt = df_world[world_date_col].max().date()

            if override_map_dates:
                date_pick = st.sidebar.date_input(
                    "Map date range",
                    value=(min_dt, max_dt),
                    min_value=min_dt,
                    max_value=max_dt,
                    key="map_date_range",
                )
                if isinstance(date_pick, tuple) and len(date_pick) == 2:
                    start_dt, end_dt = date_pick
                else:
                    start_dt, end_dt = min_dt, max_dt
            else:
                start_dt, end_dt = min_dt, max_dt

            df_world = df_world[
                (df_world[world_date_col].dt.date >= start_dt) &
                (df_world[world_date_col].dt.date <= end_dt)
            ]

            by_country = (
                df_world.groupby(world_country_col, as_index=False)[world_fatal_col]
                .sum()
                .rename(columns={world_country_col: "country", world_fatal_col: "fatalities"})
                .sort_values("fatalities", ascending=False)
            )

            # clicked country state
            if "selected_country" not in st.session_state:
                st.session_state.selected_country = country_name

            left, right = st.columns([3.2, 1.2], gap="large")

            with left:
                fig = px.choropleth(
                    by_country,
                    locations="country",
                    locationmode="country names",
                    color="fatalities",
                    hover_name="country",
                    hover_data={"country": False, "fatalities": False},
                    title=f"Fatalities by country ({start_dt.year}–{end_dt.year})",
                    color_continuous_scale="Blues",
                )

                fig.update_traces(
                    hovertemplate=(
                        "<b>%{location}</b>"
                        "<br>Total fatalities: %{z:,}"
                        "<extra></extra>"
                    )
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=60, b=0),
                    dragmode=False,
                )

                if _HAS_PLOTLY_EVENTS:
                    selected = plotly_events(
                        fig,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        override_height=600,
                        key="world_map_click",
                    )
                    if selected:
                        clicked = selected[0].get("pointNumber")
                        if clicked is not None and clicked < len(by_country):
                            st.session_state.selected_country = by_country.iloc[clicked]["country"]
                else:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Install streamlit-plotly-events for click-to-focus behavior.")

                st.caption("To change the date range, enable **Override map date range** in the sidebar.")

            with right:
                sel_country = st.session_state.selected_country
                country_row = by_country[by_country["country"] == sel_country]

                st.markdown("### Country Focus")

                if country_row.empty:
                    st.info("Click a country on the map to inspect it.")
                else:
                    total_fatalities = float(country_row["fatalities"].iloc[0])

                    st.markdown(
                        f"""
                        <div style="
                            padding:16px;
                            border:1px solid rgba(255,255,255,0.10);
                            border-radius:16px;
                            background: rgba(255,255,255,0.03);
                        ">
                            <div style="font-size:28px; font-weight:800; margin-bottom:8px;">{sel_country}</div>
                            <div style="font-size:15px; opacity:0.8; margin-bottom:14px;">
                                Country-level intelligence summary
                            </div>
                            <div style="font-size:36px; font-weight:800; line-height:1;">{total_fatalities:,.0f}</div>
                            <div style="font-size:14px; opacity:0.75; margin-top:4px;">Recorded fatalities in selected range</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # zoomed one-country mini map
                    focus_df = by_country[by_country["country"] == sel_country].copy()

                    focus_fig = px.choropleth(
                        focus_df,
                        locations="country",
                        locationmode="country names",
                        color="fatalities",
                        hover_name="country",
                        hover_data={"country": False, "fatalities": False},
                        color_continuous_scale="Blues",
                    )
                    focus_fig.update_traces(
                        hovertemplate=(
                            "<b>%{location}</b>"
                            "<br>Total fatalities: %{z:,}"
                            "<extra></extra>"
                        )
                    )
                    focus_fig.update_geos(
                        fitbounds="locations",
                        visible=False,
                        showcountries=True,
                        countrycolor="rgba(255,255,255,0.25)"
                    )
                    focus_fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=260,
                        coloraxis_showscale=False,
                    )

                    st.plotly_chart(focus_fig, use_container_width=True)

                    if st.button(f"Use {sel_country} in plot", key="use_country_btn"):
                        country_name = sel_country
                        st.session_state["country_name_override"] = sel_country

        except Exception as e:
            st.error(f"Map error: {e}")
