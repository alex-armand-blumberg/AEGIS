# app.py
import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="AEGIS Escalation Detection", layout="wide")

APP_DIR = Path(__file__).resolve().parent

# Worldwide dataset on Google Drive (must be shared so "Anyone with the link" can access)
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1lg3YUmyxb8aqXCLtnPgGXnoJb8pIAEGF/view?usp=sharing"
CACHE_FILENAME = "aegis_global.csv"  # saved alongside app.py


# =========================
# Google Drive download (handles virus-scan / confirm page)
# =========================
def extract_gdrive_file_id(url: str) -> str | None:
    # supports: https://drive.google.com/file/d/<ID>/view?...
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def _looks_like_html(b: bytes) -> bool:
    head = b[:500].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title>google drive" in head


def _get_confirm_token_from_html(html_text: str) -> str | None:
    # Try common confirm patterns Google uses
    m = re.search(r"confirm=([0-9A-Za-z_]+)", html_text)
    if m:
        return m.group(1)

    # Sometimes the confirm token is inside the download link
    m = re.search(r'href="(/uc\?export=download[^"]+)"', html_text)
    if m:
        # parse confirm=... from that href
        href = m.group(1).replace("&amp;", "&")
        m2 = re.search(r"confirm=([0-9A-Za-z_]+)", href)
        if m2:
            return m2.group(1)

    return None


@st.cache_data(show_spinner=False)
def download_gdrive_csv(url: str, local_name: str) -> Path:
    file_id = extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError("Could not parse Google Drive file id from the URL.")

    local_path = APP_DIR / local_name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    session = requests.Session()

    # Step 1: initial request
    base = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    r = session.get(base, params=params, stream=True, timeout=120)

    # If Google returns an HTML interstitial ("can't scan for viruses"), get confirm token & retry
    first_chunk = next(r.iter_content(chunk_size=1024), b"")
    content = first_chunk + b"".join([])  # keep type as bytes

    # If the content-type is html OR the bytes look like html, we need to confirm
    if "text/html" in r.headers.get("Content-Type", "") or _looks_like_html(first_chunk):
        # read full text (not huge)
        r_text = first_chunk + r.content
        html = r_text.decode("utf-8", errors="ignore")

        token = _get_confirm_token_from_html(html)
        if not token:
            raise ValueError(
                "Google Drive returned a virus-scan confirmation page, but the app couldn't extract the confirm token. "
                "Make sure the file is shared as 'Anyone with the link'."
            )

        r2 = session.get(base, params={"export": "download", "id": file_id, "confirm": token}, stream=True, timeout=300)
        r2.raise_for_status()

        # download file
        with open(local_path, "wb") as f:
            for chunk in r2.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    else:
        # It’s already the real file; write what we already read + the rest
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(first_chunk)
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise ValueError("Downloaded file is empty. Check Google Drive permissions/link.")

    return local_path


# =========================
# CSV reading / processing
# =========================
@st.cache_data(show_spinner=False)
def read_csv_any(source) -> pd.DataFrame:
    if isinstance(source, Path):
        return pd.read_csv(source)
    if hasattr(source, "read"):  # UploadedFile
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return pd.read_csv(io.BytesIO(data))
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source))
    raise ValueError("Unsupported CSV source type.")


def parse_thresholds(raw: str) -> list[float]:
    vals = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Enter at least one threshold (e.g., 25 or 25,1000).")
    return vals[:2]


def coerce_columns(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str) -> pd.DataFrame:
    missing = [c for c in [country_col, date_col, fatalities_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s): {missing}. "
            f"Available columns (first 30): {list(df.columns)[:30]}"
        )

    out = df.copy()
    out[country_col] = out[country_col].astype(str).str.strip()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[fatalities_col] = pd.to_numeric(out[fatalities_col], errors="coerce").fillna(0)
    out = out.dropna(subset=[date_col])
    return out


def daily_series_for_country(df: pd.DataFrame, country_name: str, country_col: str, date_col: str, fatalities_col: str):
    sub = df[df[country_col] == country_name].copy()
    if sub.empty:
        raise ValueError(f"No rows found for country='{country_name}'. Check spelling/case.")
    daily = (
        sub.groupby(pd.Grouper(key=date_col, freq="D"))[fatalities_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date", fatalities_col: "fatalities"})
        .sort_values("date")
    )
    return daily


def detect_escalation_starts(daily: pd.DataFrame, rolling_days: int, threshold: float, persistence_days: int):
    d = daily.copy()
    d["rolling"] = d["fatalities"].rolling(int(rolling_days), min_periods=1).sum()
    d["above"] = d["rolling"] >= float(threshold)

    d["persistent"] = (
        d["above"].rolling(int(persistence_days), min_periods=int(persistence_days)).sum()
        == int(persistence_days)
    )
    d["persistent"] = d["persistent"].fillna(False)
    d["escalation_start"] = d["persistent"] & (~d["persistent"].shift(1).fillna(False))
    return d


# =========================
# Sidebar video (optional)
# =========================
def sidebar_video(file_name: str, height_px: int = 170):
    video_path = APP_DIR / file_name
    if not video_path.exists():
        return
    b64vid = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    html = f"""
    <div style="width:100%; overflow:hidden; border-radius:16px; background:#000;">
      <video style="width:100%; height:{height_px}px; object-fit:cover; display:block;"
        autoplay muted loop playsinline>
        <source src="data:video/mp4;base64,{b64vid}" type="video/mp4"/>
      </video>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


# =========================
# Map
# =========================
def make_world_map(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str):
    max_date = pd.to_datetime(df[date_col]).max()
    if pd.isna(max_date):
        st.info("Map unavailable: date column could not be parsed.")
        return

    st.subheader("Global Snapshot")
    st.caption("Interactive map based on the dataset currently loaded (upload, demo, or Google Drive).")

    days_back = st.slider("Map time window (days back from latest date)", 30, 3650, 365, 30)
    start = max_date - pd.Timedelta(days=int(days_back))

    dsub = df[(df[date_col] >= start) & (df[date_col] <= max_date)].copy()
    if dsub.empty:
        st.info("No rows in the selected map time window.")
        return

    agg = (
        dsub.groupby(country_col, as_index=False)[fatalities_col]
        .sum()
        .rename(columns={country_col: "country", fatalities_col: "fatalities"})
    )

    fig = px.choropleth(
        agg,
        locations="country",
        locationmode="country names",
        color="fatalities",
        hover_name="country",
        color_continuous_scale="Blues",
        title=f"Total fatalities by country (last {days_back} days)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# =========================
# UI
# =========================
st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers.")

sidebar_video("logo1.mp4", height_px=170)

st.sidebar.header("Inputs")

use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help="Loads a small CSV from the repo named 'ukraine_sample.csv'.",
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset as a CSV.")

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Name must match the dataset exactly (e.g., Ukraine, Mexico, Syria).",
)

country_col = st.sidebar.text_input(
    "Country column",
    "country",
    help="Column name containing country names.",
)

date_col = st.sidebar.text_input(
    "Date column",
    "date_start",
    help="Column name containing dates (will be parsed).",
)

fatalities_col = st.sidebar.text_input(
    "Fatalities column",
    "best",
    help="Column name containing fatalities (numeric).",
)

rolling_window = st.sidebar.number_input(
    "Rolling window (days)", min_value=1, max_value=365, value=30, step=1, help="Rolling sum window."
)

thresholds_raw = st.sidebar.text_input(
    "Escalation threshold(s) (comma-separated, up to 2)",
    "25,1000",
    help="Example: 25 or 25,1000 (two thresholds will be plotted).",
)

persistence = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
    help="How many consecutive days above threshold are required to count as an escalation.",
)

run_btn = st.sidebar.button("Generate plot")

# =========================
# Load data
# =========================
df_raw = None
data_source_label = None

try:
    if use_sample:
        sample_path = APP_DIR / "ukraine_sample.csv"
        if not sample_path.exists():
            st.error("Demo is enabled, but 'ukraine_sample.csv' is missing from your repo.")
        else:
            df_raw = read_csv_any(sample_path)
            data_source_label = "Built-in demo (ukraine_sample.csv)"
    elif uploaded is not None:
        df_raw = read_csv_any(uploaded)
        data_source_label = "Uploaded CSV"
    else:
        st.info("No upload/demo selected — loading the Google Drive dataset for the map.")
        local_path = download_gdrive_csv(GOOGLE_DRIVE_URL, CACHE_FILENAME)
        df_raw = read_csv_any(local_path)
        data_source_label = "Google Drive dataset"
except Exception as e:
    st.error(f"Data loading error: {e}")

df = None
if df_raw is not None and data_source_label:
    st.caption(f"Data source: **{data_source_label}**")
    try:
        df = coerce_columns(df_raw, country_col, date_col, fatalities_col)
    except Exception as e:
        st.error(f"Column/parsing error: {e}")
        df = None

# =========================
# Map placeholder content
# =========================
if df is not None:
    make_world_map(df, country_col=country_col, date_col=date_col, fatalities_col=fatalities_col)
else:
    st.info("Load data (upload, demo, or Google Drive) to display the interactive map here.")

# =========================
# Plot on demand
# =========================
if run_btn:
    if df is None:
        st.warning("No usable dataset loaded yet.")
        st.stop()

    try:
        thresholds = parse_thresholds(thresholds_raw)
        daily = daily_series_for_country(df, country_name, country_col, date_col, fatalities_col)

        detected = []
        for th in thresholds:
            dth = detect_escalation_starts(
                daily=daily,
                rolling_days=int(rolling_window),
                threshold=float(th),
                persistence_days=int(persistence),
            )
            dth["threshold"] = float(th)
            detected.append(dth)

        base = detected[0][["date", "fatalities", "rolling"]].copy()

        col1, col2 = st.columns([2, 1], vertical_alignment="top")
        with col1:
            fig = plt.figure(figsize=(12, 5))
            plt.plot(base["date"], base["rolling"], label="Rolling fatalities")

            for dth in detected:
                th = dth["threshold"].iloc[0]
                plt.axhline(y=th, linestyle="--", label=f"Threshold {th:g}")
                starts = dth[dth["escalation_start"]]
                if not starts.empty:
                    plt.scatter(starts["date"], starts["rolling"], s=60, label=f"Escalation starts (≥{th:g})")

            plt.title(f"AEGIS Escalation Detection — {country_name} (rolling={int(rolling_window)}d, persistence={int(persistence)}d)")
            plt.xlabel("Date")
            plt.ylabel("Rolling window fatalities")
            plt.tight_layout()
            plt.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Summary")
            st.write(f"Rows (daily): **{len(base)}**")
            for dth in detected:
                th = dth["threshold"].iloc[0]
                starts = dth[dth["escalation_start"]]
                st.markdown(f"**Threshold {th:g}**")
                st.write(f"Days above: **{int(dth['above'].sum())}**")
                st.write(f"Days persistent: **{int(dth['persistent'].sum())}**")
                st.write(f"Escalation starts: **{int(dth['escalation_start'].sum())}**")
                if not starts.empty:
                    st.write("First escalation starts:")
                    st.dataframe(starts[["date", "rolling"]].head(10), use_container_width=True)
                st.divider()

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload a CSV (or enable the demo), then click **Generate plot**.")
