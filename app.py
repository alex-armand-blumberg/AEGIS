# app.py
import io
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

# Your Google Drive file (worldwide dataset)
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1lg3YUmyxb8aqXCLtnPgGXnoJb8pIAEGF/view?usp=sharing"
CACHE_FILENAME = "aegis_global.csv"  # saved in Streamlit's working dir


# =========================
# Helpers
# =========================
def extract_gdrive_file_id(url: str) -> str | None:
    # Works for /file/d/<ID>/view style URLs
    try:
        parts = url.split("/d/")
        if len(parts) < 2:
            return None
        tail = parts[1]
        file_id = tail.split("/")[0]
        return file_id.strip()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def download_gdrive_csv(url: str, local_name: str) -> Path:
    """
    Downloads a Google Drive file using the 'uc?export=download&id=' endpoint.
    Saves locally (so we don't redownload every rerun).
    """
    file_id = extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError("Could not parse Google Drive file id from the URL.")

    local_path = APP_DIR / local_name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    # Direct download endpoint
    dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with requests.get(dl_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if local_path.stat().st_size == 0:
        raise ValueError("Downloaded file is empty. Check sharing permissions (must be accessible via link).")

    return local_path


@st.cache_data(show_spinner=False)
def read_csv_any(source) -> pd.DataFrame:
    """
    source can be:
      - Path
      - UploadedFile (streamlit)
      - bytes / BytesIO
    """
    if isinstance(source, Path):
        return pd.read_csv(source)
    if hasattr(source, "read"):  # UploadedFile or file-like
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
        raise ValueError("Please enter at least one threshold (e.g., 25 or 25,1000).")
    # Keep at most 2, as requested
    return vals[:2]


def coerce_columns(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str) -> pd.DataFrame:
    missing = [c for c in [country_col, date_col, fatalities_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Available columns: {list(df.columns)[:25]}...")

    out = df.copy()

    out[country_col] = out[country_col].astype(str).str.strip()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[fatalities_col] = pd.to_numeric(out[fatalities_col], errors="coerce").fillna(0)

    out = out.dropna(subset=[date_col])
    return out


def daily_series_for_country(
    df: pd.DataFrame,
    country_name: str,
    country_col: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
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


def detect_escalation_starts(
    daily: pd.DataFrame,
    rolling_days: int,
    threshold: float,
    persistence_days: int,
) -> pd.DataFrame:
    d = daily.copy()
    d["rolling"] = d["fatalities"].rolling(int(rolling_days), min_periods=1).sum()

    d["above"] = d["rolling"] >= float(threshold)
    # "persistent" means consecutive run of >= persistence_days above threshold
    # We'll compute rolling sum of True values over the window
    d["persistent"] = (
        d["above"].rolling(int(persistence_days), min_periods=int(persistence_days)).sum() == int(persistence_days)
    )
    d["persistent"] = d["persistent"].fillna(False)

    d["escalation_start"] = d["persistent"] & (~d["persistent"].shift(1).fillna(False))
    return d


def sidebar_video(file_name: str, height_px: int = 170):
    """
    Streamlit's st.video is hard to style/fit. This HTML block gives you:
    - full width
    - rounded corners
    - object-fit: cover
    - autoplay + loop + muted (required by most browsers)
    """
    video_path = APP_DIR / file_name
    if not video_path.exists():
        st.sidebar.warning(f"Sidebar video not found: {file_name}")
        return

    b64 = video_path.read_bytes()
    # Use base64 to avoid path issues on Streamlit Cloud
    import base64

    b64vid = base64.b64encode(b64).decode("utf-8")
    html = f"""
    <div style="width:100%; overflow:hidden; border-radius:16px; background:#000;">
      <video
        style="width:100%; height:{height_px}px; object-fit:cover; display:block;"
        autoplay
        muted
        loop
        playsinline
      >
        <source src="data:video/mp4;base64,{b64vid}" type="video/mp4"/>
      </video>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


def make_world_map(df: pd.DataFrame, country_col: str, date_col: str, fatalities_col: str):
    """
    Interactive choropleth map (Plotly):
    - Aggregates fatalities per country
    - Lets user choose a time window from the most recent date
    """
    if df.empty:
        st.info("No data loaded yet.")
        return

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
    # Plotly expects country names; your dataset uses country names, so this is fine.
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
# UI Header
# =========================
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

# =========================
# Sidebar (video + inputs)
# =========================
# If you have logo1.mp4 in your repo, this will show it at the top of the sidebar.
sidebar_video("logo1.mp4", height_px=170)

st.sidebar.header("Inputs")

# Demo checkbox starts UNCHECKED
use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help="Loads a small CSV from the repo named 'ukraine_sample.csv'.",
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset as a CSV.")

# Country/column inputs
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
# Data loading logic
# =========================
df_raw = None
data_source_label = None

try:
    if use_sample:
        sample_path = APP_DIR / "ukraine_sample.csv"
        if not sample_path.exists():
            st.error(
                "Built-in demo is enabled, but 'ukraine_sample.csv' is not in your repo. "
                "Add it back to GitHub (small file), then redeploy."
            )
        else:
            df_raw = read_csv_any(sample_path)
            data_source_label = "Built-in demo (ukraine_sample.csv)"
    elif uploaded is not None:
        df_raw = read_csv_any(uploaded)
        data_source_label = "Uploaded CSV"
    else:
        # No upload + no demo: load the Google Drive dataset (for the map + optional plotting)
        st.info("No upload/demo selected — loading the Google Drive dataset for the map.")
        local_path = download_gdrive_csv(GOOGLE_DRIVE_URL, CACHE_FILENAME)
        df_raw = read_csv_any(local_path)
        data_source_label = "Google Drive dataset"
except Exception as e:
    st.error(f"Data loading error: {e}")

if df_raw is not None and data_source_label:
    st.caption(f"Data source: **{data_source_label}**")

# Clean/coerce (so map + plot both work)
df = None
if df_raw is not None:
    try:
        df = coerce_columns(df_raw, country_col, date_col, fatalities_col)
    except Exception as e:
        st.error(f"Column/parsing error: {e}")
        df = None

# =========================
# Placeholder content (map) beneath the banner when no plot yet
# =========================
# Show the interactive map ALWAYS when data exists; it fills the empty space nicely.
if df is not None:
    make_world_map(df, country_col=country_col, date_col=date_col, fatalities_col=fatalities_col)
else:
    st.info("Load data (upload, demo, or Google Drive) to display the interactive map here.")

# =========================
# Plot section (only when user clicks Generate plot)
# =========================
if run_btn:
    if df is None:
        st.warning("No usable dataset loaded yet.")
        st.stop()

    try:
        thresholds = parse_thresholds(thresholds_raw)

        daily = daily_series_for_country(
            df,
            country_name=country_name,
            country_col=country_col,
            date_col=date_col,
            fatalities_col=fatalities_col,
        )

        # Build detected frames for each threshold
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

        # Merge rolling series (same dates) from first threshold frame
        base = detected[0][["date", "fatalities", "rolling"]].copy()

        # Plot
        col1, col2 = st.columns([2, 1], vertical_alignment="top")

        with col1:
            fig = plt.figure(figsize=(12, 5))
            plt.plot(base["date"], base["rolling"], label="Rolling fatalities")

            # draw each threshold + markers
            for dth in detected:
                th = dth["threshold"].iloc[0]
                plt.axhline(y=th, linestyle="--", label=f"Threshold {th:g}")

                starts = dth[dth["escalation_start"]]
                if not starts.empty:
                    plt.scatter(starts["date"], starts["rolling"], s=60, label=f"Escalation starts (≥{th:g})")

            plt.title(
                f"AEGIS Escalation Detection — {country_name} "
                f"(rolling={int(rolling_window)}d, persistence={int(persistence)}d)"
            )
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
