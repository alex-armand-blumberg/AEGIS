import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AEGIS Escalation Detection", layout="wide")

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "ukraine_sample.csv"   # must exist in your repo
SIDEBAR_VIDEO = APP_DIR / "logo1.mp4"         # put logo1.mp4 in your repo root (same folder as app.py)


# -----------------------------
# Small helpers
# -----------------------------
def parse_thresholds(raw: str) -> list[float]:
    parts = [p.strip() for p in (raw or "").split(",")]
    vals: list[float] = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    # keep at most 2 (your request)
    return vals[:2]


@st.cache_data(show_spinner=False)
def read_csv_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    # bytes -> file-like handled by pandas via BytesIO
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def sidebar_loop_video(mp4_path: Path, *, height_px: int = 170) -> None:
    """Embed a looping, muted, autoplaying video in the sidebar."""
    if not mp4_path.exists():
        st.sidebar.warning(f"Sidebar video not found: {mp4_path.name}")
        return

    b64 = base64.b64encode(mp4_path.read_bytes()).decode("utf-8")
    html = f"""
    <div style="margin-bottom: 0.75rem;">
      <video autoplay muted loop playsinline
             style="width: 100%; height: {height_px}px; object-fit: cover; border-radius: 14px;">
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
      </video>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


# -----------------------------
# UI: Sidebar (video + inputs)
# -----------------------------
sidebar_loop_video(SIDEBAR_VIDEO, height_px=175)

st.sidebar.header("Inputs")

use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,  # MUST start unchecked
    help="Uses the repo file: ukraine_sample.csv",
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload your dataset as a .csv file.",
    )

country_name = st.sidebar.text_input(
    "Country (exact match)",
    "Ukraine",
    help="Must match the dataset exactly (case-sensitive). Examples: Ukraine, Mexico, Syria.",
)

country_col = st.sidebar.text_input(
    "Country column",
    "country",
    help="Name of the column containing country names (e.g., country).",
)

date_col = st.sidebar.text_input(
    "Date column",
    "date_start",
    help="Name of the column containing event dates (e.g., date_start).",
)

fatalities_col = st.sidebar.text_input(
    "Fatalities column",
    "best",
    help="Name of the column containing fatalities (numeric). Example (GED): best",
)

rolling_window = st.sidebar.number_input(
    "Rolling window (days)",
    min_value=1,
    max_value=365,
    value=30,
    help="Rolling window size used to compute rolling fatalities.",
)

thresholds_raw = st.sidebar.text_input(
    "Escalation threshold(s) (comma-separated, up to 2)",
    "25,1000",
    help="Example: 25 or 25,1000. Mark escalation-start when rolling fatalities cross a threshold and persist.",
)

persistence = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
    help="How many consecutive days above the threshold are required to count as escalation.",
)

run_btn = st.sidebar.button("Generate plot")


# -----------------------------
# UI: Main header
# -----------------------------
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

if use_sample:
    st.info("Using built-in demo file: **ukraine_sample.csv**")


# -----------------------------
# Load data
# -----------------------------
if use_sample:
    if not SAMPLE_CSV.exists():
        st.error("Built-in demo file not found: **ukraine_sample.csv** (add it to the repo root).")
        st.stop()
    df = read_csv_path(str(SAMPLE_CSV))
else:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar (or enable the built-in demo), then click **Generate plot**.")
        st.stop()
    df = read_csv_bytes(uploaded.getvalue())

if not run_btn:
    st.info("When ready, click **Generate plot** in the sidebar.")
    st.stop()


# -----------------------------
# Validate inputs + prep
# -----------------------------
thresholds = parse_thresholds(thresholds_raw)
if len(thresholds) == 0:
    st.error("Please enter at least one numeric threshold (e.g., 25 or 25,1000).")
    st.stop()

for col in [country_col, date_col, fatalities_col]:
    if col not in df.columns:
        st.error(f"Column not found in dataset: **{col}**")
        st.write("Available columns:", list(df.columns))
        st.stop()

sub = df[df[country_col] == country_name].copy()
if sub.empty:
    st.error(f"No rows found for **{country_name}** in column **{country_col}**.")
    st.stop()

# Parse/clean
sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
sub = sub.dropna(subset=[date_col])

sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

# Daily aggregate
daily = (
    sub.groupby(sub[date_col].dt.floor("D"))[fatalities_col]
    .sum()
    .reset_index()
)

daily.columns = ["date", "fatalities"]
daily = daily.sort_values("date")

daily["rolling"] = daily["fatalities"].rolling(
    window=int(rolling_window),
    min_periods=1,
).sum()


# -----------------------------
# Escalation detection (for each threshold)
# -----------------------------
starts_all = []
for thr in thresholds:
    tmp = daily.copy()
    tmp["above"] = tmp["rolling"] >= thr

    # consecutive streak counter
    tmp["streak"] = (
        tmp["above"].astype(int)
        .groupby((~tmp["above"]).cumsum())
        .cumsum()
    )

    tmp["persistent"] = tmp["streak"] >= int(persistence)
    tmp["start"] = tmp["persistent"] & (~tmp["persistent"].shift(1).fillna(False))

    starts = tmp[tmp["start"]].copy()
    starts["threshold"] = float(thr)

    starts_all.append(starts[["date", "rolling", "threshold"]])

starts = pd.concat(starts_all, ignore_index=True) if starts_all else pd.DataFrame(columns=["date", "rolling", "threshold"])


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(daily["date"], daily["rolling"])

for thr in thresholds:
    ax.axhline(thr, linestyle="--")

if not starts.empty:
    ax.scatter(starts["date"], starts["rolling"], s=70)

ax.set_title(
    f"AEGIS Escalation Detection — {country_name} (rolling={int(rolling_window)}d, thresholds={thresholds})"
)
ax.set_xlabel("Date")
ax.set_ylabel("Rolling window fatalities")

st.pyplot(fig, clear_figure=True)


# -----------------------------
# Summary
# -----------------------------
col_plot, col_summary = st.columns([3, 1], vertical_alignment="top")

with col_summary:
    st.subheader("Summary")
    st.write(f"Rows (daily): **{len(daily)}**")

    for thr in thresholds:
        days_above = int((daily["rolling"] >= thr).sum())
        st.write(f"Days above {thr}: **{days_above}**")

    # persistent-days per threshold (informational)
    st.write(f"Persistence: **{int(persistence)} days**")

    st.write(f"Escalation starts detected: **{len(starts)}**")

    if not starts.empty:
        st.write("First escalation starts:")
        st.dataframe(starts.sort_values("date").head(10), use_container_width=True)
