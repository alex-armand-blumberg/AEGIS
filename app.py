import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="AEGIS Escalation Detection", layout="wide")

logo = "logo.png"  # put your logo file in the repo

col1, col2 = st.columns([1,6])

with col1:
    st.image(logo, width=80)

with col2:
    st.title("AEGIS — Escalation Detection Demo")
    
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

APP_DIR = Path(__file__).resolve().parent


# -----------------------------
# Sidebar Inputs
# -----------------------------

st.sidebar.header("Inputs")

# Demo checkbox starts UNCHECKED
use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help="Data from the UCDP Georeferenced Event Dataset (GED): https://ucdp.uu.se/"
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload Your Conflict Data Here")

country_name = st.sidebar.text_input(
    "Country",
    "Ukraine",
    help="Name must match the dataset exactly (e.g., 'Ukraine', 'Mexico', 'Syria')."
)

country_col = st.sidebar.text_input(
    "Name of Country Column",
    "country",
    help="Name of your dataset's column with countries listed."
)

date_col = st.sidebar.text_input(
    "Name of Date Column",
    "date_start",
    help="Name of your dataset's column with dates listed."
)

fatalities_col = st.sidebar.text_input(
    "Name of Fatalities Column",
    "best",
    help="Name of your dataset's column with number of fatalities."
)

rolling_window = st.sidebar.number_input(
    "Rolling window (days)", min_value=1, max_value=365, value=30
)

# Two thresholds allowed
thresholds_raw = st.sidebar.text_input(
    "Escalation Thresholds",
    "25,1000",
    help="For multiple thresholds, separate values with a comma."
)

persistence = st.sidebar.number_input(
    "Persistence (days above threshold)", min_value=1, value=7
)

run_btn = st.sidebar.button("Generate Plot")


# -----------------------------
# Helper functions
# -----------------------------

def parse_thresholds(raw):
    parts = raw.split(",")
    vals = []

    for p in parts:
        p = p.strip()
        if p != "":
            vals.append(float(p))

    return vals


def load_sample():
    path = APP_DIR / "ukraine_sample.csv"

    if not path.exists():
        st.error("ukraine_sample.csv not found in repo.")
        st.stop()

    return pd.read_csv(path)


# -----------------------------
# Load data
# -----------------------------

if use_sample:
    st.info("Using built-in demo file: ukraine_sample.csv")
    df = load_sample()

else:
    if uploaded is None:
        st.info("Upload a CSV or enable the demo.")
        st.stop()

    df = pd.read_csv(uploaded)


if not run_btn:
    st.info("Click **Generate plot** when ready.")
    st.stop()


# -----------------------------
# Prepare data
# -----------------------------

try:

    if country_col not in df.columns:
        st.error(f"{country_col} not found in dataset")
        st.stop()

    if date_col not in df.columns:
        st.error(f"{date_col} not found in dataset")
        st.stop()

    if fatalities_col not in df.columns:
        st.error(f"{fatalities_col} not found in dataset")
        st.stop()

    sub = df[df[country_col] == country_name].copy()

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    daily = (
        sub.groupby(sub[date_col].dt.floor("D"))[fatalities_col]
        .sum()
        .reset_index()
    )

    daily.columns = ["date", "fatalities"]
    daily = daily.sort_values("date")

    daily["rolling"] = daily["fatalities"].rolling(
        window=int(rolling_window), min_periods=1
    ).sum()

    thresholds = parse_thresholds(thresholds_raw)

except Exception as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# Escalation Detection
# -----------------------------

starts_all = []

for threshold in thresholds:

    daily["above"] = daily["rolling"] >= threshold

    daily["streak"] = (
        daily["above"].astype(int)
        .groupby((~daily["above"]).cumsum())
        .cumsum()
    )

    daily["persistent"] = daily["streak"] >= persistence

    daily["start"] = daily["persistent"] & (~daily["persistent"].shift(1).fillna(False))

    starts = daily[daily["start"]].copy()
    starts["threshold"] = threshold

    starts_all.append(starts)


starts = pd.concat(starts_all)


# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(daily["date"], daily["rolling"])

for t in thresholds:
    ax.axhline(t, linestyle="--")

ax.scatter(starts["date"], starts["rolling"], s=80)

ax.set_title(
    f"AEGIS Escalation Detection — {country_name} (rolling={rolling_window}d, thresholds={thresholds})"
)

ax.set_xlabel("Date")
ax.set_ylabel("Rolling fatalities")

st.pyplot(fig)


# -----------------------------
# Summary
# -----------------------------

col1, col2 = st.columns([3,1])

with col2:

    st.subheader("Summary")

    st.write(f"Rows (daily): {len(daily)}")

    for t in thresholds:

        days_above = (daily["rolling"] >= t).sum()

        st.write(f"Days above threshold {t}:      {days_above}")

    st.write(f"Escalation starts detected: {len(starts)}")

    if not starts.empty:

        st.write("First escalation starts:")

        st.dataframe(
            starts[["date","rolling","threshold"]].head(10)
        )
