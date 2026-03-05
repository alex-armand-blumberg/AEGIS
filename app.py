import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="AEGIS Escalation Detection", layout="wide")

st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

APP_DIR = Path(__file__).resolve().parent


# -----------------------------
# Sidebar inputs
# -----------------------------

st.sidebar.header("Inputs")

use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)", value=True
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

country_col = st.sidebar.text_input("Country column", "country")
country_name = st.sidebar.text_input("Country (exact match)", "Ukraine")

date_col = st.sidebar.text_input("Date column", "date_start")
fatalities_col = st.sidebar.text_input("Fatalities column", "best")

rolling_window = st.sidebar.number_input(
    "Rolling window (days)", min_value=1, max_value=365, value=30
)

threshold = st.sidebar.number_input(
    "Escalation threshold", min_value=1, value=25
)

persistence = st.sidebar.number_input(
    "Persistence (days above threshold)", min_value=1, value=3
)

run_btn = st.sidebar.button("Generate plot")


# -----------------------------
# Load data
# -----------------------------

def load_sample():
    sample_path = APP_DIR / "ukraine_sample.csv"

    if not sample_path.exists():
        st.error(
            "Built-in demo file `ukraine_sample.csv` not found. "
            "Upload it to the repo next to `app.py`."
        )
        st.stop()

    return pd.read_csv(sample_path)


if use_sample:
    st.info("Using built-in demo file: ukraine_sample.csv")
    df = load_sample()

else:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, then click Generate plot.")
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
        st.error(f"Column `{country_col}` not found in dataset.")
        st.stop()

    if date_col not in df.columns:
        st.error(f"Column `{date_col}` not found in dataset.")
        st.stop()

    if fatalities_col not in df.columns:
        st.error(f"Column `{fatalities_col}` not found in dataset.")
        st.stop()

    sub = df[df[country_col] == country_name].copy()

    if sub.empty:
        st.error(f"No rows found for country `{country_name}`.")
        st.stop()

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # aggregate daily fatalities
    daily = (
        sub.groupby(sub[date_col].dt.floor("D"))[fatalities_col]
        .sum()
        .reset_index()
    )

    daily.columns = ["date", "fatalities"]
    daily = daily.sort_values("date")

    # rolling fatalities
    daily["rolling"] = daily["fatalities"].rolling(
        window=int(rolling_window), min_periods=1
    ).sum()

    # threshold detection
    daily["above"] = daily["rolling"] >= threshold

    daily["streak"] = (
        daily["above"].astype(int)
        .groupby((~daily["above"]).cumsum())
        .cumsum()
    )

    daily["persistent"] = daily["streak"] >= persistence

    daily["start"] = daily["persistent"] & (~daily["persistent"].shift(1).fillna(False))

    starts = daily[daily["start"]]

except Exception as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(daily["date"], daily["rolling"])

ax.axhline(threshold, linestyle="--")

if not starts.empty:
    ax.scatter(starts["date"], starts["rolling"], s=80)

ax.set_title(
    f"AEGIS Escalation Detection — {country_name} (rolling={rolling_window}d, threshold={threshold})"
)

ax.set_xlabel("Date")
ax.set_ylabel("Rolling fatalities")

st.pyplot(fig)


# -----------------------------
# Summary
# -----------------------------

col1, col2 = st.columns([3, 1])

with col2:

    st.subheader("Summary")

    st.write(f"Rows (daily): {len(daily)}")

    days_above = daily["above"].sum()
    days_persistent = daily["persistent"].sum()

    st.write(f"Days above threshold: {days_above}")
    st.write(f"Days persistent: {days_persistent}")

    st.write(f"Escalation starts detected: {len(starts)}")

    if not starts.empty:

        st.write("First escalation starts:")

        show = starts[["date", "rolling"]].head(10)
        st.dataframe(show)
