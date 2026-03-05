# app.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_FILENAME = "ukraine_sample.csv"

st.set_page_config(
    page_title="AEGIS — Escalation Detection Demo",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def parse_thresholds(raw: str) -> List[float]:
    """
    Parse comma-separated thresholds like:
    "25" or "25,1000" (spaces allowed)
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    return out


def consecutive_true_streak(mask: pd.Series) -> pd.Series:
    """
    For a boolean series, return the length of the current consecutive True streak at each row.
    Example: [F,T,T,F,T] -> [0,1,2,0,1]
    """
    mask = mask.fillna(False).astype(bool)
    grp = (~mask).cumsum()
    streak = mask.groupby(grp).cumcount() + 1
    streak = streak.where(mask, 0)
    return streak


@dataclass
class EscalationResult:
    df_daily: pd.DataFrame  # columns: date, fatalities, rolling
    start_events: pd.DataFrame  # columns: date, rolling, threshold
    days_above: int
    days_persistent: int
    num_starts: int


def validate_columns(df: pd.DataFrame, needed: List[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Found columns: {list(df.columns)[:50]}...")


def load_csv_from_repo(sample_filename: str) -> pd.DataFrame:
    p = (APP_DIR / sample_filename).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Built-in sample not found at {sample_filename}.\n"
            f"Put '{sample_filename}' in the same folder as app.py in your repo."
        )
    # Guard against empty files
    if p.stat().st_size == 0:
        raise ValueError(f"Sample file '{sample_filename}' is empty (0 bytes).")
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    # Pandas can read bytes via BytesIO
    from io import BytesIO

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    return pd.read_csv(BytesIO(file_bytes))


def build_daily_series(
    df: pd.DataFrame,
    country_col: str,
    country: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
    validate_columns(df, [country_col, date_col, fatalities_col])

    # Filter country
    sub = df[df[country_col].astype(str) == str(country)].copy()
    if sub.empty:
        raise ValueError(
            f"No rows matched country='{country}' in column '{country_col}'. "
            f"Tip: make sure capitalization/spelling matches exactly."
        )

    # Parse date + fatalities robustly
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])
    if sub.empty:
        raise ValueError(
            f"After parsing '{date_col}' as datetime, no valid dates remained."
        )

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # Aggregate to daily totals
    daily = (
    sub.groupby(sub[date_col].dt.floor("D"))[fatalities_col]
    .sum()
    .reset_index()
)

daily.columns = ["date", "fatalities"]
daily = daily.sort_values("date")

    # Fill missing days with 0 (continuous daily index)
full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = daily.set_index("date").reindex(full_range).fillna({"fatalities": 0.0})
daily.index.name = "date"
daily = daily.reset_index()

return daily


def detect_escalation(
    daily: pd.DataFrame,
    rolling_window_days: int,
    thresholds: List[float],
    persistence_days: int,
) -> EscalationResult:
    if rolling_window_days < 1:
        raise ValueError("Rolling window must be >= 1 day.")
    if persistence_days < 1:
        raise ValueError("Persistence must be >= 1 day.")
    if not thresholds:
        raise ValueError("Provide at least one threshold (e.g., 25 or 25,1000).")

    df_daily = daily.copy()
    df_daily["rolling"] = (
        df_daily["fatalities"]
        .rolling(window=int(rolling_window_days), min_periods=1)
        .sum()
    )

    # Union counts (for the first threshold only, used in the summary)
    base_thr = float(thresholds[0])
    above = df_daily["rolling"] >= base_thr
    days_above = int(above.sum())

    streak = consecutive_true_streak(above)
    persistent = above & (streak >= int(persistence_days))
    days_persistent = int(persistent.sum())

    # Start events for each threshold separately
    starts_all: List[pd.DataFrame] = []
    for thr in thresholds[:2]:  # keep demo clean
        thr = float(thr)
        above_t = df_daily["rolling"] >= thr
        streak_t = consecutive_true_streak(above_t)
        persistent_t = above_t & (streak_t >= int(persistence_days))
        start_t = persistent_t & ~persistent_t.shift(1, fill_value=False)

        starts = df_daily.loc[start_t, ["date", "rolling"]].copy()
        starts["threshold"] = thr
        starts_all.append(starts)

    start_events = pd.concat(starts_all, ignore_index=True) if starts_all else pd.DataFrame(
        columns=["date", "rolling", "threshold"]
    )
    start_events = start_events.sort_values(["date", "threshold"]).reset_index(drop=True)

    return EscalationResult(
        df_daily=df_daily,
        start_events=start_events,
        days_above=days_above,
        days_persistent=days_persistent,
        num_starts=int(len(start_events)),
    )


def plot_result(
    result: EscalationResult,
    country: str,
    rolling_window_days: int,
    thresholds: List[float],
) -> plt.Figure:
    dfp = result.df_daily
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(111)

    ax.plot(dfp["date"], dfp["rolling"], linewidth=2)

    # Threshold lines
    for thr in thresholds[:2]:
        ax.axhline(y=float(thr), linestyle="--", linewidth=1)

    # Start markers
    if not result.start_events.empty:
        ax.scatter(
            result.start_events["date"],
            result.start_events["rolling"],
            s=70,
            zorder=3,
        )

    ax.set_title(
        f"AEGIS Escalation Detection — {country} (rolling={rolling_window_days}d, thresholds={','.join(str(t) for t in thresholds[:2])})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling window fatalities")
    fig.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)

with st.sidebar:
    st.header("Inputs")

    use_sample = st.checkbox(
        "Use built-in Ukraine example (recommended demo)",
        value=True,
    )

    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            accept_multiple_files=False,
        )

    # You asked to remove sample-path input.
    sample_filename = DEFAULT_SAMPLE_FILENAME

    country_col = st.text_input("Country column", value="country")
    country = st.text_input("Country (exact match)", value="Ukraine")
    date_col = st.text_input("Date column", value="date_start")
    fatalities_col = st.text_input("Fatalities column", value="best")

    rolling_window_days = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
    )

    thresholds_raw = st.text_input(
        "Escalation threshold(s) (comma-separated)",
        value="25",
        help='Examples: "25" or "25,1000"',
    )

    persistence_days = st.number_input(
        "Persistence (consecutive days above threshold)",
        min_value=1,
        max_value=60,
        value=3,
        step=1,
    )

    run_btn = st.button("Generate plot")


# Main logic + messaging
if use_sample:
    st.info(f"Using built-in demo file: **{sample_filename}**")
else:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, then click **Generate plot**.")
        st.stop()

if not run_btn:
    st.info("Click **Generate plot** when you're ready.")
    st.stop()

# Load data
try:
    if use_sample:
        df = load_csv_from_repo(sample_filename)
    else:
        df = read_csv_bytes(uploaded.getvalue())

    thresholds = parse_thresholds(thresholds_raw)
    if len(thresholds) > 2:
        st.warning("You entered more than 2 thresholds. For the demo, only the first 2 will be used.")
        thresholds = thresholds[:2]

    daily = build_daily_series(
        df=df,
        country_col=country_col,
        country=country,
        date_col=date_col,
        fatalities_col=fatalities_col,
    )

    result = detect_escalation(
        daily=daily,
        rolling_window_days=int(rolling_window_days),
        thresholds=thresholds,
        persistence_days=int(persistence_days),
    )

    fig = plot_result(
        result=result,
        country=country,
        rolling_window_days=int(rolling_window_days),
        thresholds=thresholds,
    )

except Exception as e:
    st.error(str(e))
    st.stop()

# Layout: plot + summary
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Summary")
    st.write(f"Rows (daily): **{len(result.df_daily)}**")
    st.write(f"Threshold (primary): **{float(thresholds[0])}**")
    st.write(f"Days above threshold: **{result.days_above}**")
    st.write(f"Days persistent: **{result.days_persistent}**")
    st.write(f"Escalation starts detected: **{result.num_starts}**")

    st.markdown("**First escalation starts:**")
    if result.start_events.empty:
        st.write("_None detected_")
    else:
        st.dataframe(result.start_events.head(20), use_container_width=True)

    st.caption(
        "Note: this is a demo heuristic (rolling fatalities + threshold + persistence). "
        "It detects historical escalation patterns; it does not forecast future escalation by itself."
    )
