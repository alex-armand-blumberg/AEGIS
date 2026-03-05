import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AEGIS", layout="wide")
st.title("AEGIS — Escalation Detection Demo")
st.write(
    "Upload a dataset (CSV) and choose a country to generate the rolling fatalities plot and escalation-start markers."
)


# -----------------------------
# Helpers
# -----------------------------
def parse_thresholds(text: str) -> list[float]:
    """
    Accepts:
      "25"
      "25,1000"
      "25 1000"
    Returns a sorted unique list of floats.
    """
    if text is None:
        return []
    parts = (
        text.replace(";", ",")
        .replace("|", ",")
        .replace(" ", ",")
        .split(",")
    )
    vals: list[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            # ignore garbage tokens
            pass
    # unique, stable order
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_csv_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_data(
    uploaded_file,
    use_sample: bool,
    sample_path: str,
) -> pd.DataFrame | None:
    if use_sample:
        p = Path(sample_path)
        if not p.exists():
            st.error(
                f"Built-in sample not found at `{sample_path}`.\n\n"
                "Fix: add a small sample file to your repo (recommended: `sample_ukraine.csv`) "
                "and redeploy, or turn off the sample toggle and upload a CSV."
            )
            return None
        return read_csv_path(str(p))

    if uploaded_file is None:
        return None

    return read_csv_bytes(uploaded_file.getvalue())


def prepare_daily_series(
    df: pd.DataFrame,
    country_col: str,
    country_value: str,
    date_col: str,
    fatalities_col: str,
) -> pd.DataFrame:
    # Basic column validation
    missing = [c for c in [country_col, date_col, fatalities_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s): {missing}. Available columns include: {list(df.columns)[:30]}..."
        )

    sub = df[df[country_col].astype(str) == str(country_value)].copy()

    if sub.empty:
        raise ValueError(
            f"No rows match {country_col} == '{country_value}'. "
            "Check spelling/case or choose a different country value."
        )

    # Parse date + fatalities
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    sub = sub.dropna(subset=[date_col])
    if sub.empty:
        raise ValueError(
            f"After parsing `{date_col}`, no valid dates remained. "
            "Try a different date column (e.g., date_start)."
        )

    # Aggregate to daily totals
    sub["date_day"] = sub[date_col].dt.floor("D")
    daily = (
        sub.groupby("date_day", as_index=False)[fatalities_col]
        .sum()
        .rename(columns={"date_day": "date", fatalities_col: "fatalities"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


def add_rolling_and_escalations(
    daily: pd.DataFrame,
    rolling_days: int,
    thresholds: list[float],
    persistence_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = daily.copy()

    # Rolling window: rolling_days of DAILY totals
    # (Because df is daily, a window of N = N days.)
    df["rolling"] = df["fatalities"].rolling(window=rolling_days, min_periods=1).sum()

    # Detect escalation starts for each threshold
    starts_rows = []
    for thr in thresholds:
        above = df["rolling"] >= thr

        # consecutive days above threshold
        consec = above.astype(int).groupby((~above).cumsum()).cumsum()

        persistent = consec >= persistence_days
        start = persistent & (~persistent.shift(1).fillna(False))

        start_df = df.loc[start, ["date", "rolling"]].copy()
        start_df["threshold"] = thr
        starts_rows.append(start_df)

        # store helpful columns for summary (use first threshold as default summary)
        # (We keep only for the first threshold to avoid clutter.)
        if thr == thresholds[0]:
            df["above_threshold"] = above
            df["consecutive_above"] = consec
            df["persistent"] = persistent
            df["escalation_start"] = start

    starts = pd.concat(starts_rows, ignore_index=True) if starts_rows else pd.DataFrame(
        columns=["date", "rolling", "threshold"]
    )

    return df, starts


def make_plot(
    df: pd.DataFrame,
    country_value: str,
    rolling_days: int,
    thresholds: list[float],
    starts: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["rolling"])

    # Threshold lines
    for thr in thresholds:
        plt.axhline(thr, linestyle="--")

    # Mark escalation starts (color per threshold is default Matplotlib cycling)
    if not starts.empty:
        for thr in thresholds:
            s = starts[starts["threshold"] == thr]
            if not s.empty:
                plt.scatter(s["date"], s["rolling"], s=80)

    plt.title(
        f"AEGIS Escalation Detection — {country_value} "
        f"(rolling={rolling_days}d, thresholds={','.join(str(int(t)) if float(t).is_integer() else str(t) for t in thresholds)})"
    )
    plt.xlabel("Date")
    plt.ylabel("Rolling window fatalities")
    plt.tight_layout()
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Inputs")

use_sample = st.sidebar.checkbox(
    "Use built-in Ukraine example (recommended demo)",
    value=False,
    help=(
        "Uses a small file included with your repo (default path: sample_ukraine.csv). "
        "Turn this off to upload your own CSV."
    ),
)

sample_path = st.sidebar.text_input(
    "Built-in sample path",
    value="sample_ukraine.csv",
    help="If using the built-in demo, this file must exist in your repo.",
)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

country_col = st.sidebar.text_input("Country column", value="country")
country_value = st.sidebar.text_input("Country (exact match)", value="Ukraine")
date_col = st.sidebar.text_input("Date column", value="date_start")
fatalities_col = st.sidebar.text_input("Fatalities column", value="best")

rolling_days = st.sidebar.number_input("Rolling window (days)", min_value=1, max_value=365, value=30, step=1)
thresholds_raw = st.sidebar.text_input("Escalation threshold(s) (comma-separated)", value="25")
persistence_days = st.sidebar.number_input(
    "Persistence (consecutive days above threshold)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
)

run_btn = st.sidebar.button("Generate plot")


# -----------------------------
# Main logic
# -----------------------------
df = load_data(uploaded, use_sample=use_sample, sample_path=sample_path)

if df is None:
    st.info("Upload a CSV in the sidebar (or enable the built-in demo), then click **Generate plot**.")
    st.stop()

if not run_btn:
    st.info("CSV loaded — now click **Generate plot**.")
    st.stop()

try:
    thresholds = parse_thresholds(thresholds_raw)
    if not thresholds:
        st.error("Please enter at least one valid threshold (e.g., `25` or `25,1000`).")
        st.stop()
    thresholds = thresholds[:2]  # keep demo simple
    if len(parse_thresholds(thresholds_raw)) > 2:
        st.warning("You entered more than 2 thresholds. For the demo, only the first 2 are used.")

    daily = prepare_daily_series(
        df=df,
        country_col=country_col,
        country_value=country_value,
        date_col=date_col,
        fatalities_col=fatalities_col,
    )

    df_roll, starts = add_rolling_and_escalations(
        daily=daily,
        rolling_days=int(rolling_days),
        thresholds=thresholds,
        persistence_days=int(persistence_days),
    )

    fig = make_plot(
        df=df_roll,
        country_value=country_value,
        rolling_days=int(rolling_days),
        thresholds=thresholds,
        starts=starts,
    )

    # Layout: plot + summary
    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Summary")

        st.write(f"Rows (daily): **{len(df_roll)}**")
        st.write(f"Thresholds: **{', '.join(str(t) for t in thresholds)}**")
        st.write(f"Rolling window: **{int(rolling_days)} days**")
        st.write(f"Persistence: **{int(persistence_days)} days**")

        # Summary based on first threshold
        first_thr = thresholds[0]
        above = (df_roll["rolling"] >= first_thr)
        consec = above.astype(int).groupby((~above).cumsum()).cumsum()
        persistent = consec >= int(persistence_days)
        starts_first = (persistent & (~persistent.shift(1).fillna(False)))

        st.write(f"Days above first threshold: **{int(above.sum())}**")
        st.write(f"Days persistent: **{int(persistent.sum())}**")
        st.write(f"Escalation starts detected: **{int(starts_first.sum())}**")

        st.write("First escalation starts:")
        if starts.empty:
            st.info("No escalation starts detected with the current settings.")
        else:
            # show first 10 across thresholds
            show = starts.sort_values(["date", "threshold"]).head(10).copy()
            show["date"] = pd.to_datetime(show["date"])
            st.dataframe(show, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
