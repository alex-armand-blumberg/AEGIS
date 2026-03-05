import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AEGIS", layout="wide")
st.title("AEGIS — Escalation Detection Demo")
st.write("Upload a dataset (CSV) and choose a country to generate the rolling 30-day fatalities plot.")

# ---- Sidebar controls ----
st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

country = st.sidebar.text_input("Country (exact match)", value="Ukraine")
date_col = st.sidebar.text_input("Date column", value="date_start")
fatalities_col = st.sidebar.text_input("Fatalities column", value="best")

rolling_days = st.sidebar.number_input("Rolling window (days)", min_value=1, max_value=365, value=30)
threshold = st.sidebar.number_input("Escalation threshold", min_value=1, max_value=1000000, value=25)
persistence_days = st.sidebar.number_input("Persistence (days above threshold to count as escalation)", min_value=1, max_value=60, value=3)

run = st.sidebar.button("Generate plot")

def load_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if fatalities_col not in df.columns:
        raise ValueError(f"Missing fatalities column: {fatalities_col}")
    if "country" not in df.columns:
        raise ValueError("Missing required column: country")

    sub = df[df["country"] == country].copy()
    if sub.empty:
        raise ValueError(f"No rows found for country='{country}'")

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])

    sub[fatalities_col] = pd.to_numeric(sub[fatalities_col], errors="coerce").fillna(0)

    # aggregate to daily totals
    daily = (
        sub.groupby(sub[date_col].dt.date, as_index=False)[fatalities_col]
        .sum()
        .rename(columns={fatalities_col: "fatalities"})
    )
    daily["date"] = pd.to_datetime(daily[date_col]) if date_col in daily.columns else pd.to_datetime(daily.iloc[:,0])
    daily = daily.sort_values("date")

    # rolling
    daily["rolling"] = daily["fatalities"].rolling(int(rolling_days), min_periods=int(rolling_days)).sum()

    # escalation + persistence
    daily["above"] = daily["rolling"] >= threshold
    daily["persistent"] = daily["above"].rolling(int(persistence_days), min_periods=int(persistence_days)).sum() == int(persistence_days)
    daily["escalation_start"] = daily["persistent"] & (~daily["persistent"].shift(1).fillna(False))

    return daily

if uploaded and run:
    try:
        df = pd.read_csv(uploaded)
        daily = load_and_prepare(df)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = plt.figure(figsize=(12, 5))
            plt.plot(daily["date"], daily["rolling"])
            plt.axhline(y=threshold, linestyle="--")
            starts = daily[daily["escalation_start"]]
            if not starts.empty:
                plt.scatter(starts["date"], starts["rolling"], s=60)
            plt.title(f"AEGIS Escalation Detection — {country} (rolling={rolling_days}d, threshold={threshold})")
            plt.xlabel("Date")
            plt.ylabel("Rolling window fatalities")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Summary")
            st.write(f"Rows (daily): **{len(daily)}**")
            st.write(f"Days above threshold: **{int(daily['above'].sum())}**")
            st.write(f"Days persistent: **{int(daily['persistent'].sum())}**")
            st.write(f"Escalation starts detected: **{int(daily['escalation_start'].sum())}**")

            if not starts.empty:
                st.write("First escalation starts:")
                st.dataframe(starts[["date", "rolling"]].head(10))

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload a CSV in the sidebar, then click **Generate plot**.")
