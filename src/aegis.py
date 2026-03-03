import pandas as pd

def load_data(csv_path: str, date_col: str, fatalities_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'. Found: {list(df.columns)[:20]}...")
    if fatalities_col not in df.columns:
        raise ValueError(f"Missing fatalities column '{fatalities_col}'. Found: {list(df.columns)[:20]}...")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[fatalities_col] = pd.to_numeric(df[fatalities_col], errors="coerce").fillna(0)
    return df

def aggregate_daily(df: pd.DataFrame, date_col: str, fatalities_col: str, group_by=None) -> pd.DataFrame:
    if group_by is None:
        group_by = []

    cols = group_by + [date_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing grouping/date columns: {missing}")

    daily = (
        df.groupby(cols, as_index=False)[fatalities_col]
          .sum()
          .rename(columns={date_col: "date", fatalities_col: "fatalities"})
          .sort_values(cols)
    )
    return daily

def compute_rolling(daily: pd.DataFrame, window_days: int = 30, group_by=None) -> pd.DataFrame:
    if group_by is None:
        group_by = []
    daily = daily.copy()
    if group_by:
        daily["rolling_30"] = (
            daily.groupby(group_by)["fatalities"]
                 .rolling(window=window_days, min_periods=window_days)
                 .sum()
                 .reset_index(level=group_by, drop=True)
        )
    else:
        daily["rolling_30"] = daily["fatalities"].rolling(window=window_days, min_periods=window_days).sum()
    return daily

def detect_escalations(daily: pd.DataFrame, threshold: float, persistence_days: int, cooldown_days: int, group_by=None) -> pd.DataFrame:
    if group_by is None:
        group_by = []
    out = daily.copy()
    out["above"] = out["rolling_30"] >= threshold

    def per_group(g):
        g = g.sort_values("date").copy()

        # persistence requirement: above threshold for N consecutive days
        g["above_run"] = g["above"].astype(int).groupby((g["above"] != g["above"].shift()).cumsum()).cumsum()
        g["persistent"] = g["above"] & (g["above_run"] >= persistence_days)

        # cooldown: must be below for cooldown_days before a new start
        g["below"] = ~g["above"]
        g["below_run"] = g["below"].astype(int).groupby((g["below"] != g["below"].shift()).cumsum()).cumsum()
        g["cooldown_met"] = g["below"] & (g["below_run"] >= cooldown_days)

        # escalation starts when persistent becomes True AND previous day wasn't persistent
        g["escalation_start"] = g["persistent"] & (~g["persistent"].shift(1).fillna(False))

        # optional: suppress starts unless cooldown met before the start
        # We enforce that the last cooldown_met was True sometime before the start (or it's the first start)
        last_cooldown = False
        starts = []
        for i, row in g.iterrows():
            if row["cooldown_met"]:
                last_cooldown = True
            if row["escalation_start"]:
                # allow if cooldown met previously OR this is the first ever start (no previous escalation)
                if last_cooldown or (not any(starts)):
                    starts.append(True)
                    last_cooldown = False  # reset after an escalation start
                else:
                    starts.append(False)
            else:
                starts.append(False)
        g["escalation_start"] = starts

        g["escalation_flag"] = g["persistent"]
        return g

    if group_by:
        out = out.groupby(group_by, group_keys=False).apply(per_group)
    else:
        out = per_group(out)

    return out

def summarize(out: pd.DataFrame) -> dict:
    return {
        "days_above_threshold": int(out["above"].sum()),
        "days_persistent": int(out["persistent"].sum()),
        "escalation_starts": int(out["escalation_start"].sum()),
    }
