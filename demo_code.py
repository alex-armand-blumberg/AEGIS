import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================

THRESHOLD = 25
PERSISTENCE_DAYS = 3
COOLDOWN_DAYS = 30

np.random.seed(42)

# =========================
# 1. GENERATE SYNTHETIC DATA
# =========================

days = pd.date_range(start="2023-01-01", periods=365)

fatalities = np.random.poisson(lam=1.2, size=365)

# Inject escalation spike
fatalities[200:230] += 6

df = pd.DataFrame({
    "date": days,
    "fatalities": fatalities
})

# =========================
# 2. ROLLING 30-DAY SUM
# =========================

df["rolling_30"] = (
    df["fatalities"]
    .rolling(window=30, min_periods=30)
    .sum()
)

df["above_threshold"] = df["rolling_30"] >= THRESHOLD

# =========================
# 3. PERSISTENCE LOGIC
# =========================

df["above_run"] = (
    df["above_threshold"]
    .astype(int)
    .groupby((df["above_threshold"] != df["above_threshold"].shift()).cumsum())
    .cumsum()
)

df["persistent"] = df["above_threshold"] & (df["above_run"] >= PERSISTENCE_DAYS)

# =========================
# 4. ESCALATION START
# =========================

df["escalation_start"] = (
    df["persistent"] &
    (~df["persistent"].shift(1).fillna(False))
)

# =========================
# 5. SUMMARY
# =========================

print("Days above threshold:", int(df["above_threshold"].sum()))
print("Days persistent:", int(df["persistent"].sum()))
print("Escalation starts detected:", int(df["escalation_start"].sum()))

# =========================
# 6. VISUALIZATION
# =========================

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["rolling_30"], linewidth=2)

plt.axhline(y=THRESHOLD, linestyle="--")

starts = df[df["escalation_start"]]
if len(starts) > 0:
    plt.scatter(starts["date"], starts["rolling_30"], s=80)

plt.title("AEGIS Synthetic Escalation Detection")
plt.xlabel("Date")
plt.ylabel("Rolling 30-Day Fatalities")
plt.tight_layout()

plt.savefig("ukraine_escalation_plot.png", dpi=300, bbox_inches="tight")

plt.show()
