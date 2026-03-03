THRESHOLD_DEATHS_30D = 25      # escalation threshold
PERSISTENCE_DAYS = 3           # must stay above threshold this many days
COOLDOWN_DAYS = 30             # must be below threshold this many days before a new escalation can start
DATE_COL = "date_start"        # UCDP GED uses date_start/date_end
FATALITIES_COL = "best"        # UCDP GED uses best/high/low
GROUP_BY = ["country"]         # can change to ["country","adm_1"] later
