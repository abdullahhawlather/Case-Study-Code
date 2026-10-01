"""Seasonal index (ratio-to-average method) of the demand series.

Paper link: Section 2.1 - Table 4 (monthly seasonal factors) and Figure 2 (seasonal profile).
Output: results/seasonal_factors.json, figures/fig02_seasonal_factors.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import FIG_DIR, load_history, save_json

hist = load_history()
hist["month"] = pd.to_datetime(hist["Date"]).dt.month
overall = hist["Demand"].mean()
table = hist.groupby("month").agg(avg_demand=("Demand", "mean"), rows=("Demand", "size"))
table["seasonal_factor"] = table["avg_demand"] / overall
table.index = pd.to_datetime(table.index, format="%m").strftime("%B")
print("overall average demand:", round(overall, 2)); print(table.round(3))
save_json({"overall_average_demand": float(overall), "rows": int(len(hist)), "table": table.round(3).to_dict("index")},
          "seasonal_factors.json")
plt.figure(figsize=(9, 4.2))
plt.plot(table.index, table["seasonal_factor"], marker="o")
plt.axhline(1.0, linestyle="--", linewidth=.8)
plt.ylabel("Seasonal factor"); plt.xticks(rotation=30, ha="right")
plt.title("Monthly seasonal factors (ratio-to-average, 777 orders)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig02_seasonal_factors.png", dpi=170); plt.close()
