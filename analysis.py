"""
Step 2: analysis.
SQL aggregates, statistical tests, a simple predictive model, and a look at
the free-text feedback. Writes charts and an Excel summary to outputs/.
Run: python analysis.py
"""
import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/clean_sales.csv", parse_dates=["order_date"])
df["feedback"] = df["feedback"].fillna("")  # blank comments are read back as NaN
con = sqlite3.connect(":memory:")
df.to_sql("sales", con, index=False)
q = lambda sql: pd.read_sql(sql, con)

results = {}

# ---------- 1. SQL aggregates ----------
monthly = q("SELECT month, ROUND(SUM(sales),2) AS revenue FROM sales GROUP BY month ORDER BY month")
store = q("""SELECT city, COUNT(*) AS orders, ROUND(SUM(sales),0) AS revenue,
             ROUND(SUM(profit),0) AS profit,
             ROUND(100.0*SUM(profit)/SUM(sales),1) AS margin_pct
             FROM sales GROUP BY city ORDER BY revenue DESC""")
category = q("""SELECT category, ROUND(SUM(sales),0) AS revenue, ROUND(SUM(profit),0) AS profit,
                ROUND(100.0*SUM(profit)/SUM(sales),1) AS margin_pct
                FROM sales GROUP BY category ORDER BY profit DESC""")
discount = q("""SELECT discount, COUNT(*) AS orders, ROUND(AVG(quantity),2) AS avg_qty,
                ROUND(AVG(sales),1) AS avg_order_value,
                ROUND(100.0*SUM(profit)/SUM(sales),1) AS margin_pct
                FROM sales GROUP BY discount ORDER BY discount""")
segment = q("""SELECT customer_segment, COUNT(*) AS orders, ROUND(SUM(sales),0) AS revenue,
               ROUND(100.0*SUM(sales)/(SELECT SUM(sales) FROM sales),1) AS revenue_share_pct
               FROM sales GROUP BY customer_segment ORDER BY revenue DESC""")

results["total_revenue"] = round(df["sales"].sum())
results["total_profit"] = round(df["profit"].sum())
results["overall_margin_pct"] = round(100 * df["profit"].sum() / df["sales"].sum(), 1)
y = df.groupby("year")["sales"].sum()
results["revenue_2023"] = round(y[2023])
results["revenue_2024"] = round(y[2024])
results["yoy_growth_pct"] = round(100 * (y[2024] / y[2023] - 1), 1)

# ---------- 2. statistical tests ----------
# a) does a discount raise quantity per order?
disc = df[df["discount"] > 0]["quantity"]
full = df[df["discount"] == 0]["quantity"]
t, p = stats.ttest_ind(disc, full, equal_var=False)
results["discount_vs_full_price"] = {
    "avg_qty_discounted": round(disc.mean(), 3),
    "avg_qty_full_price": round(full.mean(), 3),
    "welch_t": round(t, 2), "p_value": float(f"{p:.2e}")}

# b) does discount level change margin? (correlation)
r, p_r = stats.pearsonr(df["discount"], df["margin_pct"])
results["discount_margin_correlation"] = {"pearson_r": round(r, 3), "p_value": float(f"{p_r:.2e}")}

# c) do customer segments differ in order value? (one-way ANOVA)
groups = [g["sales"].values for _, g in df.groupby("customer_segment")]
f, p_a = stats.f_oneway(*groups)
results["segment_order_value_anova"] = {"F": round(f, 2), "p_value": float(f"{p_a:.2e}")}

# d) is category related to segment? (chi-square)
ct = pd.crosstab(df["customer_segment"], df["category"])
chi2, p_c, dof, _ = stats.chi2_contingency(ct)
results["segment_category_chi2"] = {"chi2": round(chi2, 2), "dof": int(dof), "p_value": round(float(p_c), 4)}

# ---------- 3. predictive model: order value ----------
feat = pd.get_dummies(df[["product", "city", "customer_segment"]], drop_first=True)
feat["discount"] = df["discount"]
# quantity is deliberately left out: sales = price * quantity * (1 - discount),
# so including it would leak the answer and inflate R2 to ~0.99.
feat["is_weekend"] = (df["order_date"].dt.weekday >= 5).astype(int)
feat["month_num"] = df["order_date"].dt.month
target = df["sales"]
Xtr, Xte, ytr, yte = train_test_split(feat, target, test_size=0.25, random_state=42)

lr = LinearRegression().fit(Xtr, ytr)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
results["model_order_value"] = {
    "linear_regression": {"R2": round(r2_score(yte, lr.predict(Xte)), 3),
                          "MAE": round(mean_absolute_error(yte, lr.predict(Xte)), 1)},
    "random_forest": {"R2": round(r2_score(yte, rf.predict(Xte)), 3),
                      "MAE": round(mean_absolute_error(yte, rf.predict(Xte)), 1)}}
imp = pd.Series(rf.feature_importances_, index=feat.columns).sort_values(ascending=False).head(8)

# ---------- 4. unstructured text: feedback ----------
POS = {"great taste", "loved it", "friendly staff", "quick service", "fresh", "will come back"}
fb = df[df["feedback"] != ""].copy()
fb["tone"] = np.where(fb["feedback"].isin(POS), "positive", "negative")
results["feedback_comments"] = len(fb)
results["feedback_negative_share_pct"] = round(100 * (fb["tone"] == "negative").mean(), 1)
neg_themes = fb[fb["tone"] == "negative"]["feedback"].value_counts()
by_city_tone = pd.crosstab(fb["city"], fb["tone"], normalize="index").round(3) * 100

# ---------- 5. simple 3-month forecast (trend + seasonality) ----------
m = df.groupby("month")["sales"].sum().reset_index()
m["t"] = np.arange(len(m))
m["mnum"] = pd.to_datetime(m["month"]).dt.month
X = pd.get_dummies(m["mnum"], prefix="m").astype(float)
X["t"] = m["t"]
fm = LinearRegression().fit(X, m["sales"])
future = pd.DataFrame({"t": range(len(m), len(m) + 3), "mnum": [1, 2, 3]})
Xf = pd.get_dummies(future["mnum"], prefix="m").reindex(columns=X.columns, fill_value=0).astype(float)
Xf["t"] = future["t"]
fc = fm.predict(Xf)
in_sample_mape = float(np.mean(np.abs((m["sales"] - fm.predict(X)) / m["sales"])) * 100)
results["forecast_Q1_2025"] = {
    "Jan": round(fc[0]), "Feb": round(fc[1]), "Mar": round(fc[2]),
    "in_sample_MAPE_pct": round(in_sample_mape, 1)}

# ---------- charts ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(pd.to_datetime(monthly["month"]), monthly["revenue"], color="#1f6f8b", lw=2)
ax.plot(pd.to_datetime(["2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]),
        [monthly["revenue"].iloc[-1], *fc], color="#d1495b", ls="--", lw=2, label="forecast")
ax.set_title("Monthly revenue and 3-month forecast")
ax.set_ylabel("Revenue (INR)"); ax.legend()
fig.tight_layout(); fig.savefig("outputs/monthly_revenue.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(store["city"], store["margin_pct"], color="#1f6f8b")
for i, v in enumerate(store["margin_pct"]):
    ax.text(i, v + 0.2, f"{v}%", ha="center")
ax.set_title("Profit margin by city"); ax.set_ylabel("Margin %")
fig.tight_layout(); fig.savefig("outputs/margin_by_city.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar((discount["discount"] * 100).astype(int).astype(str) + "%", discount["avg_qty"], color="#1f6f8b")
ax.set_title("Average units per order by discount level"); ax.set_ylabel("Units")
fig.tight_layout(); fig.savefig("outputs/discount_vs_quantity.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
imp[::-1].plot.barh(ax=ax, color="#1f6f8b")
ax.set_title("What drives order value (random forest)")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150); plt.close(fig)

# ---------- excel summary ----------
with pd.ExcelWriter("outputs/analysis_summary.xlsx") as xw:
    monthly.to_excel(xw, "Monthly revenue", index=False)
    store.to_excel(xw, "Stores", index=False)
    category.to_excel(xw, "Categories", index=False)
    discount.to_excel(xw, "Discount impact", index=False)
    segment.to_excel(xw, "Segments", index=False)
    neg_themes.rename("count").to_frame().to_excel(xw, "Negative feedback")

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
print("\nSTORE\n", store.to_string(index=False))
print("\nCATEGORY\n", category.to_string(index=False))
print("\nDISCOUNT\n", discount.to_string(index=False))
print("\nSEGMENT\n", segment.to_string(index=False))
print("\nFEATURE IMPORTANCE\n", imp.round(3).to_string())
print("\nNEG THEMES\n", neg_themes.to_string())
print("\nTONE BY CITY (%)\n", by_city_tone.to_string())
