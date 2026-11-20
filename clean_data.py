"""
Step 1: clean the raw sales data.

Every fix is logged so the cleaning is auditable (rows in, rows out, what changed).
Run: python clean_data.py
"""
import json
import pandas as pd
import numpy as np

raw = pd.read_csv("data/raw_sales.csv")
log = {"rows_raw": len(raw)}
df = raw.copy()

# 1. exact duplicate rows
before = len(df)
df = df.drop_duplicates()
log["duplicates_removed"] = before - len(df)

# 2. dates: two formats are mixed (YYYY-MM-DD and DD/MM/YYYY)
iso = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
dmy = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["order_date"] = iso.fillna(dmy)
log["unparseable_dates"] = int(df["order_date"].isna().sum())
df = df.dropna(subset=["order_date"])

# 3. text columns: trim whitespace and fix casing
df["city"] = df["city"].str.strip().str.title()

# 4. discount: some values are strings like "10%"
def to_fraction(x):
    s = str(x).strip()
    return float(s.rstrip("%")) / 100 if s.endswith("%") else float(s)
df["discount"] = df["discount"].map(to_fraction)

# 5. invalid quantities (negative) are data entry errors, so drop them
bad_qty = df["quantity"] < 0
log["negative_quantity_removed"] = int(bad_qty.sum())
df = df[~bad_qty]

# 6. missing quantity: recover it from sales / (price * (1 - discount)) where possible
missing_qty = df["quantity"].isna()
recovered = (df["sales"] / (df["unit_price"] * (1 - df["discount"]))).round()
df.loc[missing_qty, "quantity"] = recovered[missing_qty]
log["quantity_recovered_from_sales"] = int(missing_qty.sum())
df["quantity"] = df["quantity"].astype(int)

# 7. missing segment: keep the row, label it explicitly rather than guessing
log["segment_missing_labelled_unknown"] = int(df["customer_segment"].isna().sum())
df["customer_segment"] = df["customer_segment"].fillna("Unknown")

# 8. feedback is free text (unstructured): empty means no comment
df["feedback"] = df["feedback"].fillna("").str.strip().str.lower()

# derived fields used later
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["margin_pct"] = (df["profit"] / df["sales"]).round(4)

log["rows_clean"] = len(df)
df.to_csv("data/clean_sales.csv", index=False)
with open("outputs/cleaning_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps(log, indent=2))
