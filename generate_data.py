"""
Generates a synthetic retail sales dataset with realistic mess in it
(duplicates, missing values, mixed date formats, inconsistent text).

The data is SYNTHETIC. It exists so the cleaning and analysis steps
have something realistic to work on. Run: python generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

stores = {"S01": "Nagpur", "S02": "Pune", "S03": "Mumbai", "S04": "Nashik", "S05": "Indore"}
products = {
    "Cold Brew": ("Beverages", 180, 0.35),
    "Latte": ("Beverages", 220, 0.40),
    "Green Tea": ("Beverages", 150, 0.45),
    "Croissant": ("Bakery", 130, 0.30),
    "Muffin": ("Bakery", 110, 0.32),
    "Veg Sandwich": ("Food", 190, 0.28),
    "Pasta Bowl": ("Food", 290, 0.25),
    "Coffee Beans 250g": ("Merchandise", 550, 0.22),
    "Travel Mug": ("Merchandise", 750, 0.18),
}
segments = ["Regular", "Student", "Corporate", "Tourist"]

positive = ["great taste", "loved it", "friendly staff", "quick service", "fresh", "will come back"]
negative = ["too expensive", "long wait", "cold coffee", "rude staff", "stale", "too crowded"]

# dates, with a bump in Oct-Dec and a weekend lift
dates = pd.date_range("2023-01-01", "2024-12-31")
w = np.array([1.0 + (0.35 if d.month >= 10 else 0) + (0.2 if d.weekday() >= 5 else 0) + (d.year - 2023) * 0.15
              for d in dates])
w /= w.sum()
order_dates = rng.choice(dates, size=N, p=w)

names = list(products)
prod = rng.choice(names, size=N)
store_id = rng.choice(list(stores), size=N, p=[0.28, 0.24, 0.22, 0.12, 0.14])
segment = rng.choice(segments, size=N, p=[0.5, 0.2, 0.2, 0.1])
discount = rng.choice([0, 0.05, 0.10, 0.20], size=N, p=[0.55, 0.15, 0.2, 0.1])

# discounts nudge quantity up a bit, which gives the analysis something real to find
base_qty = rng.poisson(1.6, size=N) + 1
qty = base_qty + (rng.random(N) < discount * 2.5).astype(int)

category = np.array([products[p][0] for p in prod])
price = np.array([products[p][1] for p in prod], dtype=float)
margin = np.array([products[p][2] for p in prod])
sales = price * qty * (1 - discount)
profit = sales - (price * qty * (1 - margin))

feedback = []
for d in discount:
    r = rng.random()
    if r < 0.55:
        feedback.append("")
    elif r < 0.85:
        feedback.append(str(rng.choice(positive)))
    else:
        feedback.append(str(rng.choice(negative)))

df = pd.DataFrame({
    "order_id": [f"ORD{100000 + i}" for i in range(N)],
    "order_date": pd.to_datetime(order_dates),
    "store_id": store_id,
    "city": [stores[s] for s in store_id],
    "customer_segment": segment,
    "product": prod,
    "category": category,
    "unit_price": price,
    "quantity": qty.astype(float),
    "discount": discount,
    "sales": sales.round(2),
    "profit": profit.round(2),
    "feedback": feedback,
})

# ---- add the mess ----
# 1. mixed date formats
fmt_mask = rng.random(N) < 0.12
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
df.loc[fmt_mask, "order_date"] = pd.to_datetime(df.loc[fmt_mask, "order_date"]).dt.strftime("%d/%m/%Y")

# 2. messy city text
case_mask = rng.random(N) < 0.10
df.loc[case_mask, "city"] = df.loc[case_mask, "city"].str.upper()
space_mask = rng.random(N) < 0.05
df.loc[space_mask, "city"] = " " + df.loc[space_mask, "city"] + " "

# 3. discount stored as text sometimes
df["discount"] = df["discount"].astype(object)
pct_mask = rng.random(N) < 0.08
df.loc[pct_mask, "discount"] = (df.loc[pct_mask, "discount"].astype(float) * 100).astype(int).astype(str) + "%"

# 4. missing quantity / segment
df.loc[rng.random(N) < 0.02, "quantity"] = np.nan
df.loc[rng.random(N) < 0.03, "customer_segment"] = np.nan

# 5. a few impossible values
bad = rng.choice(N, size=15, replace=False)
df.loc[bad, "quantity"] = -1

# 6. duplicate rows
dups = df.sample(90, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/raw_sales.csv", index=False)
print(f"wrote data/raw_sales.csv with {len(df)} rows")
