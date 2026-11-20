# Retail Sales Analytics: Where Discounts Help and Where They Hurt

A two-year analysis of a five-city cafe chain's sales (Jan 2023 to Dec 2024).
I wanted to answer one practical question: **are our discounts actually making money?**

> **About the data:** this is a synthetic dataset I generated with `generate_data.py`
> (6,090 raw rows, with duplicates, missing values and mixed formats put in on purpose)
> so the cleaning and testing steps have realistic problems to solve. The numbers below
> come from running this code, but they do not describe a real company.

## What I found

| Question | Answer |
|---|---|
| Overall | Rs 44.0 lakh revenue, Rs 9.8 lakh profit, 22.3% margin. Revenue grew 6.6% from 2023 to 2024. |
| Do discounts raise units per order? | Yes, slightly: 2.61 to 2.88 units (Welch t-test, p < 0.001). |
| Do they raise order value? | No. Average order value stays around Rs 720 to 745 at every discount level. |
| What do they cost in margin? | Margin falls from 26.3% (no discount) to 17.8% (10%) and 8.1% (20%). |
| Which category earns the most per rupee? | Beverages at 36.5% margin. Merchandise is half of revenue but only 15.5% margin. |
| Do customer segments spend differently? | No significant difference in order value (ANOVA, p = 0.38). |
| What do customers complain about? | 33.5% of comments are negative. Top themes: long wait, crowding, cold coffee. |

**Recommendation:** stop the 20% discount tier (more units, but margin drops to 8.1% and basket value doesn't move),
and move promotion effort toward beverages where each sale earns more.

## How the project is organised

```
generate_data.py      builds the messy raw dataset
clean_data.py         cleaning, with a log of every fix (outputs/cleaning_log.json)
tests/test_data.py    8 validation checks on the cleaned data
sql/                  the SQL queries behind the tables
analysis.py           SQL aggregates, statistical tests, model, forecast, charts
outputs/              charts, Excel summary, results.json
```

Run in this order: `generate_data.py`, `clean_data.py`, `python tests/test_data.py`, `analysis.py`

## Method, briefly

**Cleaning.** Removed 90 duplicate rows and 15 rows with negative quantities. Standardised two
date formats, city spellings and discounts stored as text ("10%"). Recovered 115 missing quantities
from sales / (price x (1 - discount)) rather than dropping them. Missing customer segments were
labelled "Unknown" instead of guessed. 6,090 rows in, 5,985 out.

**Testing.** Eight checks: unique order ids, no nulls in core columns, positive quantities,
discount range, standard city names, date window, sales recomputed from price x quantity x discount,
and a reconciliation of the SQL revenue total against the pandas total.

**Statistics.** Welch t-test (discount vs quantity), Pearson correlation (discount vs margin, r = -0.54),
one-way ANOVA (segments), chi-square (segment vs category, not significant).

**Model.** Predicting order value with linear regression and a random forest gave R-squared of about 0.55
to 0.57 on a held-out 25%. My first version scored 0.99, but that was because I had included
quantity, which almost directly determines sales. I removed it. The model mostly learns
that product choice drives order value (the Travel Mug alone accounts for about half of the importance).
Treat it as descriptive, not as a production model.

**Forecast.** A linear trend plus month-of-year effects, giving a Q1 2025 estimate of roughly
Rs 1.78 to 1.91 lakh per month. In-sample error is 3.8% (MAPE), but with only 24 months of data
and no holdout this is a rough guide, not a reliable prediction.

**Text.** Feedback comments are short fixed phrases, so I grouped them by tone and theme
by simple matching. With real free text I would use proper NLP.

## Limitations

- The data is synthetic, so findings like "discounts don't raise order value" reflect how I
  generated it. The value of the project is the method, not the conclusions.
- Segment differences were not significant, which I report as is.
- The forecast has no holdout test.

## Setup

```
pip install pandas numpy scipy scikit-learn matplotlib openpyxl
```
