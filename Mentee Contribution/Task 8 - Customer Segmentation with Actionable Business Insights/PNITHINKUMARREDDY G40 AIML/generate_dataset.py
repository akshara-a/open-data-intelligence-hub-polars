"""
Fallback data generator.

Primary data source (use this first):
Dataset : "Online Shoppers Purchasing Intention Dataset"
Origin  : UCI Machine Learning Repository (Sakar, Polat, Katircioglu & Kastro, 2019),
          mirrored on Kaggle.
Kaggle  : https://www.kaggle.com/datasets/henrysue/online-shoppers-intention
UCI     : https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Real, confirmed columns (18 total, 12,330 sessions, no missing values):
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType,
Weekend, Revenue (target - True/False, ~15.5% positive class -> realistic imbalance)

This script is ONLY a stand-in for environments that cannot reach kaggle.com
(no internet / no Kaggle API token). It regenerates data with the same column
names, dtypes, and approximate distributions/correlations described in the
original paper and dataset documentation, so the rest of the notebook runs
identically either way. If a live Kaggle download succeeds, that real data is
used instead automatically (see notebook Cell 2).
"""

import numpy as np
import pandas as pd

def generate_online_shoppers_data(n=2500, seed=42):
    rng = np.random.default_rng(seed)

    visitor_type = rng.choice(["Returning_Visitor", "New_Visitor", "Other"],
                               size=n, p=[0.86, 0.12, 0.02])
    weekend = rng.choice([True, False], size=n, p=[0.23, 0.77])
    month = rng.choice(["Jan", "Feb", "Mar", "May", "June", "Jul", "Aug",
                         "Sep", "Oct", "Nov", "Dec"], size=n)
    special_day = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n,
                              p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
    operating_systems = rng.integers(1, 9, size=n)
    browser = rng.integers(1, 14, size=n)
    region = rng.integers(1, 10, size=n)
    traffic_type = rng.integers(1, 21, size=n)

    admin = rng.poisson(2.3, size=n)
    admin_dur = (admin * rng.exponential(40, size=n)).round(2)
    info = rng.poisson(0.5, size=n)
    info_dur = (info * rng.exponential(35, size=n)).round(2)
    product_related = rng.poisson(31, size=n)
    product_dur = (product_related * rng.exponential(35, size=n)).round(2)

    bounce_rates = rng.beta(1.2, 20, size=n).round(4)
    exit_rates = (bounce_rates + rng.beta(1.5, 10, size=n) * 0.5).clip(0, 1).round(4)
    page_values = (rng.exponential(6, size=n) * (rng.random(n) < 0.35)).round(4)

    # Revenue (purchase) probability driven mainly by PageValues, ProductRelated engagement,
    # low exit rate, and returning visitors -- mirrors the well-documented pattern in the
    # real dataset (page value is by far the strongest predictor).
    logit = (-4.9
              + 0.9 * np.log1p(page_values)
              + 0.015 * product_dur / 10
              + 0.01 * product_related
              - 4.0 * exit_rates
              + 0.3 * (visitor_type == "Returning_Visitor")
              + 0.4 * (special_day == 0.0)
              + rng.normal(0, 0.6, size=n))
    prob = 1 / (1 + np.exp(-logit))
    revenue = rng.random(n) < prob

    df = pd.DataFrame({
        "Administrative": admin,
        "Administrative_Duration": admin_dur,
        "Informational": info,
        "Informational_Duration": info_dur,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_dur,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
        "Revenue": revenue,
    })
    return df

if __name__ == "__main__":
    df = generate_online_shoppers_data()
    df.to_csv("data/online_shoppers_intention.csv", index=False)
    print(df.shape)
    print(df["Revenue"].value_counts(normalize=True))
    print(df.head())
