"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Scenario: last-mile delivery delays for an e-commerce network in Kolkata.

Reference dataset: Kaggle "Food Delivery Dataset" (Indian cities delivery-time data).
If data/delivery_data.csv exists it is used; otherwise a simulated dataset with the
same structure and the same kinds of quality problems is generated.

Run:  python week2_preprocessing.py
Outputs: data/delivery_clean.csv, data/model_ready.csv, figures/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RAW_PATH = "data/delivery_data.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
RNG = np.random.default_rng(42)


# =====================================================================
# 1. DATA COLLECTION (simulated)
# =====================================================================
def simulate_raw_data(n=12000):
    """Create a raw dataset that mirrors the Kaggle delivery file, including
    its real-world issues: 'NaN ' strings, text-encoded numbers, prefixes like
    '(min) 24', trailing spaces, sign errors in coordinates, invalid ages and
    ratings, duplicates and extreme outliers."""
    hubs = np.array([[22.5726, 88.3639], [22.6265, 88.4379], [22.7046, 88.3221],
                     [22.5000, 88.3700], [22.5850, 88.3100]])   # Kolkata hubs
    h = hubs[RNG.integers(0, len(hubs), n)]
    r_lat = h[:, 0] + RNG.normal(0, 0.01, n)
    r_lon = h[:, 1] + RNG.normal(0, 0.01, n)
    dist_deg = RNG.gamma(2.0, 0.025, n)
    ang = RNG.uniform(0, 2 * np.pi, n)
    d_lat = r_lat + dist_deg * np.sin(ang)
    d_lon = r_lon + dist_deg * np.cos(ang)

    traffic = RNG.choice(["Low", "Medium", "High", "Jam"], n, p=[.34, .24, .10, .32])
    weather = RNG.choice(["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"], n)
    vehicle = RNG.choice(["motorcycle", "scooter", "electric_scooter", "bicycle"],
                         n, p=[.58, .33, .08, .01])
    festival = RNG.choice(["No", "Yes"], n, p=[.98, .02])
    city = RNG.choice(["Metropolitian", "Urban", "Semi-Urban"], n, p=[.75, .22, .03])
    age = RNG.integers(20, 40, n)
    rating = np.clip(RNG.normal(4.63, 0.33, n), 2.5, 5.0).round(1)
    multi = RNG.choice([0, 1, 2, 3], n, p=[.31, .62, .04, .03])
    vcond = RNG.choice([0, 1, 2, 3], n)
    order_date = pd.to_datetime("2022-02-11") + pd.to_timedelta(RNG.integers(0, 55, n), "D")
    order_min = RNG.integers(8 * 60, 23 * 60 + 30, n)
    pick_wait = RNG.choice([5, 10, 15], n)

    km = dist_deg * 111
    t_eff = {"Low": 0, "Medium": 4, "High": 6, "Jam": 9}
    w_eff = {"Sunny": 0, "Cloudy": 4, "Fog": 5, "Stormy": 3, "Sandstorms": 3, "Windy": 3}
    time_taken = (12 + 2.1 * km + np.vectorize(t_eff.get)(traffic)
                  + np.vectorize(w_eff.get)(weather) + 4 * multi
                  - 6 * (rating - 4.6) + 0.15 * (age - 20)
                  + 9 * (festival == "Yes") + RNG.normal(0, 3, n))
    time_taken = np.clip(time_taken, 10, 54).round().astype(int)

    fmt_time = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    df = pd.DataFrame({
        "ID": [f"0x{v:04x}" for v in RNG.choice(np.arange(10000, 99999), n, replace=False)],
        "Delivery_person_ID": [f"KOLRES{RNG.integers(1, 20):02d}DEL{RNG.integers(1, 4):02d}" for _ in range(n)],
        "Delivery_person_Age": age.astype(str),
        "Delivery_person_Ratings": rating.astype(str),
        "Restaurant_latitude": r_lat.round(6),
        "Restaurant_longitude": r_lon.round(6),
        "Delivery_location_latitude": d_lat.round(6),
        "Delivery_location_longitude": d_lon.round(6),
        "Order_Date": order_date.strftime("%d-%m-%Y"),
        "Time_Orderd": [fmt_time(m) for m in order_min],
        "Time_Order_picked": [fmt_time(m + w) for m, w in zip(order_min, pick_wait)],
        "Weatherconditions": ["conditions " + w for w in weather],
        "Road_traffic_density": [t + " " for t in traffic],
        "Vehicle_condition": vcond,
        "Type_of_order": RNG.choice(["Snack ", "Meal ", "Drinks ", "Buffet "], n),
        "Type_of_vehicle": [v + " " for v in vehicle],
        "multiple_deliveries": multi.astype(str),
        "Festival": [f + " " for f in festival],
        "City": [c + " " for c in city],
        "Time_taken(min)": [f"(min) {t}" for t in time_taken],
    })

    # ---- inject realistic quality problems ----
    def blank(col, frac, token="NaN "):
        idx = RNG.choice(n, int(frac * n), replace=False)
        df.loc[idx, col] = token
    blank("Delivery_person_Age", .041)
    blank("Delivery_person_Ratings", .042)
    blank("Time_Orderd", .038)
    blank("Weatherconditions", .013, "conditions NaN")
    blank("Road_traffic_density", .013)
    blank("multiple_deliveries", .022)
    blank("Festival", .005)
    blank("City", .026)

    idx = RNG.choice(n, 60, replace=False)            # under-age partners
    df.loc[idx, "Delivery_person_Age"] = "15"
    idx = RNG.choice(n, 50, replace=False)            # rating above max scale
    df.loc[idx, "Delivery_person_Ratings"] = "6"
    idx = RNG.choice(n, 400, replace=False)           # sign errors in coordinates
    df.loc[idx, "Restaurant_latitude"] *= -1
    df.loc[idx, "Restaurant_longitude"] *= -1
    idx = RNG.choice(n, 120, replace=False)           # GPS placeholder zeros
    df.loc[idx, ["Restaurant_latitude", "Restaurant_longitude"]] = 0.0
    idx = RNG.choice(n, 25, replace=False)            # data-entry outliers
    df.loc[idx, "Time_taken(min)"] = "(min) " + pd.Series(RNG.integers(150, 400, 25)).astype(str).values

    dups = df.sample(180, random_state=1)              # duplicate records
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=2)
    return df.reset_index(drop=True)


def collect_data():
    if os.path.exists(RAW_PATH):
        print(f"Using real dataset: {RAW_PATH}")
        return pd.read_csv(RAW_PATH)
    print("Real dataset not found - generating simulated raw data")
    df = simulate_raw_data()
    df.to_csv("data/delivery_raw_simulated.csv", index=False)
    return df


# =====================================================================
# 2. DATA QUALITY AUDIT
# =====================================================================
NA_TOKENS = ["NaN", "NaN ", "nan", "conditions NaN", ""]


def audit(df, label):
    tmp = df.replace(NA_TOKENS, np.nan)
    missing = tmp.isna().sum()
    report = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated(subset=df.columns.drop("ID")).sum()
                              if "ID" in df else df.duplicated().sum()),
        "cells_missing": int(missing.sum()),
        "pct_missing": round(100 * missing.sum() / tmp.size, 2),
    }
    print(f"\n--- Audit: {label} ---")
    for k, v in report.items():
        print(f"{k:>16}: {v}")
    return report, missing[missing > 0].sort_values(ascending=False)


# =====================================================================
# 3. CLEANING
# =====================================================================
def to_number(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def clean(df):
    log = {}
    df = df.copy()

    # 3.1 Standardise text: strip spaces, remove prefixes, real NaN
    text_cols = ["Weatherconditions", "Road_traffic_density", "Type_of_order",
                 "Type_of_vehicle", "Festival", "City"]
    for c in text_cols:
        df[c] = (df[c].astype(str).str.strip()
                 .str.replace("conditions ", "", regex=False)
                 .replace({"NaN": np.nan, "nan": np.nan}))

    # 3.2 Fix data types
    df["Time_taken(min)"] = to_number(df["Time_taken(min)"])
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = to_number(df[c])
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    df = df.rename(columns={"Time_taken(min)": "time_taken_min"})

    # 3.3 Duplicates (same order details recorded twice)
    before = len(df)
    df = df.drop_duplicates(subset=df.columns.drop("ID"))
    df = df.drop_duplicates(subset="ID")
    log["duplicates_removed"] = before - len(df)

    # 3.4 Invalid values -> missing (then imputed)
    log["invalid_age"] = int((df["Delivery_person_Age"] < 18).sum())
    df.loc[df["Delivery_person_Age"] < 18, "Delivery_person_Age"] = np.nan
    log["invalid_rating"] = int((df["Delivery_person_Ratings"] > 5).sum())
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan

    # 3.5 Coordinates: sign errors and zero placeholders
    for c in ["Restaurant_latitude", "Restaurant_longitude"]:
        log[f"sign_fixed_{c}"] = int((df[c] < 0).sum())
        df[c] = df[c].abs()
    zero = (df["Restaurant_latitude"] < 1) | (df["Restaurant_longitude"] < 1)
    log["zero_coordinates_dropped"] = int(zero.sum())
    df = df[~zero]

    # 3.6 Missing values
    miss_before = df.isna().sum()
    # Ratings: median per delivery partner, then overall median
    df["Delivery_person_Ratings"] = (df.groupby("Delivery_person_ID")["Delivery_person_Ratings"]
                                     .transform(lambda s: s.fillna(s.median())))
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["Weatherconditions", "Road_traffic_density", "Festival", "City"]:
        df[c] = df[c].fillna(df[c].mode()[0])
    # Order time: back-fill from pick-up time minus median wait (10 min)
    picked = pd.to_datetime(df["Time_Order_picked"], format="%H:%M", errors="coerce")
    ordered = pd.to_datetime(df["Time_Orderd"], format="%H:%M", errors="coerce")
    wait = (picked - ordered).dt.total_seconds() / 60
    ordered = ordered.fillna(picked - pd.Timedelta(minutes=float(wait.median())))
    df["Time_Orderd"] = ordered.dt.strftime("%H:%M")
    log["missing_filled"] = int(miss_before.sum() - df.isna().sum().sum())

    # 3.7 Feature engineering
    df["distance_km"] = haversine(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                  df["Delivery_location_latitude"],
                                  df["Delivery_location_longitude"]).round(2)
    df["order_hour"] = ordered.dt.hour
    df["pickup_wait_min"] = ((picked - ordered).dt.total_seconds() / 60).clip(lower=0)
    df["order_weekday"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = (df["order_weekday"] >= 5).astype(int)
    df["is_peak_hour"] = df["order_hour"].isin([12, 13, 19, 20, 21]).astype(int)

    # 3.8 Outliers
    # Target: values above 60 min are impossible under the platform's own limits -> drop
    q1, q3 = df["time_taken_min"].quantile([.25, .75])
    upper = q3 + 1.5 * (q3 - q1)
    log["time_iqr_upper"] = round(upper, 1)
    log["time_outliers_dropped"] = int((df["time_taken_min"] > upper).sum())
    df = df[df["time_taken_min"] <= upper]
    # Distance: real but extreme -> cap (winsorise) at 1st/99th percentile
    lo, hi = df["distance_km"].quantile([.01, .99])
    log["distance_capped"] = int(((df["distance_km"] < lo) | (df["distance_km"] > hi)).sum())
    df["distance_km"] = df["distance_km"].clip(lo, hi)

    print("\nCleaning log:")
    for k, v in log.items():
        print(f"{k:>34}: {v}")
    return df.reset_index(drop=True), log


# =====================================================================
# 4. ENCODING AND SCALING
# =====================================================================
def preprocess_for_model(df):
    traffic_order = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}     # ordinal
    city_order = {"Semi-Urban": 0, "Urban": 1, "Metropolitian": 2}
    out = df.copy()
    out["traffic_level"] = out["Road_traffic_density"].map(traffic_order)
    out["city_level"] = out["City"].map(city_order)
    out["festival_flag"] = (out["Festival"] == "Yes").astype(int)
    nominal = ["Weatherconditions", "Type_of_vehicle", "Type_of_order"]
    out = pd.get_dummies(out, columns=nominal, drop_first=True, dtype=int)

    numeric = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
               "pickup_wait_min", "multiple_deliveries", "Vehicle_condition"]
    out[[c + "_z" for c in numeric]] = StandardScaler().fit_transform(out[numeric])
    out["distance_km_minmax"] = MinMaxScaler().fit_transform(out[["distance_km"]])

    drop = ["ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked",
            "Road_traffic_density", "City", "Festival",
            "Restaurant_latitude", "Restaurant_longitude"]
    out = out.drop(columns=drop)
    return out


# =====================================================================
# 5. FIGURES
# =====================================================================
def figures(raw_missing, raw, clean_df):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(7, 3.4))
    pct = (100 * raw_missing / len(raw)).sort_values()
    ax.barh(pct.index, pct.values, color="#2E75B6")
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing values in the raw dataset")
    for i, v in enumerate(pct.values):
        ax.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig1_missing_values.png", dpi=160); plt.close(fig)

    raw_t = pd.to_numeric(raw["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].boxplot(raw_t.dropna(), widths=.5)
    axes[0].set_title("Delivery time – before"); axes[0].set_ylabel("minutes")
    axes[1].boxplot(clean_df["time_taken_min"], widths=.5)
    axes[1].set_title("Delivery time – after")
    for a in axes: a.set_xticks([])
    fig.tight_layout(); fig.savefig("figures/fig2_outliers_boxplot.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(clean_df["distance_km"], bins=40, color="#2E75B6", alpha=.85)
    ax.set_xlabel("distance (km)"); ax.set_ylabel("orders")
    ax.set_title("Engineered feature: delivery distance (after capping)")
    fig.tight_layout(); fig.savefig("figures/fig3_distance_hist.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].hist(clean_df["Delivery_person_Ratings"], bins=20, color="#8FAADC")
    axes[0].set_title("Ratings – original scale")
    z = StandardScaler().fit_transform(clean_df[["Delivery_person_Ratings"]]).ravel()
    axes[1].hist(z, bins=20, color="#1F4E79")
    axes[1].set_title("Ratings – standardised (z)")
    fig.tight_layout(); fig.savefig("figures/fig4_scaling.png", dpi=160); plt.close(fig)


# =====================================================================
if __name__ == "__main__":
    raw = collect_data()
    raw_report, raw_missing = audit(raw, "raw data")
    print(raw_missing)

    clean_df, log = clean(raw)
    clean_report, _ = audit(clean_df, "clean data")
    clean_df.to_csv("data/delivery_clean.csv", index=False)

    model_df = preprocess_for_model(clean_df)
    model_df.to_csv("data/model_ready.csv", index=False)
    print("\nModel-ready shape:", model_df.shape)
    print(clean_df[["time_taken_min", "distance_km", "Delivery_person_Age",
                    "Delivery_person_Ratings", "pickup_wait_min"]].describe().round(2))

    figures(raw_missing, raw, clean_df)
    print("\nSaved: data/delivery_clean.csv, data/model_ready.csv, figures/*.png")
