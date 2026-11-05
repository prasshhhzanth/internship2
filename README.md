# Logistics Data Analyst Internship – Week 2

**Data Collection, Cleaning and Preprocessing for Logistics Analysis**

`week2_preprocessing.py` runs the full pipeline:

1. **Collect** – uses `data/delivery_data.csv` (Kaggle "Food Delivery Dataset") if present, otherwise generates a simulated 12,000-order Kolkata dataset with the same columns and the same quality problems
2. **Audit** – rows, duplicates, missing values (including `"NaN "` text placeholders)
3. **Clean** – text standardisation, type conversion, duplicate removal, invalid ages/ratings, coordinate sign errors and zero GPS values
4. **Missing values** – group median (ratings), median, mode, and derived order time
5. **Feature engineering** – haversine distance, pickup wait, order hour, weekday, weekend and peak-hour flags
6. **Outliers** – IQR rule: drop impossible delivery times, cap extreme distances
7. **Encode and scale** – ordinal and one-hot encoding, StandardScaler and MinMaxScaler

## Run

```bash
pip install -r requirements.txt
python week2_preprocessing.py
```

Outputs: `data/delivery_clean.csv`, `data/model_ready.csv`, and four charts in `figures/`.
