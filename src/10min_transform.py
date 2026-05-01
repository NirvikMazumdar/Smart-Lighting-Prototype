import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "merged_traffic_night_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "traffic_10min_realistic_sparse.csv"


# =========================
# SETTINGS
# =========================

TRAFFIC_COLUMN = "KFZ_R1"
TIME_COLUMN = "begin"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("🚀 Step 3 started: 10-minute realistic traffic simulation")


# =========================
# LOAD DATA
# =========================

print("📥 Loading data from:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
df = df.dropna(subset=[TIME_COLUMN, TRAFFIC_COLUMN])

df[TRAFFIC_COLUMN] = pd.to_numeric(df[TRAFFIC_COLUMN], errors="coerce")
df = df.dropna(subset=[TRAFFIC_COLUMN])

print(f"✅ Loaded {len(df)} rows")


# =========================
# ZERO PROBABILITY FUNCTION
# =========================

def zero_probability(hour, hourly_cars):
    """
    Controls how often a 10-minute interval becomes 0 cars.
    Higher chance at night and when hourly traffic is low.
    """

    if hourly_cars >= 120:
        base = 0.00
    elif hourly_cars >= 80:
        base = 0.03
    elif hourly_cars >= 40:
        base = 0.10
    elif hourly_cars >= 20:
        base = 0.20
    else:
        base = 0.35

    if 0 <= hour <= 4:
        base += 0.20
    elif 5 <= hour <= 6:
        base += 0.10
    elif 22 <= hour <= 23:
        base += 0.10

    return min(base, 0.75)


# =========================
# LIGHT DIMMING FUNCTION
# =========================

def calculate_light_intensity(cars_10min):
    """
    Simple traffic-based dimming rule.
    """

    if cars_10min == 0:
        return 25
    elif cars_10min <= 5:
        return 35
    elif cars_10min <= 15:
        return 50
    elif cars_10min <= 40:
        return 75
    else:
        return 100


# =========================
# EXPAND HOURLY TO 10-MIN DATA
# =========================

expanded_rows = []

print("🔄 Expanding hourly data into realistic sparse 10-minute intervals...")

for idx, row in df.iterrows():

    if idx % 1000 == 0:
        print(f"⏳ Processing row {idx}/{len(df)}")

    hourly_cars = max(float(row[TRAFFIC_COLUMN]), 0)
    original_hour = row[TIME_COLUMN].hour

    night_factor = 1.0

    if 0 <= original_hour <= 4 and hourly_cars < 150:
        night_factor = 0.65
    elif 5 <= original_hour <= 6 and hourly_cars < 150:
        night_factor = 0.85
    elif 22 <= original_hour <= 23 and hourly_cars < 150:
        night_factor = 0.85

    adjusted_hourly_cars = hourly_cars * night_factor
    lambda_10min = adjusted_hourly_cars / 6

    p_zero = zero_probability(original_hour, hourly_cars)

    for i in range(6):
        new_row = row.copy()

        start_time = row[TIME_COLUMN] + pd.Timedelta(minutes=10 * i)
        end_time = start_time + pd.Timedelta(minutes=10)

        if np.random.rand() < p_zero:
            cars_10min = 0
        else:
            burst_factor = np.random.lognormal(mean=0, sigma=0.25)
            cars_10min = np.random.poisson(lambda_10min * burst_factor)

        new_row["begin"] = start_time
        new_row["end"] = end_time
        new_row["hour"] = start_time.hour
        new_row["minute"] = start_time.minute
        new_row["day_of_week"] = start_time.day_name()

        new_row["lambda_10min"] = round(lambda_10min, 2)
        new_row["zero_probability"] = round(p_zero, 2)
        new_row["cars_10min_simulated"] = int(cars_10min)

        new_row["light_intensity"] = calculate_light_intensity(cars_10min)

        expanded_rows.append(new_row)


# =========================
# CREATE FINAL DATAFRAME
# =========================

print("📦 Creating final dataframe...")

expanded_df = pd.DataFrame(expanded_rows)

expanded_df["light_intensity_smooth"] = (
    expanded_df["light_intensity"]
    .rolling(window=3, min_periods=1)
    .mean()
    .round(2)
)


# =========================
# REORDER COLUMNS
# =========================

important_cols = [
    "begin",
    "end",
    "is_night",
    "hour",
    "minute",
    "day_of_week",
    TRAFFIC_COLUMN,
    "lambda_10min",
    "zero_probability",
    "cars_10min_simulated",
    "light_intensity",
    "light_intensity_smooth"
]

existing_important_cols = [col for col in important_cols if col in expanded_df.columns]
other_cols = [col for col in expanded_df.columns if col not in existing_important_cols]

expanded_df = expanded_df[existing_important_cols + other_cols]


# =========================
# SAVE OUTPUT
# =========================

expanded_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Step 3 complete")
print("📁 Saved file to:", OUTPUT_FILE)
print("📊 Final rows:", len(expanded_df))
print("🔍 Preview:")
print(expanded_df.head(20))