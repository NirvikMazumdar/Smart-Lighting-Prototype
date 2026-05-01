import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from datetime import datetime, time
from pathlib import Path

# =============================
# PATH SETUP (IMPORTANT)
# =============================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "sunlight_adjusted_dynamic_night_data.csv"

# =============================
# SETTINGS
# =============================

berlin_tz = ZoneInfo("Europe/Berlin")

start = pd.Timestamp("2017-12-31 00:00:00", tz=berlin_tz)

today_date = datetime.now(berlin_tz).date()
end = pd.Timestamp(datetime.combine(today_date, time(7, 0)), tz=berlin_tz)

np.random.seed(42)

# =============================
# CREATE TIMESTAMPS
# =============================

timestamps = pd.date_range(
    start=start,
    end=end,
    freq="1h",
    tz=berlin_tz
)

df = pd.DataFrame({"timestamp": timestamps})

df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.dayofweek

# =============================
# SUN TIMES (GERMANY APPROX)
# =============================

sun_times = {
    1:  {"sunrise": 8.2, "sunset": 16.5},
    2:  {"sunrise": 7.5, "sunset": 17.4},
    3:  {"sunrise": 6.6, "sunset": 18.3},
    4:  {"sunrise": 6.3, "sunset": 20.2},
    5:  {"sunrise": 5.4, "sunset": 21.0},
    6:  {"sunrise": 5.0, "sunset": 21.5},
    7:  {"sunrise": 5.3, "sunset": 21.4},
    8:  {"sunrise": 6.0, "sunset": 20.6},
    9:  {"sunrise": 6.8, "sunset": 19.5},
    10: {"sunrise": 7.5, "sunset": 18.3},
    11: {"sunrise": 7.4, "sunset": 16.6},
    12: {"sunrise": 8.2, "sunset": 16.2}
}

# =============================
# SUNSHINE HOURS (FROM IMAGE)
# =============================

sunshine_hours = {
    1: 70, 2: 54, 3: 120, 4: 150,
    5: 215, 6: 210, 7: 237, 8: 262,
    9: 177, 10: 100, 11: 51, 12: 42
}

max_sun = max(sunshine_hours.values())
min_sun = min(sunshine_hours.values())

# =============================
# FUNCTIONS
# =============================

def decimal_hour(ts):
    return ts.hour + ts.minute / 60


def night_strength(month):
    sun = sunshine_hours[month]
    normalized = (sun - min_sun) / (max_sun - min_sun)
    return 1 - normalized


def get_sunrise(month):
    return sun_times[month]["sunrise"]


def get_sunset(month):
    return sun_times[month]["sunset"]


def is_night(row):
    h = decimal_hour(row["timestamp"])
    sunrise = get_sunrise(row["month"])
    sunset = get_sunset(row["month"])
    return h < sunrise or h >= sunset


def hourly_night_pattern(row):
    h = decimal_hour(row["timestamp"])
    sunrise = get_sunrise(row["month"])
    sunset = get_sunset(row["month"])

    if sunrise <= h < sunset:
        return 0.0
    if sunset <= h < sunset + 2:
        return 0.65
    if h >= sunset + 2 or h < sunrise - 2:
        return 1.0
    if sunrise - 2 <= h < sunrise:
        return 0.75

    return 0.0


def weekday_factor(day):
    if day in [5, 6]:
        return np.random.uniform(0.8, 1.1)
    else:
        return np.random.uniform(0.9, 1.2)

# =============================
# GENERATE VALUES
# =============================

base_value = 100

df["sunrise_hour"] = df["month"].apply(get_sunrise)
df["sunset_hour"] = df["month"].apply(get_sunset)

df["is_night"] = df.apply(is_night, axis=1)

df["night_value"] = df.apply(
    lambda row: round(
        base_value
        * hourly_night_pattern(row)
        * (0.5 + night_strength(row["month"]))
        * weekday_factor(row["weekday"])
        * np.random.uniform(0.85, 1.15),
        2
    ),
    axis=1
)

# =============================
# FINAL OUTPUT
# =============================

df = df[[
    "timestamp",
    "sunrise_hour",
    "sunset_hour",
    "is_night",
    "night_value"
]]

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Step 1 complete")
print("📁 Saved to:", OUTPUT_FILE)
print(df.head())
print(df.tail())