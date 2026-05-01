import pandas as pd
from pathlib import Path

# =============================
# PATH SETUP
# =============================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

traffic_file = RAW_DIR / "measurements.csv"
night_file = PROCESSED_DIR / "sunlight_adjusted_dynamic_night_data.csv"
output_file = PROCESSED_DIR / "merged_traffic_night_data.csv"


# =============================
# READ CSV FILES
# =============================

traffic_df = pd.read_csv(traffic_file, na_values=["null", "NULL", "", "NaN"])
night_df = pd.read_csv(night_file, na_values=["null", "NULL", "", "NaN"])


# =============================
# KEEP begin, end, AND ONLY _v COLUMNS
# =============================

value_cols = [col for col in traffic_df.columns if col.endswith("_v")]
traffic_df = traffic_df[["begin", "end"] + value_cols]


# =============================
# CLEAN COLUMN NAMES
# =============================

traffic_df.columns = (
    traffic_df.columns
    .str.replace(r"open\.bast-traffic\.6514\.", "", regex=True)
    .str.replace(r"\.csv_v", "", regex=True)
)


# =============================
# CONVERT TIMES TO EUROPE/BERLIN
# =============================

traffic_df["begin"] = pd.to_datetime(
    traffic_df["begin"], utc=True
).dt.tz_convert("Europe/Berlin")

traffic_df["end"] = pd.to_datetime(
    traffic_df["end"], utc=True
).dt.tz_convert("Europe/Berlin")

night_df["timestamp"] = pd.to_datetime(
    night_df["timestamp"], utc=True
).dt.tz_convert("Europe/Berlin")


# =============================
# CREATE SAFE MERGE KEYS
# avoids DST AmbiguousTimeError
# =============================

traffic_df["merge_key"] = traffic_df["begin"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
night_df["merge_key"] = night_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")


# =============================
# REPLACE NULL VALUES WITH COLUMN AVERAGE
# =============================

traffic_value_cols = [
    col for col in traffic_df.columns
    if col not in ["begin", "end", "merge_key"]
]

for col in traffic_value_cols:
    traffic_df[col] = pd.to_numeric(traffic_df[col], errors="coerce")
    traffic_df[col] = traffic_df[col].fillna(traffic_df[col].mean())


# =============================
# MERGE
# inner = ignore rows with no match
# =============================

merged_df = pd.merge(
    traffic_df,
    night_df,
    on="merge_key",
    how="inner"
)


# =============================
# CLEAN FINAL DATA
# =============================

merged_df = merged_df.drop(columns=["timestamp", "merge_key"])

front_cols = [
    "begin",
    "end",
    "is_night",
    "night_value",
    "sunrise_hour",
    "sunset_hour"
]

front_cols = [col for col in front_cols if col in merged_df.columns]
other_cols = [col for col in merged_df.columns if col not in front_cols]

merged_df = merged_df[front_cols + other_cols]


# =============================
# SAVE
# =============================

merged_df.to_csv(output_file, index=False)

print("\n✅ Step 2 complete")
print("📁 Saved merged CSV:", output_file)
print("Rows:", len(merged_df))
print("Columns:", len(merged_df.columns))
print(merged_df.head())