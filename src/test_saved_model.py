# src/test_saved_model.py

import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "smart_light_model.pkl"
DATA_PATH = PROCESSED_DIR / "traffic_10min_realistic_sparse.csv"

print("📂 Loading model:", MODEL_PATH)
print("📂 Loading data:", DATA_PATH)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)


# =========================
# CLEAN DATA
# =========================

df["begin"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)

df = df.dropna(subset=["begin", "end"])

if df["is_night"].dtype == "object":
    df["is_night"] = df["is_night"].map({
        "TRUE": 1,
        "FALSE": 0,
        "True": 1,
        "False": 0,
        True: 1,
        False: 0
    })

df["is_night"] = df["is_night"].astype(int)

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["day_of_week_num"] = df["day_of_week"].map(day_map)

features = [
    "cars_10min_simulated",
    "lambda_10min",
    "zero_probability",
    "hour",
    "minute",
    "day_of_week_num",
    "is_night",
    "KFZ_R1",
    "Pkw_R1",
    "Lkw_R1",
    "Bus_R1",
    "Mot_R1"
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)


# =========================
# PREDICT USING SAVED MODEL
# =========================

print("\n🔮 Generating predictions using saved .pkl model...")

df["predicted_light"] = model.predict(df[features])

df.loc[df["is_night"] == 0, "predicted_light"] = 0

df.loc[df["is_night"] == 1, "predicted_light"] = (
    df.loc[df["is_night"] == 1, "predicted_light"]
    .clip(lower=20, upper=100)
)

df["predicted_light"] = df["predicted_light"].round(2)


# =========================
# SELECT RANDOM FULL DAY
# =========================

df = df.sort_values("begin")

daily_counts = df.groupby(df["begin"].dt.date).size()
full_days = daily_counts[daily_counts >= 144].index

if len(full_days) == 0:
    raise ValueError("No full 24-hour days found in the CSV.")

selected_day = random.choice(list(full_days))

one_day = df[df["begin"].dt.date == selected_day].copy()

print("\n📅 Selected random full day:", selected_day)
print("Rows in selected day:", len(one_day))


# =========================
# PLOT GRAPH
# =========================

print("\n📈 Creating graph...")

plt.figure(figsize=(16, 6))

plt.plot(
    one_day["begin"],
    one_day["cars_10min_simulated"],
    label="Traffic count",
    alpha=0.5
)

if "light_intensity_smooth" in one_day.columns:
    one_day.loc[one_day["is_night"] == 0, "light_intensity_smooth"] = 0

    plt.plot(
        one_day["begin"],
        one_day["light_intensity_smooth"],
        label="Actual / original light intensity",
        linestyle="--"
    )

plt.plot(
    one_day["begin"],
    one_day["predicted_light"],
    label="ML predicted light intensity",
    linewidth=2
)

y_min, y_max = plt.ylim()

plt.fill_between(
    one_day["begin"],
    y_min,
    y_max,
    where=(one_day["is_night"] == 1),
    alpha=0.08
)

plt.ylim(y_min, y_max)

plt.title(f"Smart Lighting Using Saved Model - {selected_day}")
plt.xlabel("Time")
plt.ylabel("Traffic / Light Intensity")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# =========================
# SAVE FIGURE
# =========================

file_name = f"smart_light_saved_model_{selected_day}.png"
save_path = FIGURES_DIR / file_name

plt.savefig(save_path, dpi=300)

print("📁 Graph saved to:", save_path)

plt.close()

print("\n✅ Done. Model was loaded from .pkl, not retrained.")