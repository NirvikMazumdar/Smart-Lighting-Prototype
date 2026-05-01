# src/train_light_model.py

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "traffic_10min_realistic_sparse.csv"
MODEL_FILE = MODELS_DIR / "smart_light_model.pkl"
OUTPUT_FILE = PREDICTIONS_DIR / "light_predictions_full.csv"

VALIDATION_GRAPH_FILE = FIGURES_DIR / "validation_actual_vs_predicted.png"
TIME_SERIES_GRAPH_FILE = FIGURES_DIR / "full_24_hour_traffic_light_prediction.png"

print("🚀 Training on FULL dataset")


# =========================
# LOAD DATA
# =========================

print("📥 Loading data...")
print("Looking for CSV at:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

# Optional: remove first 50 unreliable rows
df = df.iloc[50:].copy()

print("Rows loaded:", len(df))


# =========================
# CLEAN DATA
# =========================

print("\n🧹 Cleaning data...")

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

numeric_cols = [
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
    "Mot_R1",
    "light_intensity_smooth"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# During daytime, street lights should be OFF
df.loc[df["is_night"] == 0, "light_intensity_smooth"] = 0

print("Rows after cleaning:", len(df))


# =========================
# FEATURES AND TARGET
# =========================

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

target = "light_intensity_smooth"

X = df[features]
y = df[target]


# =========================
# TRAIN TEST SPLIT
# =========================

print("\n✂️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================
# TRAIN MODEL
# =========================

print("\n==============================")
print("⚡ Training Extra Trees model on full dataset...")
print("==============================")

model = ExtraTreesRegressor(
    n_estimators=80,
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

print("\n✅ Model training complete.")


# =========================
# EVALUATION
# =========================

print("\n📊 Evaluating model...")

y_pred = model.predict(X_test)

y_pred = pd.Series(y_pred, index=X_test.index)
y_pred.loc[X_test["is_night"] == 0] = 0

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))


# =========================
# VALIDATION GRAPH
# =========================

print("\n📈 Creating validation graph...")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.25)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual Light Intensity")
plt.ylabel("Predicted Light Intensity")
plt.title("Validation: Actual vs Predicted Light Intensity")
plt.legend()
plt.tight_layout()

plt.savefig(VALIDATION_GRAPH_FILE, dpi=300)
print("Validation graph saved to:", VALIDATION_GRAPH_FILE)

plt.close()


# =========================
# PREDICT FULL DATASET
# =========================

print("\n🔮 Generating predictions for full dataset...")

df["predicted_light_intensity"] = model.predict(X)

df.loc[df["is_night"] == 0, "predicted_light_intensity"] = 0

df.loc[df["is_night"] == 1, "predicted_light_intensity"] = (
    df.loc[df["is_night"] == 1, "predicted_light_intensity"]
    .clip(lower=20, upper=100)
)

df["predicted_light_intensity"] = df["predicted_light_intensity"].round(2)

df.to_csv(OUTPUT_FILE, index=False)

print("Predictions saved to:", OUTPUT_FILE)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_FILE)

print("Model saved to:", MODEL_FILE)


# =========================
# ENERGY SAVING
# =========================

df["baseline_light"] = df["is_night"] * 100

baseline_energy = df["baseline_light"].sum()
ml_energy = df["predicted_light_intensity"].sum()

if baseline_energy > 0:
    energy_saved_percent = ((baseline_energy - ml_energy) / baseline_energy) * 100
else:
    energy_saved_percent = 0

print("\n⚡ Energy saving estimation:")
print("Baseline energy score:", round(baseline_energy, 2))
print("ML energy score:", round(ml_energy, 2))
print("Estimated energy saved:", round(energy_saved_percent, 2), "%")


# =========================
# 24-HOUR TIME SERIES GRAPH
# =========================

print("\n📈 Creating 24-hour graph with night shading...")

df = df.sort_values("begin")

unique_days = df["begin"].dt.date.unique()
selected_day = unique_days[len(unique_days) // 2]

one_day = df[df["begin"].dt.date == selected_day].copy()

print("Selected day:", selected_day)
print("Rows in selected day:", len(one_day))

plt.figure(figsize=(16, 6))

night_rows = one_day[one_day["is_night"] == 1]

for _, row in night_rows.iterrows():
    plt.axvspan(
        row["begin"],
        row["end"],
        alpha=0.12
    )

plt.plot(
    one_day["begin"],
    one_day["cars_10min_simulated"],
    label="Traffic count",
    alpha=0.5
)

plt.plot(
    one_day["begin"],
    one_day["light_intensity_smooth"],
    label="Actual / original light intensity",
    linestyle="--"
)

plt.plot(
    one_day["begin"],
    one_day["predicted_light_intensity"],
    label="ML predicted light intensity",
    linewidth=2
)

plt.title(f"Traffic vs Actual and Predicted Light Intensity - {selected_day}")
plt.xlabel("Time")
plt.ylabel("Traffic / Light Intensity")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(TIME_SERIES_GRAPH_FILE, dpi=300)

print("24-hour graph saved to:", TIME_SERIES_GRAPH_FILE)

plt.close()


# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance)

print("\n✅ Done.")