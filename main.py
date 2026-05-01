# main.py

from flask import Flask, request, render_template_string, send_from_directory
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "smart_light_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "traffic_10min_realistic_sparse.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model = joblib.load(MODEL_PATH)


# =========================
# FEATURES
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

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


# =========================
# RULE-BASED LIGHT FUNCTION
# =========================

def calculate_rule_based_light(cars):
    if cars == 0:
        return 25
    elif cars <= 5:
        return 35
    elif cars <= 15:
        return 50
    elif cars <= 40:
        return 75
    else:
        return 100


# =========================
# LOAD AND PREPARE DATA
# =========================

def load_data():
    df = pd.read_csv(DATA_PATH)

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
    df["day_of_week_num"] = df["day_of_week"].map(day_map)

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=features)

    return df


# =========================
# GRAPH GENERATION
# =========================

def generate_graph(selected_date):
    df = load_data()

    selected_date = pd.to_datetime(selected_date).date()

    min_date = df["begin"].dt.date.min()
    max_date = df["begin"].dt.date.max()

    # =====================================================
    # CASE 1: HISTORICAL DATE EXISTS IN DATASET
    # =====================================================

    if selected_date <= max_date:
        one_day = df[df["begin"].dt.date == selected_date].copy()

        if one_day.empty:
            return None, f"No data found for {selected_date}. Available range: {min_date} to {max_date}."

    # =====================================================
    # CASE 2: FUTURE DATE — ESTIMATE TRAFFIC FROM PAST DATA
    # =====================================================

    else:
        print("🔮 Future date selected. Generating synthetic traffic from historical patterns...")

        future_start = pd.Timestamp(selected_date, tz="UTC")
        future_times = pd.date_range(
            start=future_start,
            periods=144,
            freq="10min"
        )

        future_df = pd.DataFrame({"begin": future_times})

        future_df["end"] = future_df["begin"] + pd.Timedelta(minutes=10)
        future_df["hour"] = future_df["begin"].dt.hour
        future_df["minute"] = future_df["begin"].dt.minute
        future_df["day_of_week"] = future_df["begin"].dt.day_name()
        future_df["day_of_week_num"] = future_df["day_of_week"].map(day_map)

        future_month = pd.Timestamp(selected_date).month

        history = df[df["begin"].dt.month == future_month].copy()

        if history.empty:
            history = df.copy()

        night_pattern = (
            history
            .groupby(["hour", "minute"])["is_night"]
            .agg(lambda x: int(round(x.mean())))
            .reset_index()
        )

        feature_pattern = (
            history
            .groupby(["hour", "minute"])[[
                "cars_10min_simulated",
                "lambda_10min",
                "zero_probability",
                "KFZ_R1",
                "Pkw_R1",
                "Lkw_R1",
                "Bus_R1",
                "Mot_R1"
            ]]
            .mean()
            .reset_index()
        )

        one_day = future_df.merge(feature_pattern, on=["hour", "minute"], how="left")
        one_day = one_day.merge(night_pattern, on=["hour", "minute"], how="left")

        for col in features:
            if col in one_day.columns:
                one_day[col] = pd.to_numeric(one_day[col], errors="coerce")

        one_day = one_day.fillna(history[features].mean(numeric_only=True))
        one_day["is_night"] = one_day["is_night"].astype(int)

        # Add realistic traffic variation so future traffic is not too flat
        rng = np.random.default_rng(42)

        one_day["cars_10min_simulated"] = (
            one_day["cars_10min_simulated"]
            * rng.lognormal(mean=0, sigma=0.18, size=len(one_day))
        ).round().clip(lower=0)

        # Update dependent features after adding traffic variation
        one_day["KFZ_R1"] = one_day["cars_10min_simulated"] * 6
        one_day["lambda_10min"] = one_day["KFZ_R1"] / 6

        # Create future rule-based baseline light
        one_day["light_intensity_smooth"] = (
            one_day["cars_10min_simulated"]
            .apply(calculate_rule_based_light)
            .rolling(window=3, min_periods=1)
            .mean()
            .round(2)
        )

        # Daytime lights OFF
        one_day.loc[one_day["is_night"] == 0, "light_intensity_smooth"] = 0

    # =====================================================
    # PREDICT LIGHT USING SAVED MODEL
    # =====================================================

    one_day["predicted_light"] = model.predict(one_day[features])

    one_day.loc[one_day["is_night"] == 0, "predicted_light"] = 0

    one_day.loc[one_day["is_night"] == 1, "predicted_light"] = (
        one_day.loc[one_day["is_night"] == 1, "predicted_light"]
        .clip(lower=20, upper=100)
    )

    one_day["predicted_light"] = one_day["predicted_light"].round(2)

    if "light_intensity_smooth" in one_day.columns:
        one_day.loc[one_day["is_night"] == 0, "light_intensity_smooth"] = 0

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(16, 6))

    plt.plot(
        one_day["begin"],
        one_day["cars_10min_simulated"],
        label="Traffic count / estimated traffic",
        alpha=0.5
    )

    if (
        "light_intensity_smooth" in one_day.columns
        and one_day["light_intensity_smooth"].notna().any()
    ):
        plt.plot(
            one_day["begin"],
            one_day["light_intensity_smooth"],
            label="Rule-based light intensity",
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

    title_type = "Future Estimated" if selected_date > max_date else "Historical"
    plt.title(f"{title_type} Smart Lighting Prediction - {selected_date}")
    plt.xlabel("Time")
    plt.ylabel("Traffic / Light Intensity")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_name = f"flask_smart_light_{selected_date}.png"
    save_path = FIGURES_DIR / file_name

    plt.savefig(save_path, dpi=300)
    plt.close()

    return file_name, None


# =========================
# FLASK ROUTES
# =========================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enviotech Smart Lighting Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f7fa;
        }
        .container {
            max-width: 1100px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        }
        h1 {
            margin-bottom: 10px;
        }
        form {
            margin: 25px 0;
        }
        input, button {
            padding: 10px;
            font-size: 16px;
        }
        button {
            cursor: pointer;
            background: #1f6feb;
            color: white;
            border: none;
            border-radius: 6px;
        }
        img {
            width: 100%;
            margin-top: 25px;
            border: 1px solid #ddd;
            border-radius: 10px;
        }
        .error {
            color: #b00020;
            margin-top: 20px;
        }
        .hint {
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enviotech Smart Lighting ML Demo</h1>
        <p class="hint">Enter a date to generate a 24-hour traffic and smart-lighting prediction graph.</p>

        <form method="POST">
            <input type="date" name="selected_date" required>
            <button type="submit">Generate Graph</button>
        </form>

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}

        {% if image_file %}
            <h2>Generated Graph</h2>
            <img src="/figures/{{ image_file }}?v={{ cache_buster }}" alt="Smart lighting graph">
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    image_file = None
    error = None
    cache_buster = pd.Timestamp.now().timestamp()

    if request.method == "POST":
        selected_date = request.form.get("selected_date")
        image_file, error = generate_graph(selected_date)

    return render_template_string(
        HTML_PAGE,
        image_file=image_file,
        error=error,
        cache_buster=cache_buster
    )


@app.route("/figures/<filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True)