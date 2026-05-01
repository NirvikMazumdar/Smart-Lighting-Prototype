import subprocess
import sys
from pathlib import Path

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = Path(__file__).resolve().parent

# =========================
# PIPELINE STEPS (ORDER MATTERS)
# =========================

scripts = [
    BASE_DIR / "src" / "data_gen_sunlighthours_adjusted.py",  # Step 1
    BASE_DIR / "src" / "merge_filtering.py",                  # Step 2
    BASE_DIR / "src" / "10min_transform.py",                  # Step 3
    BASE_DIR / "src" / "train_light_model.py",                # Step 4
    BASE_DIR / "src" / "test_saved_model.py",                 # Step 5
]

print("\n🚀 Starting Enviotech Smart Lighting Pipeline\n")

# =========================
# RUN EACH STEP
# =========================

for script in scripts:
    print("=" * 60)
    print(f"▶ Running: {script.name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\n❌ Pipeline stopped. Error in: {script.name}")
        sys.exit(result.returncode)

    print(f"✅ Completed: {script.name}\n")

# =========================
# FINAL MESSAGE
# =========================

print("=" * 60)
print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\n📁 Outputs:")
print("Model → models/smart_light_model.pkl")
print("Predictions → outputs/predictions/")
print("Figures → outputs/figures/")