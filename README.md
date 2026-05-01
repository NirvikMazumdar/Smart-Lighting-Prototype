# Enviotech Smart Lighting Prototype

## Overview
ML-based street lighting system that adjusts brightness based on:
- Traffic flow
- Time of day
- Environmental conditions (sunlight)

## Features
- Traffic simulation (Poisson-based)
- 10-minute interval modeling
- ML-based dimming prediction
- Visualization outputs

## Project Structure
- data/ → raw + processed datasets
- src/ → pipeline scripts
- models/ → trained ML models
- outputs/ → figures and results
- docs/references → data sources

## Setup
```bash
git clone <****************************my-repo****************************>
cd enviotech_prototype
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run
```bash
python src/run_pipeline.py
```

## Data Sources
See: docs/references/enviotech_data_sources.md
