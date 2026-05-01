# Data Sources

## 1. Traffic Data (Germany – Multi-class Vehicle Counts)
- **Provider:** Bundesanstalt für Straßenwesen (BASt)
- **Access:** https://sensoto.io/en/
- **License:** Datenlizenz Deutschland – Namensnennung 2.0 (CC BY 2.0 DE)
- **Description:** Hourly traffic measurements including cars, trucks, buses, and motorcycles from German road sensors.
- **Variables Used:**
  - KFZ_R1 (total vehicles)
  - Pkw_R1 (cars)
  - Lkw_R1 (trucks)
  - Bus_R1 (buses)
  - Mot_R1 (motorcycles)
- **Time Range:** 2017–2026 (subset used)
- **Processing:** Data cleaned, aggregated, and transformed into 10-minute intervals for simulation and ML modeling

---

## 2. Sunshine Duration Data (Germany)
- **Provider:** Statista
- **Title:** Average monthly sunshine hours in Germany (2024–2025)
- **Description:** Monthly sunshine hours compared with long-term climatological averages (1961–1990 baseline)
- **Usage:**
  - Environmental context for smart lighting optimization
  - Supports modeling of daylight-dependent dimming behavior
- **Release Date:** January 2025

---

## 3. Derived / Synthetic Data
- **Type:** Poisson-based simulation
- **Description:**
  - Hourly traffic expanded into realistic sparse 10-minute intervals
  - Incorporates stochastic arrival processes (λ estimation)
- **Purpose:**
  - Enable high-resolution prediction for smart lighting control
  - Support ML model training where raw granularity was unavailable

---

## Notes on Data Integration
Traffic and environmental datasets were combined to create a feature-rich dataset for intelligent street lighting prediction.

**Engineered Features Include:**
- Time-based variables (hour, day of week, night indicator)
- Traffic intensity metrics
- Probability-based sparsity indicators
