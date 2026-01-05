# Data Dictionary

**Author:** Momeni Gilles  
**Project:** Mi Band 7 Time Series Analysis  
**Last Updated:** January 2026

---

## Overview

This document describes all variables in the processed datasets used for time series analysis. Data originates from Xiaomi Mi Band 7 wearable device, exported via the Mi Fitness mobile application.

---

## Source Files

### 1. hlth_center_aggregated_fitness_data.csv

Daily aggregated health metrics. Primary source for daily time series.

| Field | Type | Description |
|-------|------|-------------|
| `sid` | string | Session/device identifier |
| `time` | integer | Unix timestamp (seconds) - start of day |
| `tz` | integer | Timezone offset in seconds from UTC |
| `type` | integer | Data type identifier |
| `key` | string | Metric category identifier |
| `value` | JSON string | Nested JSON containing actual metrics |

**Parsed from `value` JSON:**

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `stp` | integer | steps | Total daily step count |
| `cal` | integer | calories | Active calories burned |
| `dis` | integer | meters | Total distance walked/run |
| `ttl` | integer | minutes | Total sleep duration |
| `score` | integer | 0-100 | Sleep quality score |
| `dp` | integer | minutes | Deep sleep duration |
| `lt` | integer | minutes | Light sleep duration |
| `rem` | integer | minutes | REM sleep duration |
| `strs_avg` | integer | 0-100 | Average daily stress level |
| `strs_max` | integer | 0-100 | Maximum stress level |
| `strs_min` | integer | 0-100 | Minimum stress level |

### 2. hlth_center_fitness_data.csv

Minute-level granular sensor readings.

| Field | Type | Description |
|-------|------|-------------|
| `sid` | string | Session/device identifier |
| `time` | integer | Unix timestamp (seconds) |
| `tz` | integer | Timezone offset |
| `type` | integer | Sensor type (1=HR, 2=Steps, etc.) |
| `key` | string | Measurement identifier |
| `value` | JSON string | Sensor reading |

**Type Codes:**
- Type 1: Heart rate measurements
- Type 2: Step counts (per-minute)
- Type 5: SpO2 readings
- Type 6: Stress measurements

### 3. user_fitness_data_records.csv

Weekly summary statistics.

| Field | Type | Description |
|-------|------|-------------|
| `sid` | string | Session identifier |
| `time` | integer | Week start timestamp |
| `type` | integer | Record type |
| `key` | string | Week identifier |
| `value` | JSON string | Weekly aggregated stats |

---

## Processed Dataset: daily_combined.csv

Main analysis-ready dataset created by preprocessing pipeline.

| Variable | Type | Unit | Description | Missing Handling |
|----------|------|------|-------------|------------------|
| `date` | date | YYYY-MM-DD | Calendar date (index) | N/A |
| `steps` | float | count | Daily step total | None expected |
| `calories` | float | kcal | Active calories | None expected |
| `distance` | float | meters | Distance covered | None expected |
| `total_duration` | float | minutes | Sleep duration | Forward fill |
| `sleep_score` | float | 0-100 | Sleep quality | Forward fill |
| `sleep_deep_duration` | float | minutes | Deep sleep | Forward fill |
| `sleep_light_duration` | float | minutes | Light sleep | Forward fill |
| `sleep_rem_duration` | float | minutes | REM sleep | Forward fill |
| `avg_stress` | float | 0-100 | Mean stress | Forward fill |
| `max_stress` | float | 0-100 | Peak stress | Forward fill |
| `min_stress` | float | 0-100 | Lowest stress | Forward fill |
| `datetime` | datetime | ISO 8601 | Full timestamp | Derived |
| `hour` | integer | 0-23 | Hour of day | Derived |
| `day_of_week` | integer | 0-6 | Mon=0, Sun=6 | Derived |
| `is_weekend` | boolean | 0/1 | Weekend flag | Derived |
| `week_of_year` | integer | 1-53 | ISO week number | Derived |

---

## Data Quality Notes

### Coverage
- **Time Period:** September 2025 - December 2025
- **Total Days:** 95 days with step data
- **Sleep Coverage:** ~65% of days
- **Stress Coverage:** ~73% of days

### Missing Value Patterns
- Sleep data missing when device not worn overnight
- Stress data requires manual measurement initiation
- Steps/calories/distance complete for all tracked days

### Outlier Treatment
- Applied IQR-based winsorization (1.5× threshold)
- Affected variables: steps, calories, distance
- Original values preserved in raw data

---

## Variable Relationships

```
Steps ←→ Calories ←→ Distance    (Strong positive correlation)
     ↓
Sleep Quality ←→ Stress Level    (Negative correlation expected)
     ↓
Day of Week → Activity Patterns  (Weekend vs weekday differences)
```

---

## Usage Notes

1. **Timezone:** All timestamps converted to UTC+8 (China Standard Time)
2. **Aggregation:** Daily values computed as sum (steps) or mean (stress)
3. **Missing Dates:** Not interpolated; gaps preserved in time series
4. **Index:** Use `date` column as pandas DatetimeIndex for time series analysis

---

**Author:** Momeni Gilles
