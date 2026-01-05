# Usage Guide - Mi Band 7 Time Series Analysis

**Author:** Momeni Gilles  
**Version:** 1.0  
**Last Updated:** January 2026

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Data Preparation](#data-preparation)
3. [Running the Analysis Pipeline](#running-the-analysis-pipeline)
4. [Understanding the Outputs](#understanding-the-outputs)
5. [Customization Options](#customization-options)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, macOS 10.15+, Linux |
| Python | 3.10 or higher |
| R | 4.0 or higher (for visualizations) |
| RAM | 4 GB minimum, 8 GB recommended |
| Storage | 500 MB free space |

### Python Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### R Dependencies

```r
ggplot2, forecast, tseries, lubridate, dplyr, tidyr, patchwork, knitr, rmarkdown
```

---

## Data Preparation

### Step 1: Export Data from Mi Fitness App

1. Open the **Mi Fitness** app on your phone
2. Go to **Profile** → **Settings** → **Data Export**
3. Select **Export All Data**
4. Download the ZIP file and extract it

### Step 2: Locate Required Files

From the exported data, you need these three CSV files:

| File Pattern | Description |
|--------------|-------------|
| `*_hlth_center_aggregated_fitness_data.csv` | Daily aggregated metrics |
| `*_hlth_center_fitness_data.csv` | Minute-level raw data |
| `*_user_fitness_data_records.csv` | Weekly summary records |

### Step 3: Place Files in Project

Copy the three CSV files to:
```
miband_analysis/data/raw/
```

---

## Running the Analysis Pipeline

### Quick Run (All Steps)

```bash
cd miband_analysis/code/python

# Run all Python scripts sequentially
python 01_data_preprocessing.py && \
python 02_exploratory_analysis.py && \
python 03_time_series_modeling.py
```

### Step-by-Step Execution

#### Script 1: Data Preprocessing

```bash
python 01_data_preprocessing.py
```

**What it does:**
- Loads raw CSV files
- Parses JSON values from the 'value' column
- Converts Unix timestamps to datetime
- Handles missing values with statistical justification
- Detects and treats outliers using IQR method
- Engineers temporal features (day_of_week, is_weekend, etc.)
- Exports clean datasets to `data/processed/`

**Expected Output:**
```
============================================================
MI BAND 7 DATA PREPROCESSING PIPELINE
============================================================
[1/7] Loading raw data files...
...
PREPROCESSING COMPLETE
============================================================
```

#### Script 2: Exploratory Analysis

```bash
python 02_exploratory_analysis.py
```

**What it does:**
- Computes descriptive statistics
- Tests for stationarity (ADF, KPSS)
- Generates ACF/PACF plots
- Performs time series decomposition
- Analyzes correlation structure
- Exports EDA results and figures

**Expected Output:**
```
============================================================
EXPLORATORY DATA ANALYSIS
============================================================
[1/6] Computing descriptive statistics...
...
EDA COMPLETE
============================================================
```

#### Script 3: Time Series Modeling

```bash
python 03_time_series_modeling.py
```

**What it does:**
- Identifies optimal ARIMA parameters
- Fits ARIMA/SARIMA models
- Performs diagnostic tests (Ljung-Box)
- Validates with out-of-sample forecasts
- Exports model summaries and forecasts

**Expected Output:**
```
============================================================
TIME SERIES MODELING
============================================================
...
MODELING COMPLETE
============================================================
```

### R Visualizations (Optional)

```bash
cd ../R
Rscript 01_visualizations.R
```

### Generate Final Report

```bash
Rscript -e "rmarkdown::render('02_analysis_report.Rmd', output_dir='../../output/reports/')"
```

---

## Understanding the Outputs

### Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `daily_combined.csv` | Main analysis dataset with all features |
| `summary_statistics.csv` | Descriptive statistics for each variable |
| `data_quality_report.json` | Missing values, outliers, transformations applied |

### Figures (`output/figures/`)

| Figure | Description |
|--------|-------------|
| `time_series_raw.png` | Original time series plots |
| `acf_pacf_*.png` | Autocorrelation diagnostics |
| `decomposition_*.png` | Trend, seasonal, residual components |
| `model_diagnostics.png` | Residual analysis plots |
| `forecast_*.png` | Model predictions vs actuals |

### Reports (`output/reports/`)

| File | Description |
|------|-------------|
| `analysis_report.html` | Full HTML report with embedded figures |
| `analysis_report.pdf` | PDF version (if LaTeX installed) |

---

## Customization Options

### Modify Analysis Parameters

Edit the configuration section at the top of each Python script:

```python
# In 01_data_preprocessing.py
RAW_DATA_PATH = Path('../../data/raw')      # Change data location
TZ_OFFSET_HOURS = 8                          # Adjust timezone
OUTLIER_THRESHOLD = 1.5                      # IQR multiplier for outliers
```

```python
# In 02_exploratory_analysis.py
MAX_LAGS = 40                                # ACF/PACF lags
DECOMPOSITION_PERIOD = 7                     # Weekly seasonality
```

```python
# In 03_time_series_modeling.py
MAX_P, MAX_D, MAX_Q = 5, 2, 5               # ARIMA search range
SEASONAL_PERIOD = 7                          # Weekly patterns
FORECAST_HORIZON = 14                        # Days to forecast
```

### Add New Variables

To analyze additional metrics, modify the `load_*` functions in the preprocessing script.

---

## Troubleshooting

### Common Issues

#### "FileNotFoundError: No such file or directory"
- **Cause:** CSV files not in `data/raw/` or incorrect path
- **Solution:** Verify files exist and path configuration is correct

#### "ModuleNotFoundError: No module named 'statsmodels'"
- **Cause:** Missing Python dependencies
- **Solution:** Run `pip install -r requirements.txt`

#### Empty or NaN results
- **Cause:** Insufficient data for analysis
- **Solution:** Ensure at least 30 days of data for reliable time series analysis

#### R script fails
- **Cause:** Missing R packages
- **Solution:** Run the installation command in R:
  ```r
  install.packages(c('ggplot2', 'forecast', 'tseries'))
  ```

### Getting Help

1. Check the `data_quality_report.json` for data issues
2. Review script console output for error messages
3. Ensure all preprocessing steps completed before running analysis

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release |

---

**Author:** Momeni Gilles
