# Mi Band 7 Time Series Analysis Project

**Author:** Momeni Gilles  
**Course:** MSc Data Science - Advanced Time Series Analysis  
**Date:** January 2026

---

## Project Overview

This project performs a comprehensive time series analysis on wearable sensor data collected from a Xiaomi Mi Band 7 fitness tracker. The analysis follows rigorous academic standards suitable for a Master's-level assignment, including statistical testing, modeling, and publication-quality visualizations.

### Key Objectives

1. **Data Preprocessing**: Clean, transform, and validate raw fitness data
2. **Exploratory Data Analysis**: Understand patterns, trends, and seasonality
3. **Time Series Modeling**: Apply ARIMA/SARIMA models with proper diagnostics
4. **Visualization**: Generate publication-quality figures using R/ggplot2
5. **Academic Report**: Produce a comprehensive analytical report

---

## Project Structure

```
miband_analysis/
├── README.md                    # This file
├── USAGE_GUIDE.md              # Detailed usage instructions
├── AI_REPRODUCTION_PROMPT.txt  # Prompt to reproduce via AI agent
│
├── data/
│   ├── raw/                    # Original CSV exports from Mi Fitness
│   │   ├── *_hlth_center_aggregated_fitness_data.csv
│   │   ├── *_hlth_center_fitness_data.csv
│   │   └── *_user_fitness_data_records.csv
│   └── processed/              # Analysis-ready datasets
│       ├── daily_combined.csv
│       ├── summary_statistics.csv
│       └── data_quality_report.json
│
├── code/
│   ├── python/                 # Python analysis scripts
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_exploratory_analysis.py
│   │   ├── 03_time_series_modeling.py
│   │   └── requirements.txt
│   └── R/                      # R visualization scripts
│       ├── 01_visualizations.R
│       └── 02_analysis_report.Rmd
│
├── docs/                       # Documentation
│   ├── data_dictionary.md
│   └── methodology.md
│
└── output/                     # Generated outputs
    ├── figures/                # Plots and visualizations
    └── reports/                # Analysis reports
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- R 4.0+ (for visualizations)
- Required packages listed in `code/python/requirements.txt`

### Installation

```bash
# Navigate to project
cd miband_analysis

# Install Python dependencies
pip install -r code/python/requirements.txt

# Install R packages (optional, for visualizations)
Rscript -e "install.packages(c('ggplot2', 'forecast', 'tseries', 'lubridate', 'dplyr', 'tidyr', 'patchwork'))"
```

### Running the Analysis

Execute scripts in numerical order:

```bash
# Step 1: Preprocess raw data
cd code/python
python 01_data_preprocessing.py

# Step 2: Exploratory analysis
python 02_exploratory_analysis.py

# Step 3: Time series modeling
python 03_time_series_modeling.py

# Step 4: Generate visualizations (R)
cd ../R
Rscript 01_visualizations.R

# Step 5: Compile report
Rscript -e "rmarkdown::render('02_analysis_report.Rmd')"
```

---

## Data Sources

| File | Description | Granularity |
|------|-------------|-------------|
| `hlth_center_aggregated_fitness_data.csv` | Daily health metrics | Daily |
| `hlth_center_fitness_data.csv` | Minute-level sensor data | Per-minute |
| `user_fitness_data_records.csv` | Weekly summaries | Weekly |

**Metrics Analyzed:**
- Steps, calories, distance
- Sleep duration, stages, quality score
- Heart rate statistics
- Stress levels
- SpO2 measurements

---

## Methodology

The analysis follows established time series methodology:

1. **Stationarity Testing**: ADF and KPSS tests
2. **Decomposition**: STL and classical decomposition
3. **Autocorrelation Analysis**: ACF/PACF plots
4. **Model Selection**: Information criteria (AIC, BIC)
5. **Diagnostics**: Ljung-Box test, residual analysis
6. **Forecasting**: Out-of-sample validation

See `docs/methodology.md` for detailed methodology.

---

## Outputs

After running all scripts, find results in:

- `output/figures/`: All generated plots (PNG/PDF)
- `output/reports/`: Final analysis report
- `data/processed/`: Cleaned datasets for further analysis

---

## License

This project is for academic purposes. Data is personal and anonymized.

---

## Contact

**Author:** Momeni Gilles  
**Program:** MSc Data Science
