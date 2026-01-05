# Mi Band 7 Time Series Analysis Project

**Author:** Momeni Gilles Christain  
**Course:** MSc Data Science - Advanced Time Series Analysis  
**Date:** January 2026

---

## Project Overview

This project performs a comprehensive time series analysis on wearable sensor data collected from a Xiaomi Mi Band 7 fitness tracker. The analysis follows rigorous academic standards suitable for a Master's-level assignment, including statistical testing, modeling, and publication-quality visualizations.

### Key Objectives

1. **Data Preprocessing**: Clean, transform, and validate raw fitness data
2. **Exploratory Data Analysis**: Understand patterns, trends, and seasonality
3. **Time Series Modeling**: Apply ARIMA/SARIMA models with proper diagnostics
4. **Visualization**: Generate publication-quality figures
5. **Academic Report**: Produce a comprehensive analytical report

---

## Project Structure

```
Momeni_Gilles_christain/
├── README.md                         # This file
├── USAGE_GUIDE.md                    # Detailed usage instructions
├── AI_REPRODUCTION_PROMPT.txt        # Prompt to reproduce via AI agent
├── run_analysis.sh                   # One-click execution script
│
├── code/
│   ├── python/
│   │   ├── 01_data_preprocessing.py  # Data cleaning and transformation
│   │   ├── 02_exploratory_analysis.py # EDA and statistical tests
│   │   ├── 03_time_series_modeling.py # ARIMA modeling
│   │   └── requirements.txt          # Python dependencies
│   └── R/
│       ├── 01_visualizations.R       # Publication-quality plots
│       ├── 02_analysis_report.Rmd    # R Markdown report
│       └── install_r_packages.R      # R dependencies installer
│
├── data/
│   ├── raw/                          # Original CSV exports
│   └── processed/                    # Analysis-ready datasets
│
├── docs/
│   ├── data_dictionary.md            # Variable definitions
│   └── methodology.md                # Statistical methods explained
│
└── output/
    ├── figures/                      # Generated plots (PNG)
    ├── tables/                       # Statistical results (CSV)
    └── reports/                      # Final reports (MD, DOCX)
```

---

## Quick Start (VSCode)

### Prerequisites

- Python 3.10+ with pip or uv
- R 4.0+ (optional, for additional visualizations)
- VSCode with Python extension

### Installation & Execution

**Option 1: Run the automated script**
```bash
bash run_analysis.sh
```

**Option 2: Step-by-step**
```bash
# Install Python dependencies
pip install -r code/python/requirements.txt

# Run preprocessing
cd code/python
python 01_data_preprocessing.py

# Run EDA
python 02_exploratory_analysis.py

# Run modeling
python 03_time_series_modeling.py
```

---

## Data Sources

| File | Description | Granularity |
|------|-------------|-------------|
| `hlth_center_aggregated_fitness_data.csv` | Daily health metrics | Daily |
| `hlth_center_fitness_data.csv` | Minute-level sensor data | Per-minute |
| `user_fitness_data_records.csv` | Weekly summaries | Weekly |

---

## Results Summary

- **Data Period:** September 17 - December 20, 2025 (95 days)
- **Primary Metric:** Daily step count (mean: 7,786 steps)
- **Stationarity:** Activity metrics are stationary (ADF p < 0.001)
- **Model:** ARIMA(0,0,0) - white noise around constant mean
- **Forecasting:** 14-day ahead predictions with 95% CI

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `daily_combined.csv` | data/processed/ | Clean analysis dataset |
| `*.png` (23 files) | output/figures/ | Visualization files |
| `analysis_report.docx` | output/reports/ | Complete Word report |
| `stationarity_tests.csv` | output/tables/ | ADF/KPSS test results |

---

## References

### Time Series Analysis

1. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. ISBN: 978-1118675021

2. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

3. Shumway, R.H., & Stoffer, D.S. (2017). *Time Series Analysis and Its Applications: With R Examples* (4th ed.). Springer. ISBN: 978-3319524511

### Statistical Tests

4. Dickey, D.A., & Fuller, W.A. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root. *Journal of the American Statistical Association*, 74(366), 427-431. https://doi.org/10.2307/2286348

5. Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). Testing the Null Hypothesis of Stationarity. *Journal of Econometrics*, 54(1-3), 159-178. https://doi.org/10.1016/0304-4076(92)90104-Y

6. Ljung, G.M., & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297-303. https://doi.org/10.1093/biomet/65.2.297

### Decomposition Methods

7. Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

### Software

8. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*, 57-61.

9. Smith, T.G. (2017). pmdarima: ARIMA estimators for Python. http://www.alkaline-ml.com/pmdarima

10. Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis* (2nd ed.). Springer. ISBN: 978-3319242774

### Wearable Technology

11. Henriksen, A., et al. (2018). Using Fitness Trackers and Smartwatches to Measure Physical Activity in Research. *Journal of Medical Internet Research*, 20(3), e110.

12. Fuller, D., et al. (2020). Reliability and Validity of Commercially Available Wearable Devices. *JMIR mHealth and uHealth*, 8(9), e18694.

---

## License

This project is for academic purposes. Personal health data is anonymized.

---

## Contact

**Author:** Momeni Gilles Christain  
**Program:** MSc Data Science

---

*Last Updated: January 2026*
