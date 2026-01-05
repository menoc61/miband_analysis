# Time Series Analysis of Wearable Fitness Data
## Mi Band 7 Health Metrics Analysis

**Author:** Momeni Gilles Christain  
**Course:** MSc Data Science - Advanced Time Series Analysis  
**Date:** January 2026

---

## Abstract

This study presents a comprehensive time series analysis of health and fitness data collected from a Xiaomi Mi Band 7 wearable device over a 95-day period (September-December 2025). The analysis encompasses data preprocessing, exploratory data analysis, stationarity testing, and ARIMA modeling. Key findings reveal that daily step counts exhibit stationary behavior with no significant autocorrelation, while stress metrics show non-stationary patterns requiring differencing. The study demonstrates the application of rigorous statistical methodology to personal health monitoring data, providing insights into activity patterns and their temporal dynamics.

**Keywords:** Time Series Analysis, Wearable Technology, ARIMA, Stationarity Testing, Health Informatics

---

## 1. Introduction

### 1.1 Background

The proliferation of wearable fitness devices has created unprecedented opportunities for continuous health monitoring. The Xiaomi Mi Band 7, released in 2022, captures multiple physiological metrics including step counts, heart rate, sleep patterns, and stress levels. This rich temporal data provides an ideal substrate for time series analysis techniques.

### 1.2 Objectives

1. To preprocess and validate raw fitness data exported from the Mi Fitness application
2. To conduct exploratory data analysis identifying trends, seasonality, and distributional characteristics
3. To test for stationarity using Augmented Dickey-Fuller and KPSS tests
4. To fit appropriate ARIMA models for short-term forecasting
5. To validate model adequacy through residual diagnostics

### 1.3 Scope

This analysis focuses on daily-aggregated metrics over a 95-day observation period, with primary emphasis on step counts as the target variable for modeling. Secondary analyses include sleep duration, sleep quality scores, and stress measurements.

---

## 2. Data Description

### 2.1 Data Source

Data was exported from the Mi Fitness mobile application (Xiaomi) in CSV format. Three data files were obtained:

| File | Description | Granularity |
|------|-------------|-------------|
| hlth_center_aggregated_fitness_data.csv | Daily health metrics | Daily |
| hlth_center_fitness_data.csv | Minute-level sensor readings | Per-minute |
| user_fitness_data_records.csv | Weekly summaries | Weekly |

### 2.2 Variables

**Primary Variables:**
- **Steps** (count): Total daily step count
- **Calories** (kcal): Active calories burned
- **Distance** (meters): Total distance traveled

**Sleep Metrics:**
- **Total Duration** (minutes): Total sleep time
- **Sleep Score** (0-100): Composite sleep quality index
- **Deep/Light/REM Duration** (minutes): Sleep stage breakdown

**Stress Metrics:**
- **Average Stress** (0-100): Daily mean stress level
- **Max/Min Stress** (0-100): Stress range

### 2.3 Data Quality

| Metric | Missing % | Treatment |
|--------|-----------|-----------|
| Steps/Calories/Distance | 0% | None required |
| Sleep metrics | 34.7% | Forward fill |
| Stress metrics | 27.4% | Forward fill |

Outliers were detected using the IQR method (1.5× threshold) and treated via winsorization:
- Steps: 2 outliers capped
- Calories: 2 outliers capped
- Distance: 2 outliers capped

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

1. **JSON Parsing**: Raw CSV files contain nested JSON in the 'value' column, requiring extraction
2. **Timestamp Conversion**: Unix timestamps converted to datetime (UTC+8)
3. **Daily Aggregation**: Minute-level data aggregated to daily summaries
4. **Missing Value Handling**: Forward fill with 3-day maximum gap
5. **Outlier Treatment**: IQR-based winsorization
6. **Feature Engineering**: Day of week, weekend flag, week of year

### 3.2 Stationarity Testing

Two complementary tests were employed:

**Augmented Dickey-Fuller (ADF) Test:**
- H₀: Series has a unit root (non-stationary)
- H₁: Series is stationary
- Rejection criterion: p-value < 0.05

**KPSS Test:**
- H₀: Series is trend-stationary
- H₁: Series has a unit root
- Rejection criterion: p-value < 0.05

### 3.3 ARIMA Modeling

The Box-Jenkins methodology was followed:

1. **Identification**: ACF/PACF analysis and auto-ARIMA search
2. **Estimation**: Maximum Likelihood Estimation
3. **Diagnostics**: Ljung-Box test for residual autocorrelation
4. **Forecasting**: 14-day ahead predictions with confidence intervals

---

## 4. Results

### 4.1 Descriptive Statistics

**Steps (n=95 days):**
- Mean: 7,786 steps/day
- Standard Deviation: 3,184 steps
- Range: [minimum] to [maximum]
- Skewness: 0.158 (approximately symmetric)
- Kurtosis: -0.139 (slightly platykurtic)

**Sleep Duration:**
- Mean: 391.4 minutes (6.5 hours)
- Standard Deviation: 171.0 minutes
- Sleep Score Mean: 59.4/100

**Stress:**
- Mean: 27.5/100
- Standard Deviation: 14.7

### 4.2 Stationarity Test Results

| Variable | ADF p-value | KPSS p-value | Conclusion |
|----------|-------------|--------------|------------|
| Steps | 0.0000 | 0.1000 | Stationary |
| Calories | 0.0000 | 0.1000 | Stationary |
| Distance | 0.0000 | 0.1000 | Stationary |
| Sleep Duration | 0.0000 | 0.1000 | Stationary |
| Sleep Score | 0.0000 | 0.0401 | Difference-stationary |
| Avg Stress | 0.0588 | 0.0115 | Non-stationary |

**Key Finding:** Activity metrics (steps, calories, distance) are stationary, while stress exhibits non-stationary behavior requiring differencing.

### 4.3 Seasonality Analysis

Spectral analysis revealed the following dominant periods:

| Variable | Dominant Period |
|----------|-----------------|
| Steps | 8.6 days |
| Calories | 4.5 days |
| Distance | 8.6 days |
| Sleep Duration | 4.3 days |
| Sleep Score | 10.6 days |
| Avg Stress | 23.7 days |

The approximately weekly periodicity in steps and distance aligns with expected behavioral patterns (weekday vs. weekend activity differences).

### 4.4 ARIMA Model Results

**Selected Model:** ARIMA(0,0,0) with constant

This selection indicates that daily step counts behave as white noise around a constant mean, suggesting:
- No significant autoregressive dependencies
- No moving average structure
- Mean-reverting behavior

**Model Diagnostics:**
- AIC: 1805.08
- Ljung-Box Q(10): 8.90, p = 0.54 (No serial correlation)
- Jarque-Bera: 0.47, p = 0.79 (Residuals approximately normal)

The model adequately captures the data generating process.

### 4.5 Autocorrelation Analysis

Ljung-Box test results at lag 20:

| Variable | Q-statistic | p-value | Serial Correlation |
|----------|-------------|---------|-------------------|
| Steps | 23.52 | 0.264 | No |
| Calories | 22.15 | 0.332 | No |
| Distance | 22.89 | 0.294 | No |
| Sleep Duration | 27.92 | 0.111 | No |
| Sleep Score | 32.74 | 0.036 | Yes |
| Avg Stress | 288.13 | <0.001 | Yes (strong) |

---

## 5. Discussion

### 5.1 Key Findings

1. **Activity Metrics Stability**: Daily step counts exhibit stationary behavior with no significant autocorrelation, suggesting that each day's activity is largely independent of previous days in this dataset.

2. **Stress Persistence**: Average stress shows strong serial correlation and non-stationarity, indicating that stress levels persist over time and may require integrated models for forecasting.

3. **Sleep Quality Patterns**: Sleep scores display intermediate behavior with some autocorrelation, potentially reflecting weekly lifestyle patterns.

4. **Model Simplicity**: The ARIMA(0,0,0) model for steps indicates that complex forecasting models are unnecessary; a simple mean predictor performs adequately.

### 5.2 Limitations

1. **Sample Size**: 95 days provides limited data for detecting long-term patterns
2. **Missing Data**: 27-35% missing values in sleep and stress metrics
3. **Single Subject**: Results may not generalize to other individuals
4. **Device Accuracy**: Wearable sensor measurements contain inherent measurement error
5. **Behavioral Context**: External factors (weather, travel, illness) not captured

### 5.3 Future Directions

1. Extend observation period to capture annual seasonality
2. Incorporate exogenous variables (weather, calendar events)
3. Apply multivariate time series methods (VAR, VECM)
4. Compare with alternative models (Prophet, LSTM neural networks)

---

## 6. Conclusion

This study demonstrated the application of classical time series methodology to wearable fitness data. The analysis revealed that daily activity metrics exhibit stationary behavior amenable to simple forecasting approaches, while stress metrics require more sophisticated modeling to account for temporal persistence. The preprocessing pipeline and analytical framework developed here provide a template for similar personal health data analyses.

---

## References

1. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

2. Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

3. Dickey, D.A., & Fuller, W.A. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root. *Journal of the American Statistical Association*, 74(366), 427-431.

4. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

5. Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root. *Journal of Econometrics*, 54(1-3), 159-178.

6. Ljung, G.M., & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297-303.

7. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*, 57-61.

8. Taylor, S.J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45.

---

## Appendix A: Project Structure

```
Momeni_Gilles_christain/
├── README.md
├── USAGE_GUIDE.md
├── AI_REPRODUCTION_PROMPT.txt
├── run_analysis.sh
├── code/
│   ├── python/
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_exploratory_analysis.py
│   │   ├── 03_time_series_modeling.py
│   │   └── requirements.txt
│   └── R/
│       ├── 01_visualizations.R
│       ├── 02_analysis_report.Rmd
│       └── install_r_packages.R
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── data_dictionary.md
│   └── methodology.md
└── output/
    ├── figures/
    ├── tables/
    └── reports/
```

---

## Appendix B: Software Environment

**Python 3.12:**
- pandas 2.0+
- numpy 1.24+
- scipy 1.11+
- statsmodels 0.14+
- pmdarima 2.0+
- matplotlib 3.7+
- seaborn 0.12+

**R 4.0+:**
- ggplot2
- forecast
- tseries
- lubridate

---

*Report generated: January 2026*  
*Author: Momeni Gilles Christain*
