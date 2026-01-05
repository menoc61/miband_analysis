# Methodology

**Author:** Momeni Gilles  
**Project:** Mi Band 7 Time Series Analysis  
**Last Updated:** January 2026

---

## Overview

This document outlines the statistical methodology employed in the time series analysis of wearable fitness data. The approach follows established practices in applied time series analysis suitable for Master's-level academic work.

---

## 1. Data Preprocessing

### 1.1 Data Cleaning

**JSON Parsing:** Raw CSV files contain nested JSON in the `value` column. We extract individual metrics using Python's `json` module with error handling for malformed entries.

**Timestamp Conversion:**
```
datetime = pd.to_datetime(unix_timestamp, unit='s', utc=True)
local_time = datetime + timedelta(hours=8)  # UTC+8
```

### 1.2 Missing Value Treatment

**Strategy:** Forward fill (last observation carried forward) with maximum gap of 3 days.

**Justification:** 
- Fitness metrics exhibit temporal continuity
- Short gaps unlikely to represent dramatic changes
- Preserves time series structure without artificial interpolation

**Documentation:** Missing percentages logged in `data_quality_report.json`

### 1.3 Outlier Detection

**Method:** Interquartile Range (IQR)

```
Q1, Q3 = data.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 × IQR
upper_bound = Q3 + 1.5 × IQR
```

**Treatment:** Winsorization (capping at bounds) rather than removal to preserve temporal structure.

---

## 2. Stationarity Testing

### 2.1 Augmented Dickey-Fuller (ADF) Test

**Purpose:** Test for unit root (non-stationarity)

**Hypotheses:**
- H₀: Series has a unit root (non-stationary)
- H₁: Series is stationary

**Decision Rule:** Reject H₀ if p-value < 0.05

**Implementation:** `statsmodels.tsa.stattools.adfuller()`

### 2.2 KPSS Test

**Purpose:** Complementary stationarity test

**Hypotheses:**
- H₀: Series is trend-stationary
- H₁: Series has a unit root

**Decision Rule:** Reject H₀ if p-value < 0.05

**Note:** ADF and KPSS test opposite hypotheses. Use both to confirm stationarity status.

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject H₀ | Fail to reject H₀ | Stationary |
| Fail to reject H₀ | Reject H₀ | Non-stationary |
| Both reject | Difference-stationary | |
| Neither reject | Trend-stationary | |

---

## 3. Time Series Decomposition

### 3.1 Classical Decomposition

**Model:** Additive
```
Yₜ = Tₜ + Sₜ + εₜ
```

Where:
- Yₜ = Observed value
- Tₜ = Trend component
- Sₜ = Seasonal component
- εₜ = Residual (irregular) component

**Period:** 7 (weekly seasonality)

### 3.2 STL Decomposition

**Method:** Seasonal and Trend decomposition using Loess

**Advantages over classical:**
- Robust to outliers
- Allows seasonal component to vary over time
- Handles missing values

**Implementation:** `statsmodels.tsa.seasonal.STL()`

---

## 4. Autocorrelation Analysis

### 4.1 Autocorrelation Function (ACF)

Measures correlation between Yₜ and Yₜ₋ₖ at various lags k.

**Interpretation:**
- Significant spikes indicate correlation at that lag
- Slow decay suggests non-stationarity
- Seasonal patterns show periodic spikes (e.g., lag 7 for weekly)

### 4.2 Partial Autocorrelation Function (PACF)

Measures direct correlation between Yₜ and Yₜ₋ₖ, controlling for intermediate lags.

**Use in ARIMA:**
- PACF cutoff after lag p → AR(p) term
- ACF cutoff after lag q → MA(q) term

**Confidence Bounds:** 95% CI = ±1.96/√n

---

## 5. ARIMA Modeling

### 5.1 Model Specification

**ARIMA(p, d, q):**
- p = Autoregressive order
- d = Differencing order
- q = Moving average order

**SARIMA(p, d, q)(P, D, Q)ₘ:**
- Adds seasonal components with period m

### 5.2 Parameter Selection

**Method:** Grid search with information criteria

**Criteria:**
- AIC (Akaike Information Criterion): Balances fit and complexity
- BIC (Bayesian Information Criterion): Penalizes complexity more heavily

**Search Range:**
- p, q: 0 to 5
- d: 0 to 2
- Seasonal: P, D, Q: 0 to 2, m = 7

### 5.3 Model Estimation

**Method:** Maximum Likelihood Estimation (MLE)

**Implementation:** `statsmodels.tsa.arima.model.ARIMA()`

---

## 6. Diagnostic Tests

### 6.1 Ljung-Box Test

**Purpose:** Test for residual autocorrelation

**Hypotheses:**
- H₀: Residuals are independently distributed (white noise)
- H₁: Residuals exhibit serial correlation

**Test Statistic:**
```
Q = n(n+2) Σₖ (ρ̂ₖ² / (n-k))
```

**Decision:** If p-value > 0.05, residuals are white noise (good fit)

### 6.2 Residual Normality

**Tests:**
- Shapiro-Wilk test
- Jarque-Bera test
- Q-Q plot visual inspection

### 6.3 Residual Diagnostics Plot

Four-panel diagnostic:
1. Standardized residuals over time
2. Histogram with normal curve
3. Q-Q plot
4. Correlogram of residuals

---

## 7. Forecast Validation

### 7.1 Train-Test Split

**Ratio:** 80% training, 20% testing

**Method:** Expanding window (no data leakage)

### 7.2 Accuracy Metrics

**RMSE (Root Mean Square Error):**
```
RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
```

**MAE (Mean Absolute Error):**
```
MAE = Σ|yᵢ - ŷᵢ| / n
```

**MAPE (Mean Absolute Percentage Error):**
```
MAPE = (100/n) Σ|yᵢ - ŷᵢ| / |yᵢ|
```

### 7.3 Forecast Intervals

**Method:** 95% prediction intervals based on estimated forecast variance

---

## 8. Software Implementation

| Task | Package | Function |
|------|---------|----------|
| Data manipulation | pandas | DataFrame operations |
| Statistical tests | statsmodels | adfuller, kpss, acf, pacf |
| ARIMA modeling | statsmodels | ARIMA, SARIMAX |
| Decomposition | statsmodels | seasonal_decompose, STL |
| Visualization | matplotlib, seaborn | Various plotting |
| R visualization | ggplot2 | Publication graphics |
| Reporting | rmarkdown | Academic report |

---

## References

1. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

2. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

3. Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

4. Ljung, G.M., & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297-303.

---

**Author:** Momeni Gilles
