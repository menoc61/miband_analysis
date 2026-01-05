"""
Advanced Time Series Analysis - Decomposition and Modeling
Mi Band 7 Wearable Sensor Data
MSc Data Science Assignment

This module performs:
- Time series decomposition (Classical, STL)
- Stationarity transformations (log, differencing)
- ARIMA/SARIMA model identification
- Parameter estimation and diagnostics
- Model adequacy checks
- Optional forecasting

Author: Momeni Gilles
Date: 2026-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Time series packages
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_PATH = Path('../../data/processed')
OUTPUT_PATH = Path('../../output/figures')
TABLES_PATH = Path('../../output/tables')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TIME SERIES DECOMPOSITION
# ============================================================================

def classical_decomposition(series, period=7, model='additive'):
    """
    Perform classical seasonal decomposition.
    
    Models:
    - Additive: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
      Appropriate when seasonal variation is constant
    - Multiplicative: Y(t) = Trend(t) × Seasonal(t) × Residual(t)
      Appropriate when seasonal variation scales with level
    
    Justification for Additive:
    Daily step counts typically show consistent weekly patterns
    regardless of overall activity level.
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 2 * period:
        print("Insufficient data for decomposition")
        return None
    
    decomposition = seasonal_decompose(series_clean, model=model, period=period)
    return decomposition

def stl_decomposition(series, period=7, robust=True):
    """
    Perform STL (Seasonal and Trend decomposition using Loess) decomposition.
    
    Advantages over classical decomposition:
    - More robust to outliers (when robust=True)
    - Handles missing values better
    - More flexible trend estimation
    
    Parameters:
    - period: Seasonal period (7 for weekly)
    - robust: Use robust fitting to reduce outlier influence
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 2 * period:
        print("Insufficient data for STL decomposition")
        return None
    
    stl = STL(series_clean, period=period, robust=robust)
    result = stl.fit()
    return result

def plot_decomposition(decomposition, name='series', method='STL', save=True):
    """Plot time series decomposition components."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    axes[0].plot(decomposition.observed, 'b-')
    axes[0].set_ylabel('Observed')
    axes[0].set_title(f'{method} Decomposition: {name}')
    
    axes[1].plot(decomposition.trend, 'g-')
    axes[1].set_ylabel('Trend')
    
    axes[2].plot(decomposition.seasonal, 'r-')
    axes[2].set_ylabel('Seasonal')
    
    axes[3].plot(decomposition.resid, 'purple')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / f'decomposition_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def analyze_decomposition_residuals(decomposition, name='series'):
    """
    Analyze residuals from decomposition for adequacy.
    
    Good decomposition should have:
    - Residuals with zero mean
    - No remaining autocorrelation
    - Approximately normal distribution
    """
    resid = decomposition.resid.dropna()
    
    print(f"\n{'='*60}")
    print(f"DECOMPOSITION RESIDUAL ANALYSIS: {name}")
    print('='*60)
    
    # Descriptive statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean: {resid.mean():.4f}")
    print(f"  Std: {resid.std():.4f}")
    print(f"  Skewness: {stats.skew(resid):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(resid):.4f}")
    
    # Ljung-Box test on residuals
    if len(resid) > 30:
        lb_result = acorr_ljungbox(resid, lags=[10], return_df=True)
        print(f"\nLjung-Box Test (lag=10):")
        print(f"  Statistic: {lb_result['lb_stat'].values[0]:.4f}")
        print(f"  p-value: {lb_result['lb_pvalue'].values[0]:.4f}")
        print(f"  White noise: {'Yes' if lb_result['lb_pvalue'].values[0] > 0.05 else 'No'}")
    
    return resid

# ============================================================================
# STATIONARITY TRANSFORMATIONS
# ============================================================================

def apply_transformations(series, name='series'):
    """
    Apply stationarity transformations and test each.
    
    Transformations:
    1. Original: No transformation
    2. Log: Stabilizes variance for multiplicative seasonality
    3. First Difference: Removes linear trend
    4. Log + Difference: Combined transformation
    5. Seasonal Difference: Removes seasonal pattern
    
    Selection Criteria:
    - Choose transformation that achieves stationarity
    - Prefer simpler transformation when multiple work
    """
    results = {}
    
    series_clean = series.dropna()
    series_clean = series_clean[series_clean > 0]  # Ensure positive for log
    
    print(f"\n{'='*60}")
    print(f"STATIONARITY TRANSFORMATIONS: {name}")
    print('='*60)
    
    # 1. Original
    adf_orig = adfuller(series_clean)[1]
    results['original'] = {'series': series_clean, 'adf_pvalue': adf_orig}
    print(f"\nOriginal: ADF p-value = {adf_orig:.4f}")
    
    # 2. Log transform
    log_series = np.log(series_clean)
    adf_log = adfuller(log_series)[1]
    results['log'] = {'series': log_series, 'adf_pvalue': adf_log}
    print(f"Log: ADF p-value = {adf_log:.4f}")
    
    # 3. First difference
    diff_series = series_clean.diff().dropna()
    adf_diff = adfuller(diff_series)[1]
    results['diff'] = {'series': diff_series, 'adf_pvalue': adf_diff}
    print(f"First Difference: ADF p-value = {adf_diff:.4f}")
    
    # 4. Log + difference
    log_diff_series = log_series.diff().dropna()
    adf_log_diff = adfuller(log_diff_series)[1]
    results['log_diff'] = {'series': log_diff_series, 'adf_pvalue': adf_log_diff}
    print(f"Log + Difference: ADF p-value = {adf_log_diff:.4f}")
    
    # 5. Seasonal difference (lag 7)
    if len(series_clean) > 14:
        seasonal_diff = series_clean.diff(7).dropna()
        adf_seasonal = adfuller(seasonal_diff)[1]
        results['seasonal_diff'] = {'series': seasonal_diff, 'adf_pvalue': adf_seasonal}
        print(f"Seasonal Difference (7): ADF p-value = {adf_seasonal:.4f}")
    
    # Find best transformation
    best_transform = min(results.items(), key=lambda x: x[1]['adf_pvalue'])
    print(f"\nRecommended transformation: {best_transform[0]}")
    
    return results

def plot_transformations(results, name='series', save=True):
    """Plot all transformations for comparison."""
    n_transforms = len(results)
    fig, axes = plt.subplots(n_transforms, 1, figsize=(14, 3*n_transforms))
    
    for idx, (transform_name, data) in enumerate(results.items()):
        series = data['series']
        pvalue = data['adf_pvalue']
        stationary = '(Stationary)' if pvalue < 0.05 else '(Non-stationary)'
        
        axes[idx].plot(series.index, series.values, 'b-', alpha=0.7)
        axes[idx].axhline(y=series.mean(), color='r', linestyle='--', alpha=0.5)
        axes[idx].set_title(f'{transform_name} - ADF p={pvalue:.4f} {stationary}')
        axes[idx].set_ylabel('Value')
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_PATH / f'transformations_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# ARIMA MODELING
# ============================================================================

def identify_arima_order(series, max_p=5, max_d=2, max_q=5):
    """
    Automatically identify ARIMA(p,d,q) order using auto_arima.
    
    Uses:
    - AIC/BIC for model selection
    - Unit root tests for differencing order
    - ACF/PACF for AR and MA orders
    """
    series_clean = series.dropna()
    
    print(f"\nIdentifying ARIMA order using stepwise search...")
    
    model = pm.auto_arima(
        series_clean,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        max_d=max_d,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        information_criterion='aic',
        trace=True
    )
    
    order = model.order
    print(f"\nSelected order: ARIMA{order}")
    print(f"AIC: {model.aic():.2f}")
    
    return model, order

def fit_arima_model(series, order):
    """
    Fit ARIMA model with specified order.
    
    Returns fitted model with parameter estimates and diagnostics.
    """
    series_clean = series.dropna()
    
    model = ARIMA(series_clean, order=order)
    fitted_model = model.fit()
    
    print(f"\n{'='*60}")
    print("ARIMA MODEL SUMMARY")
    print('='*60)
    print(fitted_model.summary())
    
    return fitted_model

def diagnose_arima_residuals(fitted_model, name='series', save=True):
    """
    Perform diagnostic checks on ARIMA residuals.
    
    Checks:
    1. Residual plot: Should show no pattern
    2. ACF of residuals: Should show no significant autocorrelation
    3. Q-Q plot: Should be approximately linear for normality
    4. Ljung-Box test: Should fail to reject H0 (no autocorrelation)
    """
    residuals = fitted_model.resid
    
    print(f"\n{'='*60}")
    print(f"ARIMA RESIDUAL DIAGNOSTICS: {name}")
    print('='*60)
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print("\nLjung-Box Test:")
    print(lb_result)
    
    adequate = all(lb_result['lb_pvalue'] > 0.05)
    print(f"\nModel Adequacy: {'ADEQUATE' if adequate else 'INADEQUATE'}")
    
    # Diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residual time plot
    axes[0, 0].plot(residuals, 'b-', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Residual')
    
    # Histogram
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='white')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    
    # ACF of residuals
    plot_acf(residuals, ax=axes[1, 0], lags=20, alpha=0.05)
    axes[1, 0].set_title('ACF of Residuals')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / f'arima_diagnostics_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return adequate

def forecast_arima(fitted_model, steps=14, alpha=0.05):
    """
    Generate forecasts with confidence intervals.
    
    Note: Forecasting is optional and should be interpreted with caution
    given the relatively short time series.
    """
    forecast = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)
    
    return forecast_mean, conf_int

# ============================================================================
# MAIN MODELING PIPELINE
# ============================================================================

def run_modeling_pipeline():
    """Execute the complete time series modeling pipeline."""
    print("="*60)
    print("TIME SERIES MODELING PIPELINE")
    print("Mi Band 7 Data")
    print("="*60)
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH / 'daily_combined.csv',
                     index_col='date', parse_dates=True)
    
    # Primary variable: steps
    target_var = 'steps'
    if target_var not in df.columns:
        print(f"Target variable '{target_var}' not found!")
        return
    
    series = df[target_var].dropna()
    print(f"\nTarget variable: {target_var}")
    print(f"Series length: {len(series)}")
    
    # 1. Time Series Decomposition
    print("\n" + "="*60)
    print("1. TIME SERIES DECOMPOSITION")
    print("="*60)
    
    # STL Decomposition
    stl_result = stl_decomposition(series, period=7)
    if stl_result:
        plot_decomposition(stl_result, name=target_var, method='STL')
        analyze_decomposition_residuals(stl_result, name=target_var)
    
    # 2. Stationarity Transformations
    print("\n" + "="*60)
    print("2. STATIONARITY TRANSFORMATIONS")
    print("="*60)
    
    transform_results = apply_transformations(series, name=target_var)
    plot_transformations(transform_results, name=target_var)
    
    # 3. ARIMA Model Identification
    print("\n" + "="*60)
    print("3. ARIMA MODEL IDENTIFICATION")
    print("="*60)
    
    auto_model, order = identify_arima_order(series)
    
    # 4. Model Fitting
    print("\n" + "="*60)
    print("4. ARIMA MODEL FITTING")
    print("="*60)
    
    fitted_model = fit_arima_model(series, order)
    
    # 5. Residual Diagnostics
    print("\n" + "="*60)
    print("5. RESIDUAL DIAGNOSTICS")
    print("="*60)
    
    model_adequate = diagnose_arima_residuals(fitted_model, name=target_var)
    
    # 6. Forecasting (Optional)
    print("\n" + "="*60)
    print("6. FORECASTING (14-day ahead)")
    print("="*60)
    
    forecast_mean, conf_int = forecast_arima(fitted_model, steps=14)
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical data
    ax.plot(series.index, series.values, 'b-', label='Observed')
    
    # Forecast
    forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1),
                                   periods=14, freq='D')
    ax.plot(forecast_index, forecast_mean, 'r-', linewidth=2, label='Forecast')
    ax.fill_between(forecast_index, 
                   conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                   color='red', alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Steps')
    ax.set_title(f'ARIMA{order} Forecast: {target_var}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f'forecast_{target_var}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model summary
    model_summary = {
        'variable': target_var,
        'order': str(order),
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'model_adequate': model_adequate,
        'n_observations': len(series)
    }
    
    pd.DataFrame([model_summary]).to_csv(TABLES_PATH / 'arima_model_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("MODELING PIPELINE COMPLETE")
    print("="*60)
    
    return fitted_model

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    
    model = run_modeling_pipeline()
