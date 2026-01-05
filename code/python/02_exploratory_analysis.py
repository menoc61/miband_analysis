"""
Advanced Time Series Analysis - Exploratory Data Analysis
Mi Band 7 Wearable Sensor Data
MSc Data Science Assignment

This module performs:
- Distributional analysis over time
- Trend estimation (parametric & non-parametric)
- Seasonality detection (daily, weekly cycles)
- Variance stability analysis
- Autocorrelation structure (ACF/PACF)
- Stationarity tests (ADF, KPSS)
- Ljung-Box test for serial correlation

Author: Momeni Gilles
Date: 2026-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import periodogram
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_PATH = Path('../../data/processed')
OUTPUT_PATH = Path('../../output/figures')
TABLES_PATH = Path('../../output/tables')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14

# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_data():
    """Load the preprocessed daily time series data."""
    df = pd.read_csv(PROCESSED_DATA_PATH / 'daily_combined.csv', 
                     index_col='date', parse_dates=True)
    return df

# ============================================================================
# DISTRIBUTIONAL ANALYSIS
# ============================================================================

def analyze_distributions(df, columns):
    """
    Perform distributional analysis for each variable.
    
    Tests:
    - Shapiro-Wilk: Test for normality
    - Skewness and Kurtosis: Shape parameters
    
    Interpretation:
    - Activity data often shows right-skewed distributions
    - Sleep data may be more normally distributed
    """
    print("="*60)
    print("DISTRIBUTIONAL ANALYSIS")
    print("="*60)
    
    results = []
    
    for col in columns:
        if col in df.columns and df[col].notna().sum() > 3:
            data = df[col].dropna()
            
            # Descriptive statistics
            desc = data.describe()
            
            # Normality test (Shapiro-Wilk, max 5000 samples)
            sample = data.sample(min(len(data), 5000), random_state=42)
            if len(sample) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(sample)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # Skewness and Kurtosis
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            result = {
                'variable': col,
                'n': len(data),
                'mean': desc['mean'],
                'std': desc['std'],
                'min': desc['min'],
                'max': desc['max'],
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'normal_dist': 'Yes' if shapiro_p > 0.05 else 'No'
            }
            results.append(result)
            
            print(f"\n{col}:")
            print(f"  Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}")
            print(f"  Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")
            print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_PATH / 'distribution_analysis.csv', index=False)
    
    return results_df

def plot_distributions(df, columns, save=True):
    """Create distribution plots for key variables."""
    n_cols = len([c for c in columns if c in df.columns])
    if n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4*n_cols))
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    idx = 0
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            
            # Histogram with KDE
            axes[idx, 0].hist(data, bins=30, density=True, alpha=0.7, 
                            color='steelblue', edgecolor='white')
            data.plot.kde(ax=axes[idx, 0], color='darkred', linewidth=2)
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Density')
            axes[idx, 0].set_title(f'Distribution of {col}')
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'Q-Q Plot: {col}')
            
            idx += 1
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# TREND ANALYSIS
# ============================================================================

def estimate_trend(series, method='lowess'):
    """
    Estimate trend component using various methods.
    
    Methods:
    - lowess: LOcally WEighted Scatterplot Smoothing (non-parametric)
    - polynomial: Polynomial regression (parametric)
    - moving_average: Simple moving average
    
    Justification: LOWESS is preferred for its flexibility in capturing
    non-linear trends without assuming a specific functional form.
    """
    if method == 'lowess':
        # LOWESS smoothing with fraction = 0.2
        x = np.arange(len(series))
        trend = lowess(series.values, x, frac=0.2, return_sorted=False)
    elif method == 'polynomial':
        x = np.arange(len(series))
        z = np.polyfit(x, series.values, 2)
        trend = np.polyval(z, x)
    elif method == 'moving_average':
        trend = series.rolling(window=7, center=True).mean()
    
    return trend

def plot_trend_analysis(df, column, save=True):
    """Plot time series with trend estimation."""
    if column not in df.columns:
        return
    
    series = df[column].dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Raw series with trend
    axes[0].plot(series.index, series.values, 'o-', alpha=0.5, 
                markersize=3, label='Observed')
    
    # Add LOWESS trend
    x = np.arange(len(series))
    trend_lowess = lowess(series.values, x, frac=0.2, return_sorted=False)
    axes[0].plot(series.index, trend_lowess, 'r-', linewidth=2, 
                label='LOWESS Trend')
    
    # Add 7-day moving average
    ma7 = series.rolling(window=7, center=True).mean()
    axes[0].plot(series.index, ma7, 'g--', linewidth=2, 
                label='7-day MA')
    
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(column)
    axes[0].set_title(f'Trend Analysis: {column}')
    axes[0].legend()
    
    # Detrended series
    detrended = series.values - trend_lowess
    axes[1].plot(series.index, detrended, 'b-', alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Detrended Values')
    axes[1].set_title(f'Detrended Series: {column}')
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / f'trend_{column}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# SEASONALITY DETECTION
# ============================================================================

def detect_seasonality(series, max_period=30):
    """
    Detect seasonal patterns using periodogram analysis.
    
    Method: Spectral analysis using Fourier transform
    Identifies dominant frequencies/periods in the data
    """
    series_clean = series.dropna()
    if len(series_clean) < 2 * max_period:
        return None
    
    # Compute periodogram
    freqs, psd = periodogram(series_clean.values, fs=1.0)
    
    # Convert frequencies to periods
    periods = 1 / (freqs[1:] + 1e-10)
    psd = psd[1:]
    
    # Filter to meaningful periods
    mask = (periods <= max_period) & (periods >= 2)
    periods = periods[mask]
    psd = psd[mask]
    
    if len(psd) > 0:
        # Find dominant period
        dominant_idx = np.argmax(psd)
        dominant_period = periods[dominant_idx]
        return dominant_period
    
    return None

def analyze_weekly_pattern(df, column):
    """
    Analyze weekly seasonal patterns in the data.
    
    Hypothesis: Human activity exhibits weekly cycles due to
    work schedules and weekend behavior changes.
    """
    if column not in df.columns:
        return None
    
    series = df[column].dropna()
    
    # Group by day of week
    if 'day_of_week' in df.columns:
        weekly = df.groupby('day_of_week')[column].agg(['mean', 'std'])
        weekly.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return weekly
    
    return None

def plot_seasonality(df, column, save=True):
    """Plot seasonal patterns analysis."""
    if column not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    series = df[column].dropna()
    
    # Weekly pattern
    if 'day_of_week' in df.columns:
        weekly = df.groupby('day_of_week')[column].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 0].bar(days, weekly.values, color='steelblue', edgecolor='white')
        axes[0, 0].set_xlabel('Day of Week')
        axes[0, 0].set_ylabel(f'Mean {column}')
        axes[0, 0].set_title('Weekly Pattern')
    
    # Weekend vs Weekday
    if 'is_weekend' in df.columns:
        weekend_data = [df[df['is_weekend']==0][column].dropna(),
                       df[df['is_weekend']==1][column].dropna()]
        axes[0, 1].boxplot(weekend_data, labels=['Weekday', 'Weekend'])
        axes[0, 1].set_ylabel(column)
        axes[0, 1].set_title('Weekend vs Weekday Comparison')
    
    # Periodogram
    if len(series) > 10:
        freqs, psd = periodogram(series.values, fs=1.0)
        periods = 1 / (freqs[1:] + 1e-10)
        psd = psd[1:]
        mask = (periods <= 30) & (periods >= 2)
        axes[1, 0].semilogy(periods[mask], psd[mask], 'b-')
        axes[1, 0].axvline(x=7, color='r', linestyle='--', label='Weekly (7 days)')
        axes[1, 0].set_xlabel('Period (days)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].set_title('Periodogram')
        axes[1, 0].legend()
    
    # Rolling variance
    rolling_var = series.rolling(window=7).var()
    axes[1, 1].plot(series.index, rolling_var, 'g-')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Rolling Variance (7-day)')
    axes[1, 1].set_title('Variance Stability')
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / f'seasonality_{column}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# STATIONARITY TESTS
# ============================================================================

def test_stationarity(series, name='series'):
    """
    Test for stationarity using ADF and KPSS tests.
    
    Augmented Dickey-Fuller (ADF) Test:
    - H0: Series has a unit root (non-stationary)
    - Ha: Series is stationary
    - Reject H0 if p-value < 0.05
    
    KPSS Test:
    - H0: Series is stationary around a trend
    - Ha: Series is non-stationary
    - Reject H0 if p-value < 0.05
    
    Interpretation Matrix:
    - ADF reject + KPSS not reject → Stationary
    - ADF not reject + KPSS reject → Non-stationary
    - Both reject → Trend-stationary (difference needed)
    - Neither reject → Inconclusive
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 20:
        print(f"{name}: Insufficient data for stationarity tests")
        return None
    
    print(f"\n{'='*60}")
    print(f"STATIONARITY TESTS: {name}")
    print('='*60)
    
    # ADF Test
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical = adf_result[4]
    
    print(f"\nAugmented Dickey-Fuller Test:")
    print(f"  Test Statistic: {adf_stat:.4f}")
    print(f"  p-value: {adf_pvalue:.4f}")
    print(f"  Critical Values:")
    for key, value in adf_critical.items():
        print(f"    {key}: {value:.4f}")
    
    adf_stationary = adf_pvalue < 0.05
    print(f"  Conclusion: {'Stationary' if adf_stationary else 'Non-stationary'} (α=0.05)")
    
    # KPSS Test
    kpss_result = kpss(series_clean, regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical = kpss_result[3]
    
    print(f"\nKPSS Test:")
    print(f"  Test Statistic: {kpss_stat:.4f}")
    print(f"  p-value: {kpss_pvalue:.4f}")
    print(f"  Critical Values:")
    for key, value in kpss_critical.items():
        print(f"    {key}: {value:.4f}")
    
    kpss_stationary = kpss_pvalue > 0.05
    print(f"  Conclusion: {'Stationary' if kpss_stationary else 'Non-stationary'} (α=0.05)")
    
    # Combined interpretation
    print(f"\nCombined Interpretation:")
    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-stationary (differencing recommended)"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Difference-stationary"
    else:
        conclusion = "Trend-stationary"
    print(f"  → {conclusion}")
    
    return {
        'name': name,
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'adf_stationary': adf_stationary,
        'kpss_stat': kpss_stat,
        'kpss_pvalue': kpss_pvalue,
        'kpss_stationary': kpss_stationary,
        'conclusion': conclusion
    }

# ============================================================================
# AUTOCORRELATION ANALYSIS
# ============================================================================

def analyze_autocorrelation(series, name='series', max_lag=30):
    """
    Analyze autocorrelation structure using ACF and PACF.
    
    ACF (Autocorrelation Function):
    - Measures correlation between series and its lagged values
    - Includes indirect correlations
    
    PACF (Partial Autocorrelation Function):
    - Measures direct correlation at each lag
    - Controls for intermediate lags
    
    Ljung-Box Test:
    - H0: No autocorrelation up to lag k
    - Ha: Autocorrelation exists
    """
    series_clean = series.dropna()
    
    if len(series_clean) < max_lag + 10:
        print(f"{name}: Insufficient data for autocorrelation analysis")
        return None
    
    print(f"\n{'='*60}")
    print(f"AUTOCORRELATION ANALYSIS: {name}")
    print('='*60)
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(series_clean, lags=[10, 20, 30], return_df=True)
    print("\nLjung-Box Test Results:")
    print(lb_result)
    
    # Significant autocorrelation?
    significant_ac = any(lb_result['lb_pvalue'] < 0.05)
    print(f"\nSignificant serial correlation: {'Yes' if significant_ac else 'No'}")
    
    return {
        'name': name,
        'ljungbox_results': lb_result,
        'significant_autocorrelation': significant_ac
    }

def plot_acf_pacf(series, name='series', max_lag=30, save=True):
    """Plot ACF and PACF."""
    series_clean = series.dropna()
    
    if len(series_clean) < max_lag + 10:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_acf(series_clean, ax=axes[0], lags=max_lag, alpha=0.05)
    axes[0].set_title(f'Autocorrelation Function (ACF): {name}')
    
    plot_pacf(series_clean, ax=axes[1], lags=max_lag, alpha=0.05, method='ywm')
    axes[1].set_title(f'Partial Autocorrelation Function (PACF): {name}')
    
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_PATH / f'acf_pacf_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_exploratory_analysis():
    """Execute the complete exploratory data analysis."""
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Mi Band 7 Time Series Data")
    print("="*60)
    
    # Load data
    df = load_processed_data()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Key variables for analysis
    key_vars = ['steps', 'calories', 'distance', 'total_duration', 
                'sleep_score', 'avg_stress']
    existing_vars = [v for v in key_vars if v in df.columns]
    
    # 1. Distributional Analysis
    print("\n" + "="*60)
    print("1. DISTRIBUTIONAL ANALYSIS")
    print("="*60)
    dist_results = analyze_distributions(df, existing_vars)
    plot_distributions(df, existing_vars)
    
    # 2. Trend Analysis
    print("\n" + "="*60)
    print("2. TREND ANALYSIS")
    print("="*60)
    for var in existing_vars:
        print(f"\nAnalyzing trend for {var}...")
        plot_trend_analysis(df, var)
    
    # 3. Seasonality Analysis
    print("\n" + "="*60)
    print("3. SEASONALITY ANALYSIS")
    print("="*60)
    for var in existing_vars:
        dominant_period = detect_seasonality(df[var].dropna())
        print(f"{var}: Dominant period = {dominant_period:.1f} days" if dominant_period else f"{var}: No clear periodicity")
        plot_seasonality(df, var)
    
    # 4. Stationarity Tests
    print("\n" + "="*60)
    print("4. STATIONARITY TESTS")
    print("="*60)
    stationarity_results = []
    for var in existing_vars:
        result = test_stationarity(df[var].dropna(), name=var)
        if result:
            stationarity_results.append(result)
    
    stationarity_df = pd.DataFrame(stationarity_results)
    stationarity_df.to_csv(TABLES_PATH / 'stationarity_tests.csv', index=False)
    
    # 5. Autocorrelation Analysis
    print("\n" + "="*60)
    print("5. AUTOCORRELATION ANALYSIS")
    print("="*60)
    for var in existing_vars:
        analyze_autocorrelation(df[var].dropna(), name=var)
        plot_acf_pacf(df[var].dropna(), name=var)
    
    print("\n" + "="*60)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {OUTPUT_PATH}")
    print(f"Tables saved to: {TABLES_PATH}")
    
    return df

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    
    df = run_exploratory_analysis()
