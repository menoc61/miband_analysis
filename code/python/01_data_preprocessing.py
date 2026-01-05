"""
Advanced Time Series Analysis - Data Preprocessing Pipeline
Mi Band 7 Wearable Sensor Data
MSc Data Science Assignment

This module handles:
- Data loading and validation
- Timestamp harmonization
- Missing value handling (with statistical justification)
- Outlier detection and treatment
- Feature engineering
- Export of analysis-ready datasets

Author: Momeni Gilles
Date: 2026-01
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_PATH = Path('../../data/raw')
PROCESSED_DATA_PATH = Path('../../data/processed')
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Timezone offset (data appears to be in UTC+8 based on Mi Fitness app)
TZ_OFFSET_HOURS = 8

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_aggregated_fitness_data(filepath):
    """
    Load and parse the aggregated fitness data (daily reports).
    
    This file contains daily-level aggregated data including:
    - Steps, calories, distance
    - Sleep metrics (duration, stages, score)
    - Heart rate statistics
    - SpO2 measurements
    - Stress levels
    
    Data Generating Process (DGP):
    - Data is collected continuously by Mi Band 7 sensors
    - Aggregated daily by Mi Fitness app at end of day
    - Subject to sensor limitations (motion artifacts, poor skin contact)
    """
    print("Loading aggregated fitness data...")
    df = pd.read_csv(filepath)
    
    # Parse the 'Value' column which contains JSON data
    records = []
    for _, row in df.iterrows():
        try:
            value = json.loads(row['Value']) if pd.notna(row['Value']) else {}
        except json.JSONDecodeError:
            value = {}
        
        record = {
            'uid': row['Uid'],
            'sid': row['Sid'],
            'tag': row['Tag'],
            'key': row['Key'],
            'timestamp': row['Time'],
            'update_time': row['UpdateTime'],
            **value
        }
        records.append(record)
    
    return pd.DataFrame(records)

def load_fitness_data(filepath):
    """
    Load and parse the minute-level fitness data.
    
    This file contains high-frequency sensor data:
    - Steps (per minute)
    - Heart rate measurements
    - Calories burned
    - Activity dynamics
    
    Sensor Limitations:
    - Heart rate: optical PPG sensor, affected by motion/skin tone
    - Steps: accelerometer-based, may miss low-intensity movement
    - Sampling rate varies based on activity detection
    """
    print("Loading minute-level fitness data...")
    df = pd.read_csv(filepath)
    
    records = []
    for _, row in df.iterrows():
        try:
            value = json.loads(row['Value']) if pd.notna(row['Value']) else {}
        except json.JSONDecodeError:
            value = {}
        
        record = {
            'uid': row['Uid'],
            'sid': row['Sid'],
            'key': row['Key'],
            'timestamp': row['Time'],
            'update_time': row['UpdateTime'],
            **value
        }
        records.append(record)
    
    return pd.DataFrame(records)

def load_weekly_records(filepath):
    """
    Load weekly statistics and fitness reports.
    """
    print("Loading weekly records...")
    df = pd.read_csv(filepath)
    
    records = []
    for _, row in df.iterrows():
        try:
            value = json.loads(row['value']) if pd.notna(row['value']) else {}
        except json.JSONDecodeError:
            value = {}
        
        record = {
            'uid': row['uid'],
            'tag': row['tag'],
            'key': row['key'],
            'timestamp': row['time'],
            'did': row['did'],
            'metric': row['metric'],
            'last_modify': row['last_modify'],
            **value
        }
        records.append(record)
    
    return pd.DataFrame(records)

# ============================================================================
# TIMESTAMP PROCESSING
# ============================================================================

def convert_unix_timestamp(ts, tz_offset=TZ_OFFSET_HOURS):
    """
    Convert Unix timestamp to datetime with timezone handling.
    
    Justification: Mi Fitness uses Unix timestamps in seconds.
    We convert to local time (UTC+8) for behavioral interpretation.
    """
    if pd.isna(ts) or ts == 0:
        return pd.NaT
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return dt
    except (ValueError, OSError):
        return pd.NaT

def add_temporal_features(df, timestamp_col='datetime'):
    """
    Engineer temporal features for time series analysis.
    
    Features:
    - hour: Hour of day (0-23) - captures diurnal patterns
    - day_of_week: Day of week (0=Mon, 6=Sun) - captures weekly cycles
    - is_weekend: Binary indicator for weekend
    - week_of_year: ISO week number
    """
    df = df.copy()
    if timestamp_col in df.columns and not df[timestamp_col].isna().all():
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        df['date'] = df[timestamp_col].dt.date
    return df

# ============================================================================
# MISSING VALUE ANALYSIS
# ============================================================================

def analyze_missingness(df, columns):
    """
    Analyze missing value patterns and mechanisms.
    
    Missingness Mechanisms:
    - MCAR (Missing Completely At Random): No pattern
    - MAR (Missing At Random): Depends on observed data
    - MNAR (Missing Not At Random): Depends on unobserved data
    
    For wearable data, missingness is typically MAR/MNAR:
    - Device not worn (MAR - observable through other sensors)
    - Poor sensor contact during activity (MNAR)
    - Battery depletion (MAR - observable through data gaps)
    """
    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)
    
    results = {}
    for col in columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            results[col] = {
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'mechanism': 'MAR' if missing_pct < 30 else 'MNAR'
            }
            print(f"{col}: {missing_count} missing ({missing_pct:.2f}%)")
    
    return results

def handle_missing_values(df, numeric_cols, method='interpolate'):
    """
    Handle missing values using statistically justified methods.
    
    Methods:
    - interpolate: Linear interpolation for continuous series
    - forward_fill: Carry forward last observation
    - median: Robust central tendency for sparse missingness
    
    Justification: Linear interpolation is appropriate for physiological
    time series as these signals change gradually over time.
    """
    df = df.copy()
    
    for col in numeric_cols:
        if col in df.columns:
            if method == 'interpolate':
                # Linear interpolation for time series continuity
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
    
    return df

# ============================================================================
# OUTLIER DETECTION AND TREATMENT
# ============================================================================

def detect_outliers_iqr(series, k=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Justification: IQR is robust to extreme values and appropriate
    for non-normal distributions common in activity data.
    
    Parameters:
    - k: Multiplier for IQR bounds (1.5 = mild outliers, 3 = extreme)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(series, threshold=3):
    """
    Detect outliers using Modified Z-score (robust to outliers).
    
    Uses Median Absolute Deviation (MAD) instead of standard deviation.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z = 0.6745 * (series - median) / (mad + 1e-10)
    
    outliers = np.abs(modified_z) > threshold
    return outliers

def treat_outliers(df, columns, method='winsorize', bounds=None):
    """
    Treat outliers using winsorization or capping.
    
    Justification: Winsorization preserves the ordering and reduces
    extreme values without removing data points - important for 
    maintaining time series continuity.
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            outliers, lower, upper = detect_outliers_iqr(df[col])
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                print(f"  {col}: {n_outliers} outliers detected")
                if method == 'winsorize':
                    df.loc[df[col] < lower, col] = lower
                    df.loc[df[col] > upper, col] = upper
    
    return df

# ============================================================================
# DAILY AGGREGATION
# ============================================================================

def create_daily_steps_series(df):
    """
    Create daily steps time series from aggregated data.
    """
    steps_data = df[df['key'] == 'steps'].copy()
    
    if 'steps' in steps_data.columns:
        steps_data['datetime'] = steps_data['timestamp'].apply(convert_unix_timestamp)
        steps_data = steps_data.dropna(subset=['datetime'])
        steps_data['date'] = steps_data['datetime'].dt.date
        
        daily_steps = steps_data.groupby('date').agg({
            'steps': 'sum',
            'calories': 'sum',
            'distance': 'sum'
        }).reset_index()
        
        daily_steps['date'] = pd.to_datetime(daily_steps['date'])
        daily_steps = daily_steps.sort_values('date').set_index('date')
        
        return daily_steps
    
    return pd.DataFrame()

def create_daily_sleep_series(df):
    """
    Create daily sleep metrics time series.
    """
    sleep_data = df[df['key'] == 'sleep'].copy()
    
    if len(sleep_data) > 0:
        sleep_data['datetime'] = sleep_data['timestamp'].apply(convert_unix_timestamp)
        sleep_data = sleep_data.dropna(subset=['datetime'])
        sleep_data['date'] = sleep_data['datetime'].dt.date
        
        sleep_cols = ['total_duration', 'sleep_score', 'sleep_deep_duration', 
                      'sleep_light_duration', 'sleep_rem_duration']
        existing_cols = [c for c in sleep_cols if c in sleep_data.columns]
        
        if existing_cols:
            daily_sleep = sleep_data.groupby('date')[existing_cols].first().reset_index()
            daily_sleep['date'] = pd.to_datetime(daily_sleep['date'])
            daily_sleep = daily_sleep.sort_values('date').set_index('date')
            return daily_sleep
    
    return pd.DataFrame()

def create_daily_hr_series(df):
    """
    Create daily heart rate statistics time series.
    """
    hr_data = df[df['key'] == 'heart_rate'].copy()
    
    if 'bpm' in hr_data.columns:
        hr_data['datetime'] = hr_data['timestamp'].apply(convert_unix_timestamp)
        hr_data = hr_data.dropna(subset=['datetime'])
        hr_data['date'] = hr_data['datetime'].dt.date
        
        daily_hr = hr_data.groupby('date').agg({
            'bpm': ['mean', 'min', 'max', 'std']
        }).reset_index()
        daily_hr.columns = ['date', 'hr_mean', 'hr_min', 'hr_max', 'hr_std']
        daily_hr['date'] = pd.to_datetime(daily_hr['date'])
        daily_hr = daily_hr.sort_values('date').set_index('date')
        
        return daily_hr
    
    return pd.DataFrame()

def create_daily_stress_series(df):
    """
    Create daily stress metrics time series.
    """
    stress_data = df[df['key'] == 'stress'].copy()
    
    if len(stress_data) > 0:
        stress_data['datetime'] = stress_data['timestamp'].apply(convert_unix_timestamp)
        stress_data = stress_data.dropna(subset=['datetime'])
        stress_data['date'] = stress_data['datetime'].dt.date
        
        stress_cols = ['avg_stress', 'max_stress', 'min_stress']
        existing_cols = [c for c in stress_cols if c in stress_data.columns]
        
        if existing_cols:
            daily_stress = stress_data.groupby('date')[existing_cols].first().reset_index()
            daily_stress['date'] = pd.to_datetime(daily_stress['date'])
            daily_stress = daily_stress.sort_values('date').set_index('date')
            return daily_stress
    
    return pd.DataFrame()

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def run_preprocessing_pipeline():
    """
    Execute the complete data preprocessing pipeline.
    
    Pipeline Steps:
    1. Load raw data files
    2. Parse JSON values and convert timestamps
    3. Analyze and handle missing values
    4. Detect and treat outliers
    5. Aggregate to daily level
    6. Engineer temporal features
    7. Export clean datasets
    """
    print("="*60)
    print("MI BAND 7 DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print("\n[1/7] Loading raw data files...")
    
    aggregated_file = RAW_DATA_PATH / '20251220_6736505735_MiFitness_hlth_center_aggregated_fitness_data.csv'
    fitness_file = RAW_DATA_PATH / '20251220_6736505735_MiFitness_hlth_center_fitness_data.csv'
    weekly_file = RAW_DATA_PATH / '20251220_6736505735_MiFitness_user_fitness_data_records.csv'
    
    df_aggregated = load_aggregated_fitness_data(aggregated_file)
    df_weekly = load_weekly_records(weekly_file)
    
    print(f"  Aggregated data: {len(df_aggregated)} records")
    print(f"  Weekly records: {len(df_weekly)} records")
    
    # Step 2: Create daily time series
    print("\n[2/7] Creating daily time series...")
    
    daily_steps = create_daily_steps_series(df_aggregated)
    daily_sleep = create_daily_sleep_series(df_aggregated)
    daily_stress = create_daily_stress_series(df_aggregated)
    
    print(f"  Daily steps: {len(daily_steps)} days")
    print(f"  Daily sleep: {len(daily_sleep)} days")
    print(f"  Daily stress: {len(daily_stress)} days")
    
    # Step 3: Merge daily datasets
    print("\n[3/7] Merging daily datasets...")
    
    # Start with steps as base
    daily_combined = daily_steps.copy()
    
    # Merge sleep data
    if len(daily_sleep) > 0:
        daily_combined = daily_combined.join(daily_sleep, how='outer')
    
    # Merge stress data  
    if len(daily_stress) > 0:
        daily_combined = daily_combined.join(daily_stress, how='outer')
    
    daily_combined = daily_combined.sort_index()
    print(f"  Combined dataset: {len(daily_combined)} days")
    
    # Step 4: Analyze missingness
    print("\n[4/7] Analyzing missing values...")
    numeric_cols = daily_combined.select_dtypes(include=[np.number]).columns.tolist()
    missingness = analyze_missingness(daily_combined, numeric_cols)
    
    # Step 5: Handle missing values
    print("\n[5/7] Handling missing values...")
    daily_clean = handle_missing_values(daily_combined, numeric_cols, method='interpolate')
    
    # Step 6: Detect and treat outliers
    print("\n[6/7] Detecting and treating outliers...")
    outlier_cols = ['steps', 'calories', 'distance']
    outlier_cols = [c for c in outlier_cols if c in daily_clean.columns]
    daily_clean = treat_outliers(daily_clean, outlier_cols, method='winsorize')
    
    # Step 7: Add temporal features
    print("\n[7/7] Engineering temporal features...")
    daily_clean = daily_clean.reset_index()
    daily_clean['datetime'] = pd.to_datetime(daily_clean['date'])
    daily_clean = add_temporal_features(daily_clean, 'datetime')
    daily_clean = daily_clean.set_index('date')
    
    # Export processed data
    print("\n" + "="*60)
    print("EXPORTING PROCESSED DATA")
    print("="*60)
    
    daily_clean.to_csv(PROCESSED_DATA_PATH / 'daily_combined.csv')
    print(f"  Saved: daily_combined.csv ({len(daily_clean)} records)")
    
    # Export summary statistics
    summary = daily_clean.describe()
    summary.to_csv(PROCESSED_DATA_PATH / 'summary_statistics.csv')
    print(f"  Saved: summary_statistics.csv")
    
    # Data quality report
    quality_report = {
        'total_days': len(daily_clean),
        'date_range': f"{daily_clean.index.min()} to {daily_clean.index.max()}",
        'columns': list(daily_clean.columns),
        'missing_after_imputation': daily_clean.isna().sum().to_dict()
    }
    
    with open(PROCESSED_DATA_PATH / 'data_quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    print(f"  Saved: data_quality_report.json")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return daily_clean

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    
    daily_data = run_preprocessing_pipeline()
    print("\nPreprocessed data shape:", daily_data.shape)
    print("\nColumns:", daily_data.columns.tolist())
    print("\nFirst 5 rows:")
    print(daily_data.head())
