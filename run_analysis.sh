#!/bin/bash
# =============================================================================
# Mi Band 7 Time Series Analysis - Full Pipeline Runner
# Author: Momeni Gilles Christain
# =============================================================================

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MI BAND 7 TIME SERIES ANALYSIS - FULL PIPELINE"
echo "Author: Momeni Gilles Christain"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install Python Dependencies
# -----------------------------------------------------------------------------
echo "[1/5] Installing Python dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -q -r code/python/requirements.txt
elif command -v pip &> /dev/null; then
    pip install -q -r code/python/requirements.txt
else
    echo "ERROR: Neither uv nor pip found. Please install Python package manager."
    exit 1
fi
echo "      Python dependencies installed."
echo ""

# -----------------------------------------------------------------------------
# Step 2: Install R Dependencies (optional)
# -----------------------------------------------------------------------------
echo "[2/5] Installing R dependencies..."
if command -v Rscript &> /dev/null; then
    Rscript code/R/install_r_packages.R 2>/dev/null || echo "      (Some R packages may need manual installation)"
else
    echo "      R not found - skipping R package installation"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Run Data Preprocessing
# -----------------------------------------------------------------------------
echo "[3/5] Running data preprocessing..."
cd code/python
python 01_data_preprocessing.py
cd "$PROJECT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run Exploratory Analysis
# -----------------------------------------------------------------------------
echo "[4/5] Running exploratory data analysis..."
cd code/python
python 02_exploratory_analysis.py
cd "$PROJECT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Run Time Series Modeling
# -----------------------------------------------------------------------------
echo "[5/5] Running time series modeling..."
cd code/python
python 03_time_series_modeling.py
cd "$PROJECT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - data/processed/daily_combined.csv"
echo "  - data/processed/summary_statistics.csv"
echo "  - data/processed/data_quality_report.json"
echo "  - output/figures/*.png"
echo "  - output/reports/*.csv"
echo ""
echo "To generate R visualizations:"
echo "  cd code/R && Rscript 01_visualizations.R"
echo ""
echo "To compile the final report:"
echo "  cd code/R && Rscript -e \"rmarkdown::render('02_analysis_report.Rmd')\""
echo ""
