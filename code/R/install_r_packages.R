# R Package Dependencies
# Install with: Rscript install_r_packages.R

packages <- c(
  "ggplot2",      # Data visualization
  "forecast",     # Time series forecasting
  "tseries",      # Time series analysis
  "lubridate",    # Date/time handling
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "patchwork",    # Combining plots
  "knitr",        # Report generation
  "rmarkdown",    # R Markdown documents
  "scales",       # Scale functions for visualization
  "gridExtra",    # Grid graphics
  "zoo"           # Time series infrastructure
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
}

invisible(sapply(packages, install_if_missing))

cat("All R packages installed successfully!\n")
