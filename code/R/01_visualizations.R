# ============================================================================
# Advanced Time Series Analysis - Statistical Visualization in R
# Mi Band 7 Wearable Sensor Data
# MSc Data Science Assignment
#
# This script produces publication-quality visualizations using ggplot2
# 
# Required packages: ggplot2, forecast, tseries, lubridate, dplyr, scales
#
# Author: Data Science MSc Student
# Date: 2026-01
# ============================================================================

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# Install packages if needed
required_packages <- c("ggplot2", "forecast", "tseries", "lubridate", 
                       "dplyr", "scales", "gridExtra", "ggthemes")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Set working directory (adjust as needed)
# setwd("path/to/miband_analysis/code/R")

# Output directory
output_dir <- "../output/figures"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Custom theme for publication-quality plots
theme_publication <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size * 1.2, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = base_size, hjust = 0.5),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size * 0.9),
      legend.position = "bottom",
      legend.title = element_text(size = base_size),
      legend.text = element_text(size = base_size * 0.9),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "grey80", fill = NA, size = 0.5)
    )
}

# Color palette
colors <- c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B")

# ============================================================================
# DATA LOADING
# ============================================================================

load_daily_data <- function(filepath = "../data/processed/daily_combined.csv") {
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  df$date <- as.Date(df$date)
  return(df)
}

# ============================================================================
# TIME SERIES VISUALIZATION
# ============================================================================

#' Plot Raw Time Series with Trend
#' 
#' Creates a time series plot with LOESS smoothed trend line
#' and confidence band.
plot_time_series <- function(df, variable, title = NULL, y_label = NULL) {
  
  if (is.null(title)) title <- paste("Daily", variable, "Over Time")
  if (is.null(y_label)) y_label <- variable
  
  p <- ggplot(df, aes(x = date, y = .data[[variable]])) +
    geom_line(color = colors[1], alpha = 0.6, size = 0.5) +
    geom_point(color = colors[1], alpha = 0.4, size = 1) +
    geom_smooth(method = "loess", span = 0.3, 
                color = colors[2], fill = colors[2], alpha = 0.2) +
    scale_x_date(date_labels = "%b %d", date_breaks = "2 weeks") +
    scale_y_continuous(labels = scales::comma) +
    labs(
      title = title,
      subtitle = paste("n =", sum(!is.na(df[[variable]])), "observations"),
      x = "Date",
      y = y_label
    ) +
    theme_publication()
  
  return(p)
}

#' Plot Multiple Time Series
#' 
#' Creates faceted time series plot for multiple variables
plot_multiple_series <- function(df, variables, title = "Activity Metrics Over Time") {
  
  # Reshape data to long format
  df_long <- df %>%
    select(date, all_of(variables)) %>%
    tidyr::pivot_longer(cols = -date, names_to = "metric", values_to = "value")
  
  p <- ggplot(df_long, aes(x = date, y = value)) +
    geom_line(color = colors[1], alpha = 0.7) +
    geom_smooth(method = "loess", span = 0.3, color = colors[2], se = FALSE) +
    facet_wrap(~metric, scales = "free_y", ncol = 1) +
    scale_x_date(date_labels = "%b %d", date_breaks = "3 weeks") +
    scale_y_continuous(labels = scales::comma) +
    labs(title = title, x = "Date", y = "Value") +
    theme_publication()
  
  return(p)
}

# ============================================================================
# ROLLING STATISTICS VISUALIZATION
# ============================================================================

#' Plot Rolling Statistics
#' 
#' Visualizes rolling mean and standard deviation to assess
#' stationarity and variance stability
plot_rolling_stats <- function(df, variable, window = 7) {
  
  # Calculate rolling statistics
  df$rolling_mean <- zoo::rollmean(df[[variable]], k = window, fill = NA, align = "center")
  df$rolling_sd <- zoo::rollapply(df[[variable]], width = window, FUN = sd, 
                                   fill = NA, align = "center")
  
  # Plot 1: Original with rolling mean
  p1 <- ggplot(df, aes(x = date)) +
    geom_line(aes(y = .data[[variable]]), color = "grey60", alpha = 0.5) +
    geom_line(aes(y = rolling_mean), color = colors[1], size = 1) +
    scale_x_date(date_labels = "%b %d") +
    labs(
      title = paste(variable, "with", window, "-Day Rolling Mean"),
      x = "Date",
      y = variable
    ) +
    theme_publication()
  
  # Plot 2: Rolling standard deviation
  p2 <- ggplot(df, aes(x = date, y = rolling_sd)) +
    geom_line(color = colors[3], size = 1) +
    geom_hline(yintercept = sd(df[[variable]], na.rm = TRUE), 
               linetype = "dashed", color = "darkred") +
    scale_x_date(date_labels = "%b %d") +
    labs(
      title = paste(window, "-Day Rolling Standard Deviation"),
      subtitle = "Dashed line = overall standard deviation",
      x = "Date",
      y = "Rolling SD"
    ) +
    theme_publication()
  
  return(gridExtra::grid.arrange(p1, p2, ncol = 1))
}

# ============================================================================
# SEASONAL PATTERN VISUALIZATION
# ============================================================================

#' Plot Weekly Pattern
#' 
#' Bar chart showing average values by day of week
plot_weekly_pattern <- function(df, variable) {
  
  # Calculate day of week
  df$day_of_week <- factor(weekdays(df$date, abbreviate = TRUE),
                           levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
  
  # Aggregate by day
  weekly_summary <- df %>%
    group_by(day_of_week) %>%
    summarise(
      mean_val = mean(.data[[variable]], na.rm = TRUE),
      se_val = sd(.data[[variable]], na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(weekly_summary, aes(x = day_of_week, y = mean_val)) +
    geom_col(fill = colors[1], alpha = 0.8) +
    geom_errorbar(aes(ymin = mean_val - 1.96*se_val, 
                      ymax = mean_val + 1.96*se_val),
                  width = 0.2, color = "grey30") +
    scale_y_continuous(labels = scales::comma, expand = c(0, 0)) +
    labs(
      title = paste("Weekly Pattern:", variable),
      subtitle = "Mean \u00b1 95% CI",
      x = "Day of Week",
      y = paste("Mean", variable)
    ) +
    theme_publication()
  
  return(p)
}

#' Plot Weekend vs Weekday Comparison
plot_weekend_comparison <- function(df, variable) {
  
  df$is_weekend <- ifelse(weekdays(df$date) %in% c("Saturday", "Sunday"), 
                          "Weekend", "Weekday")
  
  p <- ggplot(df, aes(x = is_weekend, y = .data[[variable]], fill = is_weekend)) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.5) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
    scale_fill_manual(values = c("Weekday" = colors[1], "Weekend" = colors[2])) +
    scale_y_continuous(labels = scales::comma) +
    labs(
      title = paste(variable, ": Weekend vs Weekday"),
      x = "",
      y = variable
    ) +
    theme_publication() +
    theme(legend.position = "none")
  
  return(p)
}

# ============================================================================
# ACF/PACF VISUALIZATION
# ============================================================================

#' Plot ACF and PACF
#' 
#' Creates publication-quality ACF and PACF plots using forecast package
plot_acf_pacf <- function(df, variable, max_lag = 30) {
  
  series <- na.omit(df[[variable]])
  
  # Create time series object
  ts_data <- ts(series, frequency = 7)
  
  # Calculate ACF and PACF
  acf_vals <- Acf(ts_data, lag.max = max_lag, plot = FALSE)
  pacf_vals <- Pacf(ts_data, lag.max = max_lag, plot = FALSE)
  
  # Create data frames for plotting
  acf_df <- data.frame(
    lag = acf_vals$lag[-1],
    acf = acf_vals$acf[-1]
  )
  
  pacf_df <- data.frame(
    lag = pacf_vals$lag,
    pacf = pacf_vals$acf
  )
  
  # Critical value (95% CI)
  n <- length(series)
  ci <- 1.96 / sqrt(n)
  
  # ACF Plot
  p1 <- ggplot(acf_df, aes(x = lag, y = acf)) +
    geom_hline(yintercept = 0, color = "grey50") +
    geom_hline(yintercept = c(-ci, ci), linetype = "dashed", color = "blue", alpha = 0.7) +
    geom_segment(aes(xend = lag, yend = 0), color = colors[1], size = 0.8) +
    geom_point(color = colors[1], size = 2) +
    scale_x_continuous(breaks = seq(0, max_lag, by = 5)) +
    labs(
      title = paste("Autocorrelation Function (ACF):", variable),
      subtitle = "Dashed lines = 95% confidence bounds",
      x = "Lag (days)",
      y = "ACF"
    ) +
    theme_publication()
  
  # PACF Plot
  p2 <- ggplot(pacf_df, aes(x = lag, y = pacf)) +
    geom_hline(yintercept = 0, color = "grey50") +
    geom_hline(yintercept = c(-ci, ci), linetype = "dashed", color = "blue", alpha = 0.7) +
    geom_segment(aes(xend = lag, yend = 0), color = colors[2], size = 0.8) +
    geom_point(color = colors[2], size = 2) +
    scale_x_continuous(breaks = seq(0, max_lag, by = 5)) +
    labs(
      title = paste("Partial Autocorrelation Function (PACF):", variable),
      subtitle = "Dashed lines = 95% confidence bounds",
      x = "Lag (days)",
      y = "PACF"
    ) +
    theme_publication()
  
  return(gridExtra::grid.arrange(p1, p2, ncol = 1))
}

# ============================================================================
# DECOMPOSITION VISUALIZATION
# ============================================================================

#' Plot STL Decomposition
#' 
#' Creates a 4-panel plot showing observed, trend, seasonal, and remainder
plot_stl_decomposition <- function(df, variable, period = 7) {
  
  series <- na.omit(df[[variable]])
  dates <- df$date[!is.na(df[[variable]])]
  
  # Create time series and decompose
  ts_data <- ts(series, frequency = period)
  stl_result <- stl(ts_data, s.window = "periodic")
  
  # Create data frame with components
  decomp_df <- data.frame(
    date = dates,
    observed = as.numeric(ts_data),
    trend = as.numeric(stl_result$time.series[, "trend"]),
    seasonal = as.numeric(stl_result$time.series[, "seasonal"]),
    remainder = as.numeric(stl_result$time.series[, "remainder"])
  )
  
  # Create individual plots
  p1 <- ggplot(decomp_df, aes(x = date, y = observed)) +
    geom_line(color = colors[1]) +
    labs(title = "Observed", y = variable) +
    theme_publication() + theme(axis.title.x = element_blank())
  
  p2 <- ggplot(decomp_df, aes(x = date, y = trend)) +
    geom_line(color = colors[2], size = 1) +
    labs(title = "Trend", y = "") +
    theme_publication() + theme(axis.title.x = element_blank())
  
  p3 <- ggplot(decomp_df, aes(x = date, y = seasonal)) +
    geom_line(color = colors[3]) +
    labs(title = "Seasonal (Weekly)", y = "") +
    theme_publication() + theme(axis.title.x = element_blank())
  
  p4 <- ggplot(decomp_df, aes(x = date, y = remainder)) +
    geom_line(color = colors[4]) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    labs(title = "Remainder", x = "Date", y = "") +
    theme_publication()
  
  # Arrange plots
  combined <- gridExtra::grid.arrange(
    p1, p2, p3, p4, 
    ncol = 1,
    top = grid::textGrob(
      paste("STL Decomposition:", variable),
      gp = grid::gpar(fontsize = 14, fontface = "bold")
    )
  )
  
  return(combined)
}

# ============================================================================
# MAIN VISUALIZATION SCRIPT
# ============================================================================

generate_all_visualizations <- function() {
  
  cat("========================================\n")
  cat("Generating R Visualizations\n")
  cat("========================================\n\n")
  
  # Load data
  cat("Loading data...\n")
  df <- load_daily_data()
  
  cat(paste("Data loaded:", nrow(df), "observations\n"))
  cat(paste("Date range:", min(df$date), "to", max(df$date), "\n\n"))
  
  # Key variables
  variables <- c("steps", "calories", "distance")
  variables <- variables[variables %in% names(df)]
  
  for (var in variables) {
    cat(paste("Processing:", var, "\n"))
    
    # 1. Time series plot
    p <- plot_time_series(df, var)
    ggsave(file.path(output_dir, paste0("R_timeseries_", var, ".png")), 
           p, width = 12, height = 6, dpi = 150)
    
    # 2. Weekly pattern
    p <- plot_weekly_pattern(df, var)
    ggsave(file.path(output_dir, paste0("R_weekly_", var, ".png")), 
           p, width = 10, height = 6, dpi = 150)
    
    # 3. Weekend comparison
    p <- plot_weekend_comparison(df, var)
    ggsave(file.path(output_dir, paste0("R_weekend_", var, ".png")), 
           p, width = 8, height = 6, dpi = 150)
    
    # 4. ACF/PACF
    p <- plot_acf_pacf(df, var)
    ggsave(file.path(output_dir, paste0("R_acf_pacf_", var, ".png")), 
           p, width = 12, height = 10, dpi = 150)
    
    # 5. Rolling statistics
    p <- plot_rolling_stats(df, var, window = 7)
    ggsave(file.path(output_dir, paste0("R_rolling_", var, ".png")), 
           p, width = 12, height = 10, dpi = 150)
    
    # 6. STL decomposition
    tryCatch({
      p <- plot_stl_decomposition(df, var, period = 7)
      ggsave(file.path(output_dir, paste0("R_stl_", var, ".png")), 
             p, width = 12, height = 12, dpi = 150)
    }, error = function(e) {
      cat(paste("  Warning: STL decomposition failed for", var, "\n"))
    })
  }
  
  cat("\n========================================\n")
  cat("Visualization generation complete!\n")
  cat(paste("Output saved to:", output_dir, "\n"))
  cat("========================================\n")
}

# ============================================================================
# EXECUTION
# ============================================================================

# Uncomment to run:
# generate_all_visualizations()

cat("R Visualization Script Loaded\n")
cat("Run generate_all_visualizations() to create all plots\n")
