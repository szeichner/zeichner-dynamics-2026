################################################################################
# PROCESS CO₂ FLUX ANALYSIS FROM SOIL MICROCOSMS
#LAST UPDATED: SSZ 2/11/26
################################################################################

# Set working directory
setwd("/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /data/Gas samples/UC Davis gas bench/")

# Clear environment (optional - comment out if you want to keep objects)
# rm(list = ls())

# Load required libraries
library(tidyverse)
library(readr)
library(scales)

# Set plot theme
theme_set(theme_bw(base_size = 12))

# Define custom colors
plot_colors <- c(
  "glucose" = "olivedrab", 
  "xylose" = "#E1AD01", 
  "cellulose" = "#874C62"
)

cat("Color scheme loaded:\n")
print(plot_colors)
cat("\nWorking directory:", getwd(), "\n")

# ============================================================================ #
# SUBSTRATE ADDITIONS ----
# ============================================================================ #

# Define substrate additions based on experimental design
substrate_additions <- tibble(
  treatment = c("glucose", "xylose", "cellulose"),
  
  # Total C added (mg C per g soil)
  mg_C_added_per_g_soil = c(
    6.940675119,    # glucose
    6.940675119,    # xylose  
    8.888888889     # cellulose
  ),
  
  # Percent 13C enrichment of added substrate
  pct_13C_enrichment = c(
    74.3,    # glucose is 74.3% 13C
    25.7,    # xylose is 25.7% 13C
    97.0     # cellulose is 97% 13C
  )
)

# Calculate 13C added
substrate_additions <- substrate_additions %>%
  mutate(
    # Amount of 13C added = total C × (% 13C / 100)
    mg_13C_added_per_g_soil = mg_C_added_per_g_soil * (pct_13C_enrichment / 100)
  )

cat("\n=== SUBSTRATE ADDITIONS ===\n")
print(substrate_additions)
cat("\n")

# ============================================================================ #
# LOAD AND EXPLORE DATA ----
# ============================================================================ #

# Read the data
data <- read_csv("sugar_experiment_CO2_mgC_withJarInfo.csv", show_col_types = FALSE)

# Quick exploration
cat("\n=== DATA OVERVIEW ===\n")
cat("Dimensions:", nrow(data), "rows ×", ncol(data), "columns\n\n")

cat("Treatments:\n")
print(table(data$treatment))

cat("\nTime points:\n")
print(sort(unique(data$`time point`)))

# ============================================================================ #
# DATA PREPARATION ----
# ============================================================================ #

# Clean and prepare data
data <- data %>%
  mutate(
    # Convert scientific notation to numeric
    converted_CO2_mg = as.numeric(`Converted CO2 (mg)`),
    
    # Clean column names
    jar_number = `jar number`,
    grams_in_jar = `grams in jar`,
    time_point = `time point`,
    delta13C = `δ13CVPDB (‰)`,
    at_pct_13C = `at-% 13C`,
    
    # Calculate carbon masses
    # CO2 MW = 44.01, C AW = 12.01
    mg_total_C = converted_CO2_mg * (12.01 / 44.01),
    mg_13C = mg_total_C * (at_pct_13C/100)
  )

cat("\n=== CARBON MASS CALCULATIONS COMPLETE ===\n")
cat("Summary of total C (mg):\n")
print(summary(data$mg_total_C))
cat("\nSummary of 13C (mg):\n")
print(summary(data$mg_13C))

# ============================================================================ #
# IDENTIFY MEASUREMENT TYPES ----
# ============================================================================ #

# Define paired time points (T0/T1 pairs)
paired_times <- list(
  c(71, 72),
  c(167, 168),
  c(335, 336),
  c(671, 672)
)

# Add measurement type identifiers
data <- data %>%
  mutate(
    is_paired = time_point %in% unlist(paired_times),
    pair_id = case_when(
      time_point %in% c(71, 72) ~ "72h",
      time_point %in% c(167, 168) ~ "168h",
      time_point %in% c(335, 336) ~ "336h",
      time_point %in% c(671, 672) ~ "672h",
      TRUE ~ NA_character_
    ),
    measurement_type = case_when(
      time_point %in% c(71, 167, 335, 671) ~ "T0",
      time_point %in% c(72, 168, 336, 672) ~ "T1",
      TRUE ~ "single"
    )
  )

# Identify jar-tracked measurements
jar_tracking <- data %>%
  filter(time_point %in% c(1, 4, 8)) %>%
  group_by(jar_number, treatment, replicate) %>%
  summarise(
    n_measurements = n(),
    time_points = paste(sort(unique(time_point)), collapse = ","),
    .groups = "drop"
  ) %>%
  mutate(is_tracked = n_measurements > 1)

# Add tracking info to main data
data <- data %>%
  left_join(
    jar_tracking %>% select(jar_number, is_tracked),
    by = "jar_number"
  ) %>%
  mutate(is_tracked = replace_na(is_tracked, FALSE))

cat("\n=== MEASUREMENT TYPE IDENTIFICATION ===\n")
cat("Paired measurements:", sum(data$is_paired), "observations\n")
cat("Jar-tracked measurements:", sum(data$is_tracked & data$time_point %in% c(1,4,8)), "observations\n")

# ============================================================================ #
# FILTER TO TREATMENTS OF INTEREST ----
# ============================================================================ #

# Select which treatments to analyze
treatments_to_analyze <- c("glucose", "xylose", "cellulose")

data_analysis <- data %>%
  filter(treatment %in% treatments_to_analyze)

cat("\n=== FILTERED TO ANALYSIS TREATMENTS ===\n")
cat("Treatments:", paste(treatments_to_analyze, collapse = ", "), "\n")
cat("Total observations:", nrow(data_analysis), "\n")

# ============================================================================ #
# CALCULATE PAIRED MEASUREMENT FLUXES (71-72, 167-168, 335-336, 671-672) ----
# ============================================================================ #

paired_fluxes <- data_analysis %>%
  filter(is_paired) %>%
  arrange(jar_number, time_point) %>%
  group_by(jar_number, treatment, replicate, pair_id) %>%
  filter(n() == 2) %>%
  summarise(
    time_T0 = time_point[measurement_type == "T0"],
    time_T1 = time_point[measurement_type == "T1"],
    mg_C_T0 = mg_total_C[measurement_type == "T0"],
    mg_C_T1 = mg_total_C[measurement_type == "T1"],
    mg_13C_T0 = mg_13C[measurement_type == "T0"],
    mg_13C_T1 = mg_13C[measurement_type == "T1"],
    grams_soil = first(grams_in_jar),
    at_pct_13C_avg = mean(at_pct_13C),
    .groups = "drop"
  ) %>%
  mutate(
    delta_t_h = 1,
    flux_total_C = (mg_C_T1 - mg_C_T0) / (grams_soil * delta_t_h),
    flux_13C = (mg_13C_T1 - mg_13C_T0) / (grams_soil * delta_t_h),
    time_point_plot = time_T1
  )

# Summary statistics
paired_flux_summary <- paired_fluxes %>%
  group_by(treatment, time_point_plot) %>%
  summarise(
    n_reps = n(),
    mean_flux_C = mean(flux_total_C, na.rm = TRUE),
    se_flux_C = sd(flux_total_C, na.rm = TRUE) / sqrt(n()),
    mean_flux_13C = mean(flux_13C, na.rm = TRUE),
    se_flux_13C = sd(flux_13C, na.rm = TRUE) / sqrt(n()),
    mean_at_pct_13C = mean(at_pct_13C_avg, na.rm = TRUE),
    se_at_pct_13C = sd(at_pct_13C_avg, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ratio_13C_to_C = mean_flux_13C / mean_flux_C,
    se_ratio = ratio_13C_to_C * sqrt((se_flux_13C/mean_flux_13C)^2 + 
                                       (se_flux_C/mean_flux_C)^2)
  )

cat("\n=== PAIRED FLUX CALCULATIONS ===\n")
cat("Number of paired flux measurements:", nrow(paired_fluxes), "\n")

# ============================================================================ #
# CALCULATE JAR-TRACKED FLUXES (1, 4, 8 hours) ----
# ============================================================================ #

tracked_early <- data_analysis %>%
  filter(time_point %in% c(1, 4, 8), is_tracked) %>%
  arrange(jar_number, time_point) %>%
  group_by(jar_number, treatment, replicate) %>%
  mutate(
    prev_time = lag(time_point),
    prev_mg_C = lag(mg_total_C),
    prev_mg_13C = lag(mg_13C),
    delta_t_h = time_point - prev_time,
    flux_total_C = (mg_total_C - prev_mg_C) / (grams_in_jar * delta_t_h),
    flux_13C = (mg_13C - prev_mg_13C) / (grams_in_jar * delta_t_h)
  ) %>%
  filter(!is.na(flux_total_C)) %>%
  ungroup() %>%
  select(jar_number, treatment, replicate, time_point, grams_in_jar,
         flux_total_C, flux_13C, at_pct_13C) %>%
  rename(time_point_plot = time_point)

# Summary statistics
tracked_early_summary <- tracked_early %>%
  group_by(treatment, time_point_plot) %>%
  summarise(
    n_reps = n(),
    mean_flux_C = mean(flux_total_C, na.rm = TRUE),
    se_flux_C = sd(flux_total_C, na.rm = TRUE) / sqrt(n()),
    mean_flux_13C = mean(flux_13C, na.rm = TRUE),
    se_flux_13C = sd(flux_13C, na.rm = TRUE) / sqrt(n()),
    mean_at_pct_13C = mean(at_pct_13C, na.rm = TRUE),
    se_at_pct_13C = sd(at_pct_13C, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ratio_13C_to_C = mean_flux_13C / mean_flux_C,
    se_ratio = ratio_13C_to_C * sqrt((se_flux_13C/mean_flux_13C)^2 + 
                                       (se_flux_C/mean_flux_C)^2)
  )

cat("\n=== JAR-TRACKED FLUX CALCULATIONS ===\n")
cat("Number of tracked flux measurements:", nrow(tracked_early), "\n")

# ============================================================================ #
# CALCULATE CROSS-SECTIONAL FLUXES (0, 1, 2, 24 hours) ----
# ============================================================================ #

# Identify cross-sectional time points 
cross_sectional_times <- data_analysis %>%
  filter(!is_paired) %>%  # Exclude paired measurements only
  pull(time_point) %>%
  unique() %>%
  sort()

cat("\n=== CROSS-SECTIONAL MEASUREMENTS ===\n")
cat("Time points:", paste(cross_sectional_times, collapse = ", "), "\n")

# Calculate population means 
# This includes all jars regardless of tracking status
cross_sectional_means <- data_analysis %>%
  filter(time_point %in% c(0, cross_sectional_times)) %>%
  group_by(treatment, time_point) %>%
  summarise(
    n_reps = n(),
    mean_mg_C = mean(mg_total_C, na.rm = TRUE),
    se_mg_C = sd(mg_total_C, na.rm = TRUE) / sqrt(n()),
    mean_mg_13C = mean(mg_13C, na.rm = TRUE),
    se_mg_13C = sd(mg_13C, na.rm = TRUE) / sqrt(n()),
    mean_grams = mean(grams_in_jar, na.rm = TRUE),
    mean_at_pct_13C = mean(at_pct_13C, na.rm = TRUE),
    se_at_pct_13C = sd(at_pct_13C, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  arrange(treatment, time_point)

# Calculate fluxes from population means
cross_sectional_fluxes <- cross_sectional_means %>%
  group_by(treatment) %>%
  arrange(time_point) %>%
  mutate(
    prev_time = lag(time_point),
    prev_mg_C = lag(mean_mg_C),
    prev_se_C = lag(se_mg_C),
    prev_mg_13C = lag(mean_mg_13C),
    prev_se_13C = lag(se_mg_13C),
    delta_t_h = time_point - prev_time,
    mean_flux_C = (mean_mg_C - prev_mg_C) / (mean_grams * delta_t_h),
    mean_flux_13C = (mean_mg_13C - prev_mg_13C) / (mean_grams * delta_t_h),
    se_flux_C = sqrt(se_mg_C^2 + prev_se_C^2) / (mean_grams * delta_t_h),
    se_flux_13C = sqrt(se_mg_13C^2 + prev_se_13C^2) / (mean_grams * delta_t_h)
  ) %>%
  filter(!is.na(mean_flux_C)) %>%
  ungroup() %>%
  select(treatment, time_point, n_reps, mean_flux_C, se_flux_C, 
         mean_flux_13C, se_flux_13C, mean_at_pct_13C, se_at_pct_13C) %>%
  rename(time_point_plot = time_point) %>%
  mutate(
    ratio_13C_to_C = mean_flux_13C / mean_flux_C,
    se_ratio = ratio_13C_to_C * sqrt((se_flux_13C/mean_flux_13C)^2 + 
                                       (se_flux_C/mean_flux_C)^2)
  )

cat("Number of cross-sectional flux calculations:", nrow(cross_sectional_fluxes), "\n")
cat("Cross-sectional time points with flux data:\n")
print(sort(unique(cross_sectional_fluxes$time_point_plot)))

# ============================================================================ #
# COMBINE ALL FLUX CALCULATIONS ----
# ============================================================================ #

all_fluxes <- bind_rows(
  paired_flux_summary %>% mutate(measurement_type = "paired"),
  tracked_early_summary %>% mutate(measurement_type = "tracked"),
  cross_sectional_fluxes %>% mutate(measurement_type = "cross-sectional")
) %>%
  arrange(treatment, time_point_plot)

cat("\n=== ALL FLUX CALCULATIONS COMBINED ===\n")
cat("Total data points:", nrow(all_fluxes), "\n")
cat("  Paired:", sum(all_fluxes$measurement_type == "paired"), "\n")
cat("  Tracked:", sum(all_fluxes$measurement_type == "tracked"), "\n")
cat("  Cross-sectional:", sum(all_fluxes$measurement_type == "cross-sectional"), "\n")
cat("\nAll time points included:\n")
print(sort(unique(all_fluxes$time_point_plot)))

# ============================================================================ #
# DIAGNOSTIC: CHECK FOR NEGATIVE FLUXES ----
# ============================================================================ #

cat("\n=== DIAGNOSTIC: CHECKING FOR NEGATIVE FLUXES ===\n")

negative_fluxes <- all_fluxes %>%
  filter(mean_flux_C < 0 | mean_flux_13C < 0)

if(nrow(negative_fluxes) > 0) {
  cat("⚠️  Found", nrow(negative_fluxes), "negative flux values:\n")
  print(negative_fluxes %>% 
          select(treatment, time_point_plot, measurement_type, 
                 mean_flux_C, mean_flux_13C))
  cat("\nNOTE: Negative fluxes are RETAINED (not set to zero)\n")
} else {
  cat("✓ No negative fluxes detected\n")
}

# ============================================================================ #
# CALCULATE CUMULATIVE RESPIRATION ----
# ============================================================================ #

cumulative_resp <- all_fluxes %>%
  group_by(treatment) %>%
  arrange(time_point_plot) %>%
  mutate(
    time_start = lag(time_point_plot, default = 0),
    time_end = time_point_plot,
    delta_t = time_end - time_start,
    avg_flux_C = (mean_flux_C + lag(mean_flux_C, default = 0)) / 2,
    avg_flux_13C = (mean_flux_13C + lag(mean_flux_13C, default = 0)) / 2,
    cum_C_interval = avg_flux_C * delta_t,
    cum_13C_interval = avg_flux_13C * delta_t
  ) %>%
  mutate(
    cumulative_mg_C = cumsum(cum_C_interval),
    cumulative_mg_13C = cumsum(cum_13C_interval),
    se_cumulative_C = sqrt(cumsum((se_flux_C * delta_t)^2)),
    se_cumulative_13C = sqrt(cumsum((se_flux_13C * delta_t)^2))
  ) %>%
  ungroup()

cat("\n=== CUMULATIVE RESPIRATION CALCULATED ===\n")
cat("Final cumulative values by treatment:\n")
cumulative_resp %>%
  group_by(treatment) %>%
  filter(time_point_plot == max(time_point_plot)) %>%
  select(treatment, time_point_plot, cumulative_mg_C, cumulative_mg_13C) %>%
  print()

# ============================================================================ #
# NORMALIZE BY SUBSTRATE ADDITION ----
# ============================================================================ #

cat("\n=== NORMALIZING BY SUBSTRATE ADDITION ===\n")

# Add substrate info to flux data
all_fluxes_normalized <- all_fluxes %>%
  left_join(substrate_additions, by = "treatment") %>%
  mutate(
    # Flux normalized by TOTAL C added (for total C flux)
    flux_C_normalized = mean_flux_C / mg_C_added_per_g_soil,
    se_flux_C_normalized = se_flux_C / mg_C_added_per_g_soil,
    
    # Flux normalized by 13C added (for 13C flux)
    flux_13C_normalized = mean_flux_13C / mg_13C_added_per_g_soil,
    se_flux_13C_normalized = se_flux_13C / mg_13C_added_per_g_soil
  )

# Add to cumulative data
cumulative_resp_normalized <- cumulative_resp %>%
  left_join(substrate_additions, by = "treatment") %>%
  mutate(
    # Percent of added substrate respired
    pct_C_respired = (cumulative_mg_C / mg_C_added_per_g_soil) * 100,
    se_pct_C_respired = (se_cumulative_C / mg_C_added_per_g_soil) * 100,
    
    # Percent of added 13C recovered
    pct_13C_recovered = (cumulative_mg_13C / mg_13C_added_per_g_soil) * 100,
    se_pct_13C_recovered = (se_cumulative_13C / mg_13C_added_per_g_soil) * 100
  )

cat("Normalization complete!\n")
cat("\nFinal % of substrate respired:\n")
cumulative_resp_normalized %>%
  group_by(treatment) %>%
  filter(time_point_plot == max(time_point_plot)) %>%
  select(treatment, time_point_plot, pct_C_respired, pct_13C_recovered) %>%
  print()

# ============================================================================ #
# OUTLIER DETECTION ----
# ============================================================================ #

outlier_threshold <- 3

outlier_check <- data_analysis %>%
  group_by(treatment, time_point) %>%
  mutate(
    mean_C = mean(mg_total_C, na.rm = TRUE),
    sd_C = sd(mg_total_C, na.rm = TRUE),
    z_score_C = abs((mg_total_C - mean_C) / sd_C),
    is_outlier_C = z_score_C > outlier_threshold,
    mean_13C = mean(mg_13C, na.rm = TRUE),
    sd_13C = sd(mg_13C, na.rm = TRUE),
    z_score_13C = abs((mg_13C - mean_13C) / sd_13C),
    is_outlier_13C = z_score_13C > outlier_threshold
  ) %>%
  ungroup()

outliers <- outlier_check %>%
  filter(is_outlier_C | is_outlier_13C) %>%
  select(`Sample ID`, jar_number, treatment, time_point, replicate,
         mg_total_C, z_score_C, is_outlier_C,
         mg_13C, z_score_13C, is_outlier_13C)

cat("\n=== OUTLIER DETECTION ===\n")
cat("Threshold:", outlier_threshold, "SD\n")
if(nrow(outliers) > 0) {
  cat("Found", nrow(outliers), "potential outliers:\n")
  print(outliers)
} else {
  cat("No outliers detected\n")
}

# ============================================================================ #
# CREATE SUMMARY TABLES ----
# ============================================================================ #

summary_table <- all_fluxes %>%
  left_join(
    cumulative_resp %>% select(treatment, time_point_plot, 
                               cumulative_mg_C, se_cumulative_C,
                               cumulative_mg_13C, se_cumulative_13C),
    by = c("treatment", "time_point_plot")
  ) %>%
  select(
    treatment, time_point = time_point_plot, measurement_type, n_reps,
    mean_flux_C, se_flux_C,
    mean_flux_13C, se_flux_13C,
    ratio_13C_to_C, se_ratio,
    mean_at_pct_13C, se_at_pct_13C,
    cumulative_mg_C, se_cumulative_C,
    cumulative_mg_13C, se_cumulative_13C
  ) %>%
  mutate(across(where(is.numeric), ~round(., 6)))

# Normalized summary table
normalized_summary_table <- cumulative_resp_normalized %>%
  select(treatment, time_point_plot, measurement_type,
         mg_C_added_per_g_soil, mg_13C_added_per_g_soil, pct_13C_enrichment,
         cumulative_mg_C, se_cumulative_C,
         pct_C_respired, se_pct_C_respired,
         cumulative_mg_13C, se_cumulative_13C,
         pct_13C_recovered, se_pct_13C_recovered) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n=== SUMMARY TABLES CREATED ===\n")

# ============================================================================ #
# PLOT 1: TOTAL CO2-C FLUX RATE----
# ============================================================================ #

p1 <- ggplot(all_fluxes, aes(x = time_point_plot, y = mean_flux_C, 
                             color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  # Open circles for cross-sectional (population averages)
  geom_point(data = all_fluxes %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  # Filled circles for paired/tracked (jar-specific)
  geom_point(data = all_fluxes %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = mean_flux_C - se_flux_C, 
                     ymax = mean_flux_C + se_flux_C)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Total CO2-C Flux Rate Over Time",
    subtitle = "Open circles = population averages; Filled circles = jar-specific measurements",
    x = "Time (hours)",
    y = expression("CO"[2]*"-C flux rate (mg C g"^-1*" soil h"^-1*")"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p1)

# ============================================================================ #
# PLOT 2: 13C FLUX RATE ----
# ============================================================================ #

p2 <- ggplot(all_fluxes, aes(x = time_point_plot, y = mean_flux_13C, 
                             color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = all_fluxes %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = all_fluxes %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = mean_flux_13C - se_flux_13C, 
                     ymax = mean_flux_13C + se_flux_13C)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "¹³C Flux Rate Over Time",
    subtitle = "Open circles = population averages; Filled circles = jar-specific measurements",
    x = "Time (hours)",
    y = expression("¹³C flux rate (mg ¹³C g"^-1*" soil h"^-1*")"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p2)

# ============================================================================ #
# PLOT 3: RATIO OF 13C TO TOTAL C FLUX ----
# ============================================================================ #

p3 <- ggplot(all_fluxes, aes(x = time_point_plot, y = ratio_13C_to_C, 
                             color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = all_fluxes %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = all_fluxes %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = ratio_13C_to_C - se_ratio, 
                     ymax = ratio_13C_to_C + se_ratio)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(
    title = "Ratio of ¹³C Flux to Total C Flux Over Time",
    x = "Time (hours)",
    y = "¹³C flux / Total C flux (%)",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p3)

# ============================================================================ #
# PLOT 4: CUMULATIVE C RESPIRED ----
# ============================================================================ #

p4 <- ggplot(cumulative_resp, aes(x = time_point_plot, y = cumulative_mg_C, 
                                  color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = cumulative_resp %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = cumulative_resp %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = cumulative_mg_C - se_cumulative_C, 
                     ymax = cumulative_mg_C + se_cumulative_C)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Cumulative CO₂-C Respired Over Time",
    x = "Time (hours)",
    y = expression("Cumulative C respired (mg C g"^-1*" soil)"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p4)

# ============================================================================ #
# PLOT 5: CUMULATIVE 13C RESPIRED ----
# ============================================================================ #

p5 <- ggplot(cumulative_resp, aes(x = time_point_plot, y = cumulative_mg_13C, 
                                  color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = cumulative_resp %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = cumulative_resp %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = cumulative_mg_13C - se_cumulative_13C, 
                     ymax = cumulative_mg_13C + se_cumulative_13C)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Cumulative ¹³C Respired Over Time",
    x = "Time (hours)",
    y = expression("Cumulative ¹³C respired (mg ¹³C g"^-1*" soil)"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p5)

# ============================================================================ #
# PLOT 6: ATOMIC % 13C OVER TIME ----
# ============================================================================ #

at_pct_data <- data_analysis %>%
  group_by(treatment, time_point) %>%
  summarise(
    mean_at_pct = mean(at_pct_13C, na.rm = TRUE),
    se_at_pct = sd(at_pct_13C, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p6 <- ggplot(at_pct_data, aes(x = time_point, y = mean_at_pct, 
                              color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_linerange(aes(ymin = mean_at_pct - se_at_pct, 
                     ymax = mean_at_pct + se_at_pct)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Atomic % ¹³C Over Time",
    x = "Time (hours)",
    y = "Atomic % ¹³C (at-%)",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p6)

# ============================================================================ #
# PLOT 7: NORMALIZED TOTAL C FLUX RATE ----
# ============================================================================ #

p7 <- ggplot(all_fluxes_normalized, 
             aes(x = time_point_plot, y = flux_C_normalized, 
                 color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = all_fluxes_normalized %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = all_fluxes_normalized %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = flux_C_normalized - se_flux_C_normalized, 
                     ymax = flux_C_normalized + se_flux_C_normalized)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Normalized Total CO2-C Flux Rate Over Time",
    subtitle = "Flux per mg substrate-C added",
    x = "Time (hours)",
    y = expression("Flux rate per mg C added (h"^-1*")"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p7)

# ============================================================================ #
# PLOT 8: NORMALIZED 13C FLUX RATE ----
# ============================================================================ #

p8 <- ggplot(all_fluxes_normalized, 
             aes(x = time_point_plot, y = flux_13C_normalized, 
                 color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = all_fluxes_normalized %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = all_fluxes_normalized %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = flux_13C_normalized - se_flux_13C_normalized, 
                     ymax = flux_13C_normalized + se_flux_13C_normalized)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Normalized ¹³C Flux Rate Over Time",
    subtitle = "Flux per mg ¹³C added (accounts for different ¹³C enrichment)",
    x = "Time (hours)",
    y = expression("¹³C flux rate per mg ¹³C added (h"^-1*")"),
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p8)

# ============================================================================ #
# PLOT 9: PERCENT OF ADDED C RESPIRED ----
# ============================================================================ #

p9 <- ggplot(cumulative_resp_normalized, 
             aes(x = time_point_plot, y = pct_C_respired, 
                 color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = cumulative_resp_normalized %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = cumulative_resp_normalized %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = pct_C_respired - se_pct_C_respired, 
                     ymax = pct_C_respired + se_pct_C_respired)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  scale_y_continuous(limits = c(NA, NA)) +  # Allow negative values to show
  labs(
    title = "Percent of Added Substrate-C Respired Over Time",
    subtitle = "Shows substrate lability/degradability",
    x = "Time (hours)",
    y = "% of Added C Respired",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p9)

# ============================================================================ #
# PLOT 10: PERCENT OF ADDED 13C RECOVERED ----
# ============================================================================ #

p10 <- ggplot(cumulative_resp_normalized, 
              aes(x = time_point_plot, y = pct_13C_recovered, 
                  color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(data = cumulative_resp_normalized %>% filter(measurement_type == "cross-sectional"),
             size = 3, shape = 21, fill = "white", stroke = 1.5) +
  geom_point(data = cumulative_resp_normalized %>% filter(measurement_type != "cross-sectional"),
             size = 3, shape = 16) +
  geom_linerange(aes(ymin = pct_13C_recovered - se_pct_13C_recovered, 
                     ymax = pct_13C_recovered + se_pct_13C_recovered)) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  scale_y_continuous(limits = c(NA, NA)) +  # Allow negative values to show
  labs(
    title = "Percent of Added ¹³C Recovered as CO2 Over Time",
    subtitle = "Shows substrate-specific mineralization",
    x = "Time (hours)",
    y = "% of Added ¹³C Recovered",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p10)

# ============================================================================ #
# PLOT 11: FINAL SUBSTRATE MINERALIZATION COMPARISON ----
# ============================================================================ #

final_pct <- cumulative_resp_normalized %>%
  group_by(treatment) %>%
  filter(time_point_plot == max(time_point_plot)) %>%
  select(treatment, pct_C_respired, se_pct_C_respired, 
         pct_13C_recovered, se_pct_13C_recovered)

p11 <- ggplot(final_pct, aes(x = treatment, y = pct_C_respired, fill = treatment)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = pct_C_respired - se_pct_C_respired,
                    ymax = pct_C_respired + se_pct_C_respired),
                width = 0.2) +
  geom_text(aes(label = sprintf("%.1f%%", pct_C_respired)),
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = plot_colors) +
  labs(
    title = "Total Substrate Mineralization by Treatment",
    subtitle = paste("After", max(cumulative_resp_normalized$time_point_plot), "hours incubation"),
    x = "Treatment",
    y = "% of Added Substrate-C Respired",
    fill = "Treatment"
  ) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank()
  )

print(p11)

# ============================================================================ #
# CALCULATE NORMALIZED ATOMIC % 13C ----
# ============================================================================ #

# Approach: Calculate excess 13C above natural abundance, 
# normalized by the 13C enrichment of the substrate added

# Natural abundance of 13C
natural_abundance_13C <- 1.1  # atomic %

# Join atomic % data with substrate additions
at_pct_normalized_data <- at_pct_data %>%
  left_join(substrate_additions, by = "treatment") %>%
  mutate(
    # Excess 13C above natural abundance
    excess_at_pct_13C = mean_at_pct - natural_abundance_13C,
    se_excess_at_pct = se_at_pct,  # SE doesn't change with subtraction of constant
    
    # Normalize by substrate enrichment
    # This shows what % of the theoretical maximum enrichment is achieved
    normalized_at_pct_13C = (excess_at_pct_13C / (pct_13C_enrichment - natural_abundance_13C)) * 100,
    se_normalized_at_pct = (se_excess_at_pct / (pct_13C_enrichment - natural_abundance_13C)) * 100
  )

cat("\n=== NORMALIZED ATOMIC % 13C CALCULATED ===\n")
cat("This shows the % of maximum possible 13C enrichment achieved in respired CO2\n")
cat("(accounts for different substrate 13C enrichment levels)\n\n")

# Show final values
at_pct_normalized_data %>%
  group_by(treatment) %>%
  filter(time_point == max(time_point)) %>%
  select(treatment, time_point, mean_at_pct, excess_at_pct_13C, 
         normalized_at_pct_13C, pct_13C_enrichment) %>%
  print()

# ============================================================================ #
# PLOT 12: NORMALIZED ATOMIC % 13C ----
# ============================================================================ #

p12 <- ggplot(at_pct_normalized_data, 
              aes(x = time_point, y = normalized_at_pct_13C, 
                  color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_linerange(aes(ymin = normalized_at_pct_13C - se_normalized_at_pct, 
                     ymax = normalized_at_pct_13C + se_normalized_at_pct)) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Normalized Atomic % ¹³C in Respired CO₂",
    subtitle = "Normalized by substrate ¹³C enrichment (100% = all CO₂ from labeled substrate)",
    x = "Time (hours)",
    y = "% of Maximum ¹³C Enrichment",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p12)

# ============================================================================ #
# ALTERNATIVE PLOT 12B: EXCESS ATOMIC % 13C (not normalized) ----
# ============================================================================ #

# This shows absolute excess 13C above natural abundance
p12b <- ggplot(at_pct_normalized_data, 
               aes(x = time_point, y = excess_at_pct_13C, 
                   color = treatment, group = treatment)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_linerange(aes(ymin = excess_at_pct_13C - se_excess_at_pct, 
                     ymax = excess_at_pct_13C + se_excess_at_pct)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_manual(values = plot_colors) +
  scale_x_continuous(trans = "log10", 
                     breaks = c(1, 2, 4, 8, 24, 72, 168, 336, 672)) +
  labs(
    title = "Excess Atomic % ¹³C Above Natural Abundance",
    subtitle = "Shows ¹³C enrichment from labeled substrate (natural abundance = 1.1 at-%)",
    x = "Time (hours)",
    y = "Excess Atomic % ¹³C (at-%)",
    color = "Treatment"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p12b)

# ============================================================================ #
# SAVE NEW PLOTS ----
# ============================================================================ #

if(save_outputs) {
  
  cat("\n✓ Saved normalized atomic % 13C plots\n")
}

# ============================================================================ #
# SAVE OUTPUTS ----
# ============================================================================ #

save_outputs <- TRUE

if(save_outputs) {
  
  # Create output directory if it doesn't exist
  if(!dir.exists("outputs")) dir.create("outputs")
  
  # Save tables
  write_csv(summary_table, "outputs/flux_summary_table.csv")
  write_csv(normalized_summary_table, "outputs/normalized_summary_table.csv")
  
  if(nrow(outliers) > 0) {
    write_csv(outliers, "outputs/outliers_identified.csv")
  }
  
  # Save all plots
  ggsave("outputs/plot01_total_C_flux_rate.png", p1, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot02_13C_flux_rate.png", p2, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot03_13C_to_C_ratio.png", p3, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot04_cumulative_C.png", p4, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot05_cumulative_13C.png", p5, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot06_atomic_pct_13C.png", p6, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot07_normalized_total_C_flux.png", p7, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot08_normalized_13C_flux.png", p8, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot09_percent_C_respired.png", p9, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot10_percent_13C_recovered.png", p10, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot11_final_comparison.png", p11, 
         width = 8, height = 6, dpi = 300)
  ggsave("outputs/plot12_normalized_atomic_pct_13C.png", p12, 
         width = 10, height = 6, dpi = 300)
  ggsave("outputs/plot12b_excess_atomic_pct_13C.png", p12b, 
         width = 10, height = 6, dpi = 300)

  
  # Save workspace
  save.image("outputs/CO2_flux_analysis_workspace.RData")
  
  cat("\n=== OUTPUTS SAVED ===\n")
  cat("Files saved to 'outputs/' directory\n")
  cat("  - 11 plots (PNG, 300 DPI)\n")
  cat("  - 2 summary tables (CSV)\n")
  if(nrow(outliers) > 0) cat("  - Outliers table (CSV)\n")
  cat("  - Workspace file (RData)\n")
}

# ============================================================================ #
# INTERACTIVE EXPLORATION TOOLS ----
# ============================================================================ #

# Quick filter functions
filter_treatment <- function(df, treat) {
  df %>% filter(treatment == treat)
}

filter_time <- function(df, min_time, max_time) {
  df %>% filter(time_point >= min_time & time_point <= max_time)
}

get_stats <- function(treat, time) {
  all_fluxes %>%
    filter(treatment == treat, time_point_plot == time) %>%
    select(treatment, time_point_plot, measurement_type, mean_flux_C, se_flux_C, 
           mean_flux_13C, se_flux_13C, ratio_13C_to_C)
}

# Plot single treatment
plot_single_treatment <- function(treat, plot_type = "flux_C") {
  df <- all_fluxes %>% filter(treatment == treat)
  
  if(plot_type == "flux_C") {
    p <- ggplot(df, aes(x = time_point_plot, y = mean_flux_C)) +
      geom_line(linewidth = 1, color = plot_colors[treat]) +
      geom_point(data = df %>% filter(measurement_type == "cross-sectional"),
                 size = 3, shape = 21, fill = "white", stroke = 1.5,
                 color = plot_colors[treat]) +
      geom_point(data = df %>% filter(measurement_type != "cross-sectional"),
                 size = 3, shape = 16, color = plot_colors[treat]) +
      geom_linerange(aes(ymin = mean_flux_C - se_flux_C, 
                         ymax = mean_flux_C + se_flux_C),
                     color = plot_colors[treat]) +
      scale_x_continuous(trans = "log10") +
      labs(title = paste("Total C Flux -", treat),
           x = "Time (hours)",
           y = expression("Flux (mg C g"^-1*" h"^-1*")"))
  } else if(plot_type == "flux_13C") {
    p <- ggplot(df, aes(x = time_point_plot, y = mean_flux_13C)) +
      geom_line(linewidth = 1, color = plot_colors[treat]) +
      geom_point(data = df %>% filter(measurement_type == "cross-sectional"),
                 size = 3, shape = 21, fill = "white", stroke = 1.5,
                 color = plot_colors[treat]) +
      geom_point(data = df %>% filter(measurement_type != "cross-sectional"),
                 size = 3, shape = 16, color = plot_colors[treat]) +
      geom_linerange(aes(ymin = mean_flux_13C - se_flux_13C, 
                         ymax = mean_flux_13C + se_flux_13C),
                     color = plot_colors[treat]) +
      scale_x_continuous(trans = "log10") +
      labs(title = paste("13C Flux -", treat),
           x = "Time (hours)",
           y = expression("Flux (mg ¹³C g"^-1*" h"^-1*")"))
  }
  
  return(p)
}

# Compare two treatments
compare_treatments <- function(treat1, treat2) {
  df <- all_fluxes %>% filter(treatment %in% c(treat1, treat2))
  
  ggplot(df, aes(x = time_point_plot, y = mean_flux_C, 
                 color = treatment, group = treatment)) +
    geom_line(linewidth = 1) +
    geom_point(data = df %>% filter(measurement_type == "cross-sectional"),
               size = 3, shape = 21, fill = "white", stroke = 1.5) +
    geom_point(data = df %>% filter(measurement_type != "cross-sectional"),
               size = 3, shape = 16) +
    geom_linerange(aes(ymin = mean_flux_C - se_flux_C, 
                       ymax = mean_flux_C + se_flux_C)) +
    scale_color_manual(values = plot_colors) +
    scale_x_continuous(trans = "log10") +
    labs(title = paste("Comparison:", treat1, "vs", treat2),
         x = "Time (hours)",
         y = expression("Flux (mg C g"^-1*" h"^-1*")"),
         color = "Treatment") +
    theme_bw()
}

cat("\n")
cat("================================================================================\n")
cat("              ANALYSIS COMPLETE \n")
cat("================================================================================\n")
cat("\n")
cat("Working directory:", getwd(), "\n\n")
cat("Substrate additions used:\n")
print(substrate_additions)
cat("\n")
cat("Key changes in this version:\n")
cat("  ✓ Working directory set to your specified path\n")
cat("  ✓ Negative fluxes RETAINED (not set to zero)\n")
cat("  ✓ 1h and 2h data INCLUDED from population averages\n")
cat("  ✓ Open circles = cross-sectional (population avg)\n")
cat("  ✓ Filled circles = paired/tracked (jar-specific)\n")
cat("\n")
cat("Interactive functions available:\n")
cat("  - filter_treatment(df, 'glucose')           # Filter by treatment\n")
cat("  - filter_time(df, 0, 24)                    # Filter by time range\n")
cat("  - get_stats('glucose', 72)                  # Get stats for treatment at time\n")
cat("  - plot_single_treatment('glucose')          # Plot single treatment\n")
cat("  - compare_treatments('glucose', 'xylose')   # Compare two treatments\n")
cat("\n")
cat("Key objects created:\n")
cat("  - data                          # Full dataset\n")
cat("  - all_fluxes                    # Flux calculations\n")
cat("  - all_fluxes_normalized         # Normalized fluxes\n")
cat("  - cumulative_resp               # Cumulative respiration\n")
cat("  - cumulative_resp_normalized    # Normalized cumulative\n")
cat("  - summary_table                 # Complete summary\n")
cat("  - normalized_summary_table      # Normalized summary\n")
cat("  - p1-p11                        # All plot objects\n")
cat("\n")
cat("All time points included in plots:\n")
print(sort(unique(all_fluxes$time_point_plot)))
cat("\n")
cat("================================================================================\n")