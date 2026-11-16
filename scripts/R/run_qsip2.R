################################################################################
# qSIP2 analysis for Zeichner et al. — 16S
# EAF (excess atom fraction 13C) and per-capita growth rate calculations
# qSIP2 v0.23.9
################################################################################

library(qSIP2)
library(tidyverse)

BASE <- "/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/data/qSIP/"

# ============================================================================
# STEP 1: Load raw data
# ============================================================================
cat("STEP 1: Loading data...\n")

source_meta <- read_tsv(file.path(BASE, "input/16S/z16S_source_metadata.txt"), show_col_types = FALSE)
sample_16S  <- read_tsv(file.path(BASE, "derived/qsip_16S_sample_data.tsv"), show_col_types = FALSE)
otu_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_table_5x5.txt"), show_col_types = FALSE)
tax_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_5x5_RDP_v19.txt"), show_col_types = FALSE)

cat(sprintf("  Sources: %d | Fractions: %d | ASVs: %d\n",
            nrow(source_meta), nrow(sample_16S), nrow(otu_16S)))

# ============================================================================
# STEP 2: Prepare source metadata — assign isotope labels
# ============================================================================
cat("\nSTEP 2: Assigning isotope labels...\n")

source_16S <- source_meta %>%
  filter(source_mat_id %in% sample_16S$source_mat_id) %>%
  mutate(
    isotope = case_when(
      isotopolog %in% c("GLUCOSE_xylose", "glucose_XYLOSE", "CELLULOSE") ~ "13C",
      TRUE ~ "12C"),
    timepoint_hours = as.numeric(gsub("hrs", "", time)),
    treatment_group = case_when(
      isotopolog %in% c("GLUCOSE_xylose", "glucose_XYLOSE") ~ "monomer",
      isotopolog == "CELLULOSE" ~ "hemicellulose",
      TRUE ~ "control")
  )

## SPOT CHECK 2
cat("  12C:", sum(source_16S$isotope == "12C"), "sources |",
    "13C:", sum(source_16S$isotope == "13C"), "sources\n")

# ============================================================================
# STEP 3: Create three standalone qSIP2 objects, then combine
# ============================================================================
cat("\nSTEP 3: Building qSIP2 objects...\n")

# Compute total_abundance per source (sum of qPCR across fractions)
# Growth calculations require this to normalize per-capita rates
total_abund <- sample_16S %>%
  group_by(source_mat_id) %>%
  summarize(total_abundance = sum(copies_per_ul, na.rm = TRUE), .groups = "drop")

source_16S <- source_16S %>%
  left_join(total_abund, by = "source_mat_id")

cat(sprintf("  Total abundance range: [%.0f, %.0f] copies/source\n",
            min(source_16S$total_abundance, na.rm = TRUE),
            max(source_16S$total_abundance, na.rm = TRUE)))

# Source object
src_obj <- qsip_source_data(
  source_16S,
  isotope = "isotope", isotopolog = "isotopolog",
  source_mat_id = "source_mat_id", timepoint = "timepoint_hours",
  total_abundance = "total_abundance")

# Sample object
sample_df <- sample_16S %>%
  select(sample_id, source_mat_id, fraction, density, copies_per_ul)

samp_obj <- qsip_sample_data(
  sample_df,
  sample_id = "sample_id", source_mat_id = "source_mat_id",
  gradient_position = "fraction", gradient_pos_density = "density",
  gradient_pos_amt = "copies_per_ul")

# Feature object — filter OTU table to matching samples
samp_ids_in_obj <- get_dataframe(samp_obj)$sample_id
valid_samples <- intersect(colnames(otu_16S)[-1], samp_ids_in_obj)
otu_filtered <- otu_16S %>% select(ASV, all_of(valid_samples))

feat_obj <- qsip_feature_data(otu_filtered, feature_id = "ASV")

# Combine
qsip_obj <- qsip_data(
  source_data = src_obj,
  sample_data = samp_obj,
  feature_data = feat_obj)

## SPOT CHECK 3
cat("  Combined object:\n")
print(get_object_summary(qsip_obj))
cat(sprintf("  Shared sources: %d\n", length(get_shared_ids(qsip_obj)[[1]]$shared)))

# ============================================================================
# STEP 4: Filter features
# ============================================================================
cat("\nSTEP 4: Filtering features...\n")

unlabeled_ids <- source_16S %>% filter(isotope == "12C") %>% pull(source_mat_id)
labeled_ids   <- source_16S %>% filter(isotope == "13C") %>% pull(source_mat_id)

cat(sprintf("  Unlabeled: %d sources (%s)\n", length(unlabeled_ids), paste(unlabeled_ids, collapse = ", ")))
cat(sprintf("  Labeled: %d sources\n", length(labeled_ids)))

qsip_obj <- run_feature_filter(
  qsip_obj,
  unlabeled_source_mat_ids = unlabeled_ids,
  labeled_source_mat_ids = labeled_ids,
  min_unlabeled_sources = 3,
  min_labeled_sources = 5,
  min_unlabeled_fractions = 3,
  min_labeled_fractions = 3)

## SPOT CHECK 4
fr <- get_filter_results(qsip_obj)
cat("  Filter results:\n")
print(fr)
cat("  → ~2000-5000 ASVs passing is typical for stringent filtering\n")

# ============================================================================
# STEP 5: Resampling (bootstrap)
# ============================================================================
cat("\nSTEP 5: Resampling (1000 bootstraps)...\n")
cat("  This may take 10-20 minutes...\n")

qsip_obj <- run_resampling(qsip_obj, resamples = 1000, with_seed = 42,
                           allow_failures = TRUE)

## SPOT CHECK 5
resamp <- get_resample_counts(qsip_obj)
cat(sprintf("  Resampled %d ASVs\n", nrow(resamp)))
cat(sprintf("  Labeled resamples range: [%d, %d]\n",
            min(resamp$labeled_resamples), max(resamp$labeled_resamples)))
cat(sprintf("  Unlabeled resamples range: [%d, %d]\n",
            min(resamp$unlabeled_resamples), max(resamp$unlabeled_resamples)))

# ============================================================================
# STEP 6: EAF calculations
# ============================================================================
cat("\nSTEP 6: Calculating EAF...\n")

qsip_obj <- run_EAF_calculations(qsip_obj)

## SPOT CHECK 6
eaf <- get_EAF_data(qsip_obj)
cat(sprintf("  EAF for %d ASVs\n", nrow(eaf)))
cat(sprintf("  EAF range: [%.4f, %.4f]\n", min(eaf$EAF, na.rm = TRUE), max(eaf$EAF, na.rm = TRUE)))
cat(sprintf("  Incorporators (EAF > 0): %d (%.1f%%)\n",
            sum(eaf$EAF > 0, na.rm = TRUE), 100 * mean(eaf$EAF > 0, na.rm = TRUE)))

# Save EAF with taxonomy
eaf_out <- eaf %>% left_join(tax_16S, by = c("feature_id" = "ASV"))
write_tsv(eaf_out, file.path(BASE, "output/tables/qsip_16S_EAF_results.tsv"))
cat(sprintf("  Saved: qsip_16S_EAF_results.tsv\n"))

# ============================================================================
# STEP 7: Compute N_total at t=0 for growth calculations
# ============================================================================
cat("\nSTEP 7: Computing N_total at t=0...\n")

t0_sources <- source_16S %>% filter(isotope == "12C") %>% pull(source_mat_id)
t0_samples <- sample_df %>% filter(source_mat_id %in% t0_sources)

# N_total per ASV = sum(qPCR × relative_abundance) across fractions, averaged across controls
t0_otu <- otu_filtered %>% select(ASV, any_of(t0_samples$sample_id))

N_total_t0 <- t0_otu %>%
  pivot_longer(-ASV, names_to = "sample_id", values_to = "count") %>%
  group_by(sample_id) %>%
  mutate(rel_abund = count / sum(count)) %>%
  ungroup() %>%
  left_join(t0_samples %>% select(sample_id, source_mat_id, copies_per_ul), by = "sample_id") %>%
  mutate(N_frac = rel_abund * copies_per_ul) %>%
  group_by(feature_id = ASV, source_mat_id) %>%
  summarize(N_source = sum(N_frac, na.rm = TRUE), .groups = "drop") %>%
  group_by(feature_id) %>%
  summarize(N_total_i0 = mean(N_source), .groups = "drop") %>%
  mutate(timepoint1 = 0)

## SPOT CHECK 7
cat(sprintf("  N_total for %d ASVs, range [%.0f, %.0f]\n",
            nrow(N_total_t0), min(N_total_t0$N_total_i0), max(N_total_t0$N_total_i0)))

# ============================================================================
# STEP 8: Growth calculations
# ============================================================================
cat("\nSTEP 8: Calculating growth rates...\n")

qsip_obj <- run_growth_calculations(
  qsip_obj,
  N_total_it = N_total_t0,
  growth_model = "exponential",
  timepoint = "timepoint")

## SPOT CHECK 8
growth <- get_growth_data(qsip_obj)
cat(sprintf("  Growth rates for %d ASVs\n", nrow(growth)))
cat(sprintf("  Columns: %s\n", paste(colnames(growth), collapse = ", ")))

# Save with taxonomy
growth_out <- growth %>% left_join(tax_16S, by = c("feature_id" = "ASV"))
write_tsv(growth_out, file.path(BASE, "output/tables/qsip_16S_growth_results.tsv"))
cat(sprintf("  Saved: qsip_16S_growth_results.tsv\n"))

# ============================================================================
# STEP 9: Save and summarize
# ============================================================================
cat("\nSTEP 9: Saving...\n")

save(qsip_obj, N_total_t0, source_16S, eaf_out, growth_out,
     file = file.path(BASE, "output/RData/qsip2_16S_complete.RData"))

# Quick phylum summary
if ("Phylum" %in% colnames(eaf_out)) {
  cat("\n  EAF by Phylum (top 10):\n")
  eaf_out %>%
    filter(!is.na(EAF)) %>%
    group_by(Phylum) %>%
    summarize(n = n(), mean_EAF = mean(EAF), pct_pos = 100 * mean(EAF > 0), .groups = "drop") %>%
    filter(n >= 5) %>%
    arrange(desc(mean_EAF)) %>%
    head(10) %>%
    print()
}

cat("\n=== 16S ANALYSIS COMPLETE ===\n")
cat("Output files:\n")
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_16S_EAF_results.tsv")))
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_16S_growth_results.tsv")))
cat(sprintf("  %s\n", file.path(BASE, "output/RData/qsip2_16S_complete.RData")))
