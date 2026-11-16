################################################################################
# qSIP2 per-timepoint EAF analysis — 16S and ITS
#
# Runs qSIP2 separately for each timepoint, comparing that timepoint's
# labeled samples against t=0 unlabeled controls. This gives per-timepoint
# EAF values that reveal temporal dynamics of 13C incorporation.
################################################################################

library(qSIP2)
library(tidyverse)

BASE <- "/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/data/qSIP/"

# ============================================================================
# Load data (shared across all timepoint runs)
# ============================================================================
cat("Loading data...\n")

source_meta <- read_tsv(file.path(BASE, "input/16S/z16S_source_metadata.txt"), show_col_types = FALSE)
sample_16S  <- read_tsv(file.path(BASE, "derived/qsip_16S_sample_data.tsv"), show_col_types = FALSE)
sample_ITS  <- read_tsv(file.path(BASE, "derived/qsip_ITS_sample_data.tsv"), show_col_types = FALSE)
otu_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_table_5x5.txt"), show_col_types = FALSE)
otu_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_table_5x5.txt"), show_col_types = FALSE)
tax_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_5x5_RDP_v19.txt"), show_col_types = FALSE)
tax_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_5x5_unite10.txt"), show_col_types = FALSE)

# Prepare source metadata
source_all <- source_meta %>%
  filter(source_mat_id %in% sample_16S$source_mat_id) %>%
  mutate(
    isotope = case_when(
      isotopolog %in% c("GLUCOSE_xylose", "glucose_XYLOSE", "CELLULOSE") ~ "13C",
      TRUE ~ "12C"),
    timepoint_hours = as.numeric(gsub("hrs", "", time)))

# Identify controls (12C, t=0) — these are reused for every timepoint
control_ids <- source_all %>% filter(isotope == "12C") %>% pull(source_mat_id)
cat(sprintf("  Controls (12C): %d sources (%s)\n", length(control_ids), paste(control_ids, collapse = ", ")))

# Labeled timepoints
labeled_sources <- source_all %>% filter(isotope == "13C")
timepoints <- sort(unique(labeled_sources$timepoint_hours))
cat(sprintf("  Labeled timepoints: %s\n", paste(timepoints, collapse = ", ")))

# ============================================================================
# Function to run qSIP2 for one marker × one timepoint
# ============================================================================
run_one_timepoint <- function(tp, marker, source_df, sample_df, otu_df, tax_df) {

  cat(sprintf("\n--- %s, timepoint = %d h ---\n", marker, tp))

  # Subset sources: controls + this timepoint's labeled samples
  tp_labeled_ids <- source_df %>%
    filter(isotope == "13C", timepoint_hours == tp) %>%
    pull(source_mat_id)

  if (length(tp_labeled_ids) == 0) {
    cat("  No labeled sources at this timepoint, skipping\n")
    return(NULL)
  }

  keep_ids <- c(control_ids, tp_labeled_ids)
  src_sub <- source_df %>% filter(source_mat_id %in% keep_ids)

  # Total abundance per source
  total_abund <- sample_df %>%
    filter(source_mat_id %in% keep_ids) %>%
    group_by(source_mat_id) %>%
    summarize(total_abundance = sum(copies_per_ul, na.rm = TRUE), .groups = "drop")
  src_sub <- src_sub %>% left_join(total_abund, by = "source_mat_id")

  cat(sprintf("  Sources: %d controls + %d labeled = %d total\n",
              length(control_ids), length(tp_labeled_ids), nrow(src_sub)))

  # Build objects
  src_obj <- tryCatch(
    qsip_source_data(src_sub, isotope = "isotope", isotopolog = "isotopolog",
                     source_mat_id = "source_mat_id", timepoint = "timepoint_hours",
                     total_abundance = "total_abundance"),
    error = function(e) { cat("  Source object error:", e$message, "\n"); return(NULL) })
  if (is.null(src_obj)) return(NULL)

  samp_sub <- sample_df %>%
    filter(source_mat_id %in% keep_ids) %>%
    select(sample_id, source_mat_id, fraction, density, copies_per_ul)

  samp_obj <- qsip_sample_data(samp_sub, sample_id = "sample_id",
                                source_mat_id = "source_mat_id",
                                gradient_position = "fraction",
                                gradient_pos_density = "density",
                                gradient_pos_amt = "copies_per_ul")

  valid_samples <- intersect(colnames(otu_df)[-1], get_dataframe(samp_obj)$sample_id)
  if (length(valid_samples) < 10) {
    cat(sprintf("  Too few valid samples (%d), skipping\n", length(valid_samples)))
    return(NULL)
  }

  otu_sub <- otu_df %>% select(1, all_of(valid_samples))
  feat_obj <- qsip_feature_data(otu_sub, feature_id = colnames(otu_df)[1])

  qsip_obj <- tryCatch(
    qsip_data(source_data = src_obj, sample_data = samp_obj, feature_data = feat_obj),
    error = function(e) { cat("  qsip_data error:", e$message, "\n"); return(NULL) })
  if (is.null(qsip_obj)) return(NULL)

  # Filter
  qsip_obj <- tryCatch(
    run_feature_filter(qsip_obj,
                       unlabeled_source_mat_ids = control_ids,
                       labeled_source_mat_ids = tp_labeled_ids,
                       min_unlabeled_sources = 2,
                       min_labeled_sources = 2,
                       min_unlabeled_fractions = 2,
                       min_labeled_fractions = 2,
                       quiet = TRUE),
    error = function(e) { cat("  Filter error:", e$message, "\n"); return(NULL) })
  if (is.null(qsip_obj)) return(NULL)

  # Resample
  qsip_obj <- tryCatch(
    run_resampling(qsip_obj, resamples = 1000, with_seed = 42,
                   allow_failures = TRUE, quiet = TRUE),
    error = function(e) { cat("  Resampling error:", e$message, "\n"); return(NULL) })
  if (is.null(qsip_obj)) return(NULL)

  # EAF
  qsip_obj <- tryCatch(
    run_EAF_calculations(qsip_obj),
    error = function(e) { cat("  EAF error:", e$message, "\n"); return(NULL) })
  if (is.null(qsip_obj)) return(NULL)

  eaf <- get_EAF_data(qsip_obj)

  # Keep observed values only
  eaf_obs <- eaf %>% filter(is.na(resample) | observed == TRUE)
  if (!"observed" %in% colnames(eaf)) {
    # Fall back: deduplicate by feature_id
    eaf_obs <- eaf %>% distinct(feature_id, .keep_all = TRUE)
  }

  eaf_obs$timepoint <- tp
  eaf_obs$marker <- marker

  cat(sprintf("  EAF: %d ASVs, mean=%.4f, median=%.4f, %% pos=%.1f%%\n",
              nrow(eaf_obs), mean(eaf_obs$EAF, na.rm = TRUE),
              median(eaf_obs$EAF, na.rm = TRUE),
              100 * mean(eaf_obs$EAF > 0, na.rm = TRUE)))

  return(eaf_obs)
}

# ============================================================================
# Run for all timepoints × both markers
# ============================================================================
all_results <- list()

for (tp in timepoints) {
  # 16S
  res_16S <- run_one_timepoint(tp, "16S", source_all, sample_16S, otu_16S, tax_16S)
  if (!is.null(res_16S)) {
    res_16S <- res_16S %>% left_join(tax_16S, by = c("feature_id" = "ASV"))
    all_results[[paste0("16S_", tp)]] <- res_16S
  }

  # ITS
  res_ITS <- run_one_timepoint(tp, "ITS", source_all, sample_ITS, otu_ITS, tax_ITS)
  if (!is.null(res_ITS)) {
    res_ITS <- res_ITS %>% left_join(tax_ITS, by = c("feature_id" = "ASV"))
    all_results[[paste0("ITS_", tp)]] <- res_ITS
  }
}

# ============================================================================
# Combine and save
# ============================================================================
cat("\n========================================\n")
cat("Combining results...\n")

eaf_all <- bind_rows(all_results)
cat(sprintf("  Total rows: %d\n", nrow(eaf_all)))
cat(sprintf("  Timepoints × markers: %d\n", n_distinct(paste(eaf_all$marker, eaf_all$timepoint))))

# Summary table
summary_table <- eaf_all %>%
  group_by(marker, timepoint) %>%
  summarize(
    n_ASVs = n(),
    mean_EAF = mean(EAF, na.rm = TRUE),
    median_EAF = median(EAF, na.rm = TRUE),
    pct_incorporators = 100 * mean(EAF > 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(marker, timepoint)

cat("\n  Summary:\n")
print(summary_table, n = 30)

# By phylum
phylum_summary <- eaf_all %>%
  filter(!is.na(Phylum)) %>%
  group_by(marker, timepoint, Phylum) %>%
  summarize(
    n_ASVs = n(),
    mean_EAF = mean(EAF, na.rm = TRUE),
    .groups = "drop"
  )

# Save
write_tsv(eaf_all, file.path(BASE, "output/tables/qsip_per_timepoint_EAF.tsv"))
write_tsv(summary_table, file.path(BASE, "output/tables/qsip_per_timepoint_summary.tsv"))
write_tsv(phylum_summary, file.path(BASE, "output/tables/qsip_per_timepoint_by_phylum.tsv"))

cat(sprintf("\nSaved:\n"))
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_per_timepoint_EAF.tsv")))
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_per_timepoint_summary.tsv")))
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_per_timepoint_by_phylum.tsv")))

cat("\n=== PER-TIMEPOINT ANALYSIS COMPLETE ===\n")
