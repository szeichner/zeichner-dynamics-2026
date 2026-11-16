################################################################################
# qSIP2 per-treatment × per-timepoint EAF analysis — 16S and ITS
#
# Runs qSIP2 separately for each treatment × timepoint combination,
# comparing against t=0 unlabeled controls.
# Treatments: GLUCOSE_xylose, glucose_XYLOSE, CELLULOSE
################################################################################

library(qSIP2)
library(tidyverse)

BASE <- "/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/data/qSIP/"

cat("Loading data...\n")

source_meta <- read_tsv(file.path(BASE, "input/16S/z16S_source_metadata.txt"), show_col_types = FALSE)
sample_16S  <- read_tsv(file.path(BASE, "derived/qsip_16S_sample_data.tsv"), show_col_types = FALSE)
sample_ITS  <- read_tsv(file.path(BASE, "derived/qsip_ITS_sample_data.tsv"), show_col_types = FALSE)
otu_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_table_5x5.txt"), show_col_types = FALSE)
otu_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_table_5x5.txt"), show_col_types = FALSE)
tax_16S     <- read_tsv(file.path(BASE, "input/16S/z16S_5x5_RDP_v19.txt"), show_col_types = FALSE)
tax_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_5x5_unite10.txt"), show_col_types = FALSE)

source_all <- source_meta %>%
  filter(source_mat_id %in% sample_16S$source_mat_id) %>%
  mutate(
    isotope = case_when(
      isotopolog %in% c("GLUCOSE_xylose", "glucose_XYLOSE", "CELLULOSE") ~ "13C",
      TRUE ~ "12C"),
    timepoint_hours = as.numeric(gsub("hrs", "", time)))

control_ids <- source_all %>% filter(isotope == "12C") %>% pull(source_mat_id)

treatments <- c("GLUCOSE_xylose", "glucose_XYLOSE", "CELLULOSE")
treat_labels <- c(GLUCOSE_xylose = "Glucose", glucose_XYLOSE = "Xylose", CELLULOSE = "Hemicellulose")

cat(sprintf("  Controls: %d | Treatments: %s\n", length(control_ids), paste(treatments, collapse = ", ")))

# ============================================================================
# Function: run one treatment × timepoint × marker
# ============================================================================
run_one <- function(treat, tp, marker, source_df, sample_df, otu_df) {

  cat(sprintf("  %s | %s | %dh ... ", marker, treat_labels[treat], tp))

  # Get labeled sources for this treatment × timepoint
  labeled_ids <- source_df %>%
    filter(isotope == "13C", isotopolog == treat, timepoint_hours == tp) %>%
    pull(source_mat_id)

  if (length(labeled_ids) < 2) {
    cat(sprintf("skip (n=%d labeled)\n", length(labeled_ids)))
    return(NULL)
  }

  keep_ids <- c(control_ids, labeled_ids)
  src_sub <- source_df %>% filter(source_mat_id %in% keep_ids)

  # Total abundance
  total_abund <- sample_df %>%
    filter(source_mat_id %in% keep_ids) %>%
    group_by(source_mat_id) %>%
    summarize(total_abundance = sum(copies_per_ul, na.rm = TRUE), .groups = "drop")
  src_sub <- src_sub %>% left_join(total_abund, by = "source_mat_id")

  # Build objects
  src_obj <- tryCatch(
    qsip_source_data(src_sub, isotope = "isotope", isotopolog = "isotopolog",
                     source_mat_id = "source_mat_id", timepoint = "timepoint_hours",
                     total_abundance = "total_abundance"),
    error = function(e) NULL)
  if (is.null(src_obj)) { cat("src err\n"); return(NULL) }

  samp_sub <- sample_df %>%
    filter(source_mat_id %in% keep_ids) %>%
    select(sample_id, source_mat_id, fraction, density, copies_per_ul)

  samp_obj <- tryCatch(
    qsip_sample_data(samp_sub, sample_id = "sample_id", source_mat_id = "source_mat_id",
                     gradient_position = "fraction", gradient_pos_density = "density",
                     gradient_pos_amt = "copies_per_ul"),
    error = function(e) NULL)
  if (is.null(samp_obj)) { cat("samp err\n"); return(NULL) }

  valid <- intersect(colnames(otu_df)[-1], get_dataframe(samp_obj)$sample_id)
  if (length(valid) < 5) { cat(sprintf("skip (%d samples)\n", length(valid))); return(NULL) }

  otu_sub <- otu_df %>% select(1, all_of(valid))
  feat_obj <- tryCatch(qsip_feature_data(otu_sub, feature_id = colnames(otu_df)[1]), error = function(e) NULL)
  if (is.null(feat_obj)) { cat("feat err\n"); return(NULL) }

  qsip_obj <- tryCatch(qsip_data(source_data = src_obj, sample_data = samp_obj, feature_data = feat_obj),
                        error = function(e) NULL)
  if (is.null(qsip_obj)) { cat("data err\n"); return(NULL) }

  qsip_obj <- tryCatch(
    run_feature_filter(qsip_obj, unlabeled_source_mat_ids = control_ids,
                       labeled_source_mat_ids = labeled_ids,
                       min_unlabeled_sources = 2, min_labeled_sources = 2,
                       min_unlabeled_fractions = 2, min_labeled_fractions = 2, quiet = TRUE),
    error = function(e) NULL)
  if (is.null(qsip_obj)) { cat("filter err\n"); return(NULL) }

  qsip_obj <- tryCatch(
    run_resampling(qsip_obj, resamples = 1000, with_seed = 42,
                   allow_failures = TRUE, quiet = TRUE),
    error = function(e) NULL)
  if (is.null(qsip_obj)) { cat("resamp err\n"); return(NULL) }

  qsip_obj <- tryCatch(run_EAF_calculations(qsip_obj), error = function(e) NULL)
  if (is.null(qsip_obj)) { cat("EAF err\n"); return(NULL) }

  eaf <- get_EAF_data(qsip_obj)
  eaf_obs <- eaf %>% distinct(feature_id, .keep_all = TRUE)
  eaf_obs$timepoint <- tp
  eaf_obs$treatment <- treat
  eaf_obs$treatment_label <- treat_labels[treat]
  eaf_obs$marker <- marker

  cat(sprintf("n=%d, EAF=%.4f\n", nrow(eaf_obs), mean(eaf_obs$EAF, na.rm = TRUE)))
  return(eaf_obs)
}

# ============================================================================
# Run all combinations
# ============================================================================
cat("\nRunning per-treatment × per-timepoint analysis...\n")

all_results <- list()
idx <- 1

for (treat in treatments) {
  # Get timepoints available for this treatment
  tps <- source_all %>%
    filter(isotopolog == treat) %>%
    pull(timepoint_hours) %>%
    unique() %>% sort()

  cat(sprintf("\n=== %s (%s) — timepoints: %s ===\n",
              treat_labels[treat], treat, paste(tps, collapse = ", ")))

  for (tp in tps) {
    # 16S
    res <- run_one(treat, tp, "16S", source_all, sample_16S, otu_16S)
    if (!is.null(res)) {
      res <- res %>% left_join(tax_16S, by = c("feature_id" = "ASV"))
      all_results[[idx]] <- res; idx <- idx + 1
    }

    # ITS
    res <- run_one(treat, tp, "ITS", source_all, sample_ITS, otu_ITS)
    if (!is.null(res)) {
      res <- res %>% left_join(tax_ITS, by = c("feature_id" = "ASV"))
      all_results[[idx]] <- res; idx <- idx + 1
    }
  }
}

# ============================================================================
# Combine and save
# ============================================================================
cat("\n========================================\n")
cat("Combining results...\n")

eaf_all <- bind_rows(all_results)
cat(sprintf("  Total rows: %d\n", nrow(eaf_all)))
cat(sprintf("  Combinations: %d\n",
            n_distinct(paste(eaf_all$marker, eaf_all$treatment, eaf_all$timepoint))))

# Summary table
summary_table <- eaf_all %>%
  group_by(marker, treatment_label, timepoint) %>%
  summarize(
    n_ASVs = n(),
    mean_EAF = mean(EAF, na.rm = TRUE),
    median_EAF = median(EAF, na.rm = TRUE),
    pct_incorporators = 100 * mean(EAF > 0, na.rm = TRUE),
    .groups = "drop") %>%
  arrange(marker, treatment_label, timepoint)

cat("\n  Summary:\n")
print(summary_table, n = 50)

# By phylum
phylum_summary <- eaf_all %>%
  filter(!is.na(Phylum)) %>%
  group_by(marker, treatment_label, timepoint, Phylum) %>%
  summarize(n_ASVs = n(), mean_EAF = mean(EAF, na.rm = TRUE),
            median_EAF = median(EAF, na.rm = TRUE),
            pct_incorporators = 100 * mean(EAF > 0, na.rm = TRUE),
            .groups = "drop")

# By genus
# NOTE ON GENUS-LEVEL INTERPRETATION:
# Genus-level qSIP results should be interpreted with caution because:
#   1. Many ASVs lack genus-level classification (~48% unassigned at genus level
#      in 16S). Results are biased toward well-characterized, cultured lineages.
#   2. qSIP EAF is computed per-ASV, then averaged across ASVs within a genus.
#      Genera with few ASVs (n<5) have unstable mean EAF estimates.
#   3. Taxonomic assignment is based on short amplicon regions (V4 for 16S, ITS2
#      for ITS) which have limited resolution below family level.
#   4. The GC content of DNA (which affects buoyant density and thus EAF
#      estimation) varies between species within a genus, adding noise.
#
# For these reasons, genus-level results are best used for:
#   - Identifying candidate taxa for targeted follow-up (isolation, metagenomics)
#   - Corroborating phylum-level patterns with finer taxonomic resolution
#   - Comparing functional guilds (e.g., known iron reducers vs fermenters)
# They should NOT be used to:
#   - Make definitive claims about a single genus's growth rate
#   - Compare EAF values between genera with very different ASV counts
#   - Infer species-level physiology from genus-level averages

genus_summary <- eaf_all %>%
  filter(!is.na(Genus)) %>%
  mutate(
    genus_clean = gsub("^g__", "", Genus),
    genus_label = ifelse(!is.na(Family),
                         paste0(genus_clean, " (", gsub("^f__", "", Family), ")"),
                         genus_clean)
  ) %>%
  group_by(marker, treatment_label, timepoint, genus_label, Phylum) %>%
  summarize(
    n_ASVs = n(),
    mean_EAF = mean(EAF, na.rm = TRUE),
    median_EAF = median(EAF, na.rm = TRUE),
    pct_incorporators = 100 * mean(EAF > 0, na.rm = TRUE),
    .groups = "drop")

# Top genera summary (overall, across all treatments and timepoints)
top_genera <- eaf_all %>%
  filter(!is.na(Genus)) %>%
  mutate(genus_clean = gsub("^g__", "", Genus)) %>%
  group_by(marker, genus_clean, Phylum) %>%
  summarize(
    mean_EAF = mean(EAF, na.rm = TRUE),
    n_ASVs = n_distinct(feature_id),
    n_timepoints = n_distinct(timepoint),
    .groups = "drop") %>%
  group_by(marker) %>%
  # Require at least 3 ASVs and 2 timepoints for reliable genus-level estimates
  filter(n_ASVs >= 3, n_timepoints >= 2) %>%
  arrange(desc(mean_EAF)) %>%
  ungroup()

cat("\n  Top 15 genera by mean EAF (16S, n_ASVs >= 3):\n")
top_genera %>% filter(marker == "16S") %>% head(15) %>%
  mutate(line = sprintf("    %-25s (%-20s) EAF=%+.4f  n_ASVs=%d  n_tp=%d",
                        genus_clean, Phylum, mean_EAF, n_ASVs, n_timepoints)) %>%
  pull(line) %>% cat(sep = "\n")

cat("\n\n  Top 15 genera by mean EAF (ITS, n_ASVs >= 3):\n")
top_genera %>% filter(marker == "ITS") %>% head(15) %>%
  mutate(line = sprintf("    %-25s (%-20s) EAF=%+.4f  n_ASVs=%d  n_tp=%d",
                        genus_clean, Phylum, mean_EAF, n_ASVs, n_timepoints)) %>%
  pull(line) %>% cat(sep = "\n")

# Save all outputs
OUT <- file.path(BASE, "output/tables")
write_tsv(eaf_all, file.path(OUT, "qsip_per_treatment_timepoint_EAF.tsv"))
write_tsv(summary_table, file.path(OUT, "qsip_per_treatment_timepoint_summary.tsv"))
write_tsv(phylum_summary, file.path(OUT, "qsip_per_treatment_timepoint_by_phylum.tsv"))
write_tsv(genus_summary, file.path(OUT, "qsip_per_treatment_timepoint_by_genus.tsv"))
write_tsv(top_genera, file.path(OUT, "qsip_top_genera_summary.tsv"))

cat(sprintf("\nSaved:\n"))
cat(sprintf("  %s\n", file.path(OUT, "qsip_per_treatment_timepoint_EAF.tsv")))
cat(sprintf("  %s\n", file.path(OUT, "qsip_per_treatment_timepoint_summary.tsv")))
cat(sprintf("  %s\n", file.path(OUT, "qsip_per_treatment_timepoint_by_phylum.tsv")))
cat(sprintf("  %s\n", file.path(OUT, "qsip_per_treatment_timepoint_by_genus.tsv")))
cat(sprintf("  %s\n", file.path(OUT, "qsip_top_genera_summary.tsv")))
cat("\n=== PER-TREATMENT ANALYSIS COMPLETE ===\n")
