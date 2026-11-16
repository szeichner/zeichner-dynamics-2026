################################################################################
# qSIP2 analysis — ITS (fungi)
# Same pipeline as 16S, adapted for ITS data
################################################################################

library(qSIP2)
library(tidyverse)

BASE <- "/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/data/qSIP/"

cat("=== ITS qSIP2 Analysis ===\n")

# Load data
source_meta <- read_tsv(file.path(BASE, "input/16S/z16S_source_metadata.txt"), show_col_types = FALSE)
sample_ITS  <- read_tsv(file.path(BASE, "derived/qsip_ITS_sample_data.tsv"), show_col_types = FALSE)
otu_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_table_5x5.txt"), show_col_types = FALSE)
tax_ITS     <- read_tsv(file.path(BASE, "input/ITS/ITS_exp_5x5_unite10.txt"), show_col_types = FALSE)

cat(sprintf("  Sources: %d | Fractions: %d | ASVs: %d\n",
            n_distinct(sample_ITS$source_mat_id), nrow(sample_ITS), nrow(otu_ITS)))

# Source metadata
source_ITS <- source_meta %>%
  filter(source_mat_id %in% sample_ITS$source_mat_id) %>%
  mutate(
    isotope = case_when(
      isotopolog %in% c("GLUCOSE_xylose", "glucose_XYLOSE", "CELLULOSE") ~ "13C",
      TRUE ~ "12C"),
    timepoint_hours = as.numeric(gsub("hrs", "", time)))

# Total abundance per source
total_abund <- sample_ITS %>%
  group_by(source_mat_id) %>%
  summarize(total_abundance = sum(copies_per_ul, na.rm = TRUE), .groups = "drop")
source_ITS <- source_ITS %>% left_join(total_abund, by = "source_mat_id")

cat(sprintf("  12C: %d | 13C: %d sources\n",
            sum(source_ITS$isotope == "12C"), sum(source_ITS$isotope == "13C")))

# Build objects
src_obj <- qsip_source_data(source_ITS, isotope = "isotope", isotopolog = "isotopolog",
                             source_mat_id = "source_mat_id", timepoint = "timepoint_hours",
                             total_abundance = "total_abundance")

sample_df <- sample_ITS %>% select(sample_id, source_mat_id, fraction, density, copies_per_ul)
samp_obj <- qsip_sample_data(sample_df, sample_id = "sample_id",
                              source_mat_id = "source_mat_id", gradient_position = "fraction",
                              gradient_pos_density = "density", gradient_pos_amt = "copies_per_ul")

valid <- intersect(colnames(otu_ITS)[-1], get_dataframe(samp_obj)$sample_id)
otu_filt <- otu_ITS %>% select(ASV, all_of(valid))
feat_obj <- qsip_feature_data(otu_filt, feature_id = "ASV")

qsip_obj <- qsip_data(source_data = src_obj, sample_data = samp_obj, feature_data = feat_obj)
cat(sprintf("  Shared sources: %d\n", length(get_shared_ids(qsip_obj)[[1]]$shared)))

# Filter
unlabeled_ids <- source_ITS %>% filter(isotope == "12C") %>% pull(source_mat_id)
labeled_ids   <- source_ITS %>% filter(isotope == "13C") %>% pull(source_mat_id)

qsip_obj <- run_feature_filter(qsip_obj,
  unlabeled_source_mat_ids = unlabeled_ids, labeled_source_mat_ids = labeled_ids,
  min_unlabeled_sources = 3, min_labeled_sources = 5,
  min_unlabeled_fractions = 3, min_labeled_fractions = 3)

# Resample
cat("\nResampling (1000 bootstraps, allow_failures)...\n")
qsip_obj <- run_resampling(qsip_obj, resamples = 1000, with_seed = 42, allow_failures = TRUE)
resamp <- get_resample_counts(qsip_obj)
cat(sprintf("  Resampled: %d ASVs\n", nrow(resamp)))

# EAF
qsip_obj <- run_EAF_calculations(qsip_obj)
eaf <- get_EAF_data(qsip_obj)
cat(sprintf("  EAF: %d values, range [%.4f, %.4f]\n", nrow(eaf),
            min(eaf$EAF, na.rm = TRUE), max(eaf$EAF, na.rm = TRUE)))

# N_total at t=0
t0_sources <- source_ITS %>% filter(isotope == "12C") %>% pull(source_mat_id)
t0_samples <- sample_df %>% filter(source_mat_id %in% t0_sources)
t0_otu <- otu_filt %>% select(ASV, any_of(t0_samples$sample_id))

N_total_t0 <- t0_otu %>%
  pivot_longer(-ASV, names_to = "sample_id", values_to = "count") %>%
  group_by(sample_id) %>% mutate(rel_abund = count / sum(count)) %>% ungroup() %>%
  left_join(t0_samples %>% select(sample_id, source_mat_id, copies_per_ul), by = "sample_id") %>%
  mutate(N_frac = rel_abund * copies_per_ul) %>%
  group_by(feature_id = ASV, source_mat_id) %>%
  summarize(N_source = sum(N_frac, na.rm = TRUE), .groups = "drop") %>%
  group_by(feature_id) %>%
  summarize(N_total_i0 = mean(N_source), .groups = "drop") %>%
  mutate(timepoint1 = 0)

# Growth
qsip_obj <- run_growth_calculations(qsip_obj, N_total_it = N_total_t0,
                                     growth_model = "exponential", timepoint = "timepoint")

growth <- get_growth_data(qsip_obj)
cat(sprintf("  Growth: %d values\n", nrow(growth)))

# Save
eaf_out <- eaf %>% left_join(tax_ITS, by = c("feature_id" = "ASV"))
growth_out <- growth %>% left_join(tax_ITS, by = c("feature_id" = "ASV"))
write_tsv(eaf_out, file.path(BASE, "output/tables/qsip_ITS_EAF_results.tsv"))
write_tsv(growth_out, file.path(BASE, "output/tables/qsip_ITS_growth_results.tsv"))
save(qsip_obj, N_total_t0, eaf_out, growth_out, file = file.path(BASE, "output/RData/qsip2_ITS_complete.RData"))

cat("\n=== ITS ANALYSIS COMPLETE ===\n")
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_ITS_EAF_results.tsv")))
cat(sprintf("  %s\n", file.path(BASE, "output/tables/qsip_ITS_growth_results.tsv")))
