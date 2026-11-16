#!/usr/bin/env Rscript
# run_spline_lrt_vst.R — Complete Edison-matching pipeline in R
# 1. Load counts + metadata
# 2. DESeq2 with cond_combined design → VST with use_design=TRUE
# 3. OLS F-test on VST (amended samples only) with cr(log_time, df=3)
# 4. BH FDR correction
# 5. Output VST matrix + LRT results

suppressPackageStartupMessages({
    library(DESeq2)
    library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
counts_file <- args[1]
metadata_file <- args[2]
output_dir <- args[3]

cat("[R] Loading data...\n")
counts_df <- read.delim(counts_file, row.names = 1, check.names = FALSE)
meta_df <- read.delim(metadata_file, row.names = 1)

# Build substrate_group and cond_combined
meta_df$substrate_group <- ifelse(
    meta_df$treatment %in% c("GLUCOSE", "XYLOSE"), "MONOMER",
    ifelse(meta_df$treatment == "CELLULOSE", "HEMICELLULOSE", "control")
)
meta_df$cond_combined <- paste0(meta_df$substrate_group, "_t", meta_df$time_hours)
meta_df$cond_combined <- factor(meta_df$cond_combined)

cat(sprintf("[R] %d samples, %d KOs\n", nrow(counts_df), ncol(counts_df)))

# Transpose: DESeq2 wants genes as rows
count_matrix <- t(as.matrix(counts_df))

# Feature filter: >= 5 counts in >= 3 samples
keep_feat <- rowSums(count_matrix >= 5) >= 3
count_matrix <- count_matrix[keep_feat, ]
cat(sprintf("[R] After feature filter: %d KOs\n", nrow(count_matrix)))

# Full DESeq2 + VST with design
cat("[R] Running DESeq2 + VST (cond_combined design)...\n")
dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = meta_df,
    design = ~ cond_combined
)
dds <- DESeq(dds, quiet = TRUE)
vsd <- varianceStabilizingTransformation(dds, blind = FALSE)  # use_design=TRUE equivalent
vst_mat <- assay(vsd)  # KOs × samples

# Save VST matrix
vst_out <- file.path(output_dir, "vst_counts_R.tsv.gz")
vst_df <- as.data.frame(t(vst_mat))  # samples × KOs for consistency
gz <- gzfile(vst_out, "w")
write.table(vst_df, gz, sep = "\t", quote = FALSE)
close(gz)
cat(sprintf("[R] Saved VST: %s (%d samples x %d KOs)\n",
    vst_out, nrow(vst_df), ncol(vst_df)))

# OLS F-test on amended samples
amended <- meta_df$substrate_group %in% c("MONOMER", "HEMICELLULOSE")
amended_meta <- meta_df[amended, ]
amended_meta$log_time <- log(as.numeric(amended_meta$time_hours))
amended_meta$substrate_group <- factor(amended_meta$substrate_group,
    levels = c("HEMICELLULOSE", "MONOMER"))  # HEMICELLULOSE as reference

vst_amended <- t(vst_mat[, amended])  # samples × KOs
n_samples <- nrow(vst_amended)
n_kos <- ncol(vst_amended)

cat(sprintf("[R] OLS F-test on %d amended samples x %d KOs...\n", n_samples, n_kos))

# Design matrices using patsy-equivalent splines
X_full <- model.matrix(
    ~ substrate_group * ns(log_time, df = 3),
    data = amended_meta
)
X_red <- model.matrix(
    ~ substrate_group + ns(log_time, df = 3),
    data = amended_meta
)

p_full <- ncol(X_full)
p_red <- ncol(X_red)
df_int <- p_full - p_red
df_resid <- n_samples - p_full
cat(sprintf("[R] Full: %d params, Reduced: %d params, Interaction df: %d, Resid df: %d\n",
    p_full, p_red, df_int, df_resid))

# Vectorized computation
I <- diag(n_samples)
H_f <- X_full %*% solve(t(X_full) %*% X_full) %*% t(X_full)
H_r <- X_red %*% solve(t(X_red) %*% X_red) %*% t(X_red)

resid_f <- (I - H_f) %*% vst_amended
resid_r <- (I - H_r) %*% vst_amended
ssr_f <- colSums(resid_f^2)
ssr_r <- colSums(resid_r^2)

ssr_f[ssr_f == 0] <- 1e-10
F_stat <- ((ssr_r - ssr_f) / df_int) / (ssr_f / df_resid)
p_val <- 1 - pf(F_stat, df_int, df_resid)

# BH FDR
padj <- p.adjust(p_val, method = "BH")
n_sig <- sum(padj < 0.05, na.rm = TRUE)
cat(sprintf("[R] Significant KOs (FDR < 0.05): %d / %d\n", n_sig, n_kos))

# Compute baseMean from the DESeq2 object (mean normalized count per gene)
base_mean <- rowMeans(counts(dds, normalized = TRUE))

# Save results
results <- data.frame(
    ko_id = colnames(vst_amended),
    baseMean = base_mean[colnames(vst_amended)],
    F_stat = F_stat,
    p_value = p_val,
    padj = padj,
    significant = padj < 0.05,
    stringsAsFactors = FALSE
)
results <- results[order(results$padj), ]

out_file <- file.path(output_dir, "spline_lrt_results_R.tsv")
write.table(results, out_file, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("[R] Saved: %s\n", out_file))
