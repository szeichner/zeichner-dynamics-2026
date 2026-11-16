#!/usr/bin/env python3
"""
compute_bc_pcoa.py — Analysis A for v6 CLAUDE.md
=================================================

Compute Bray-Curtis principal coordinates analysis (PCoA) on KO relative
abundances from the metatranscriptomics dataset. This replaces the PCA on
control-standardized VST as the main-text ordination (Figure 1C-D).

Steps:
  1. Load KO count matrix from ko_output.json (raw integer counts).
  2. Apply QC exclusions (3 failed + basal1 zero-count sample).
  3. Compute KO relative abundance per sample.
  4. Compute Bray-Curtis distance matrix.
  5. Classical MDS (PCoA) via eigendecomposition of the double-centered
     distance matrix.
  6. PERMANOVA on the BC distance matrix (treatment factor, 9999 permutations).
  7. Compute distance-from-baseline (control_t0 centroid) per sample.
  8. Save outputs to data/derived/.

Outputs:
  - data/derived/bc_pcoa_coords.tsv
  - data/derived/bc_distance_matrix.npz
  - data/derived/bc_pcoa_eigenvalues.tsv
  - data/derived/bc_distance_from_baseline.tsv
  - Console: variance explained, PERMANOVA results, sanity checks
"""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, braycurtis

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAT_DIR = os.path.join(BASE, 'data', 'metatranscriptomics', 'img data')
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')

# QC exclusions per CLAUDE.md
QC_EXCLUDED = [
    'sample20_GLUCOSE_xylose_168hrs_rep4',
    'sample25_glucose_XYLOSE_2hrs_rep1',
    'sample71_CELLULOSE_336hrs_rep3',
]
ZERO_COUNT_EXCLUDED = []  # basal1 excluded in oxidized-sugars paper but kept here

# ============================================================================
# 1. Load metadata
# ============================================================================
print("[1/7] Loading metadata...")
meta = pd.read_csv(os.path.join(METAT_DIR, 'IMG_metadata.tsv'), sep='\t')
meta['taxon_id'] = meta['taxon_id'].astype(str)

# Apply exclusions
all_excluded = QC_EXCLUDED + ZERO_COUNT_EXCLUDED
meta = meta[~meta['sample_name'].isin(all_excluded)].copy()
meta = meta.reset_index(drop=True)
print(f"  {len(meta)} samples retained after QC exclusions")

# ============================================================================
# 2. Load KO count matrix from JSON
# ============================================================================
print("[2/7] Loading KO count matrix from ko_output.json...")
ko_json_path = os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')
with open(ko_json_path) as f:
    ko_data = json.load(f)

keep_taxa = set(meta['taxon_id'])
rows = {}
for ko_id, taxon_counts in ko_data.items():
    for tid, count in taxon_counts.items():
        if tid not in keep_taxa:
            continue
        rows.setdefault(tid, {})[ko_id] = count

counts = pd.DataFrame(rows).T.fillna(0).astype(int)
counts.index.name = 'taxon_id'
# Reindex to match metadata order
counts = counts.reindex(meta['taxon_id'].values)
print(f"  Count matrix: {counts.shape[0]} samples × {counts.shape[1]} KO features")
print(f"  Total counts range: {counts.sum(axis=1).min()} - {counts.sum(axis=1).max()}")

# Feature filter: retain KOs with >= 1 count in >= 3 samples
min_samples = 3
feature_mask = (counts > 0).sum(axis=0) >= min_samples
counts_filt = counts.loc[:, feature_mask]
print(f"  After filtering (>0 in >= {min_samples} samples): {counts_filt.shape[1]} KO features retained")

# ============================================================================
# 3. Compute relative abundance
# ============================================================================
print("[3/7] Computing relative abundance...")
sample_totals = counts_filt.sum(axis=1)
rel_abund = counts_filt.div(sample_totals, axis=0)
print(f"  Relative abundance sums: {rel_abund.sum(axis=1).min():.6f} - {rel_abund.sum(axis=1).max():.6f}")

# ============================================================================
# 4. Compute Bray-Curtis distance matrix
# ============================================================================
print("[4/7] Computing Bray-Curtis distance matrix...")
bc_condensed = pdist(rel_abund.values, metric='braycurtis')
bc_matrix = squareform(bc_condensed)
print(f"  Distance matrix: {bc_matrix.shape}")
print(f"  BC distance range: {bc_condensed.min():.4f} - {bc_condensed.max():.4f}")

# Save distance matrix
np.savez_compressed(os.path.join(DERIVED_DIR, 'bc_distance_matrix.npz'),
                    distances=bc_matrix,
                    sample_ids=meta['taxon_id'].values)
print(f"  Saved bc_distance_matrix.npz")

# ============================================================================
# 5. PCoA via eigendecomposition of double-centered distance matrix
# ============================================================================
print("[5/7] Running PCoA (classical MDS via eigendecomposition)...")

n = bc_matrix.shape[0]
# Double-center the squared distance matrix
D2 = bc_matrix ** 2
centering = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * centering @ D2 @ centering

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(B)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Keep only positive eigenvalues
pos_mask = eigenvalues > 0
eigenvalues_pos = eigenvalues[pos_mask]
eigenvectors_pos = eigenvectors[:, pos_mask]

# Variance explained
total_var = eigenvalues_pos.sum()
var_explained = eigenvalues_pos / total_var * 100

print(f"  Positive eigenvalues: {len(eigenvalues_pos)}")
print(f"  PCo1: {var_explained[0]:.2f}% variance")
print(f"  PCo2: {var_explained[1]:.2f}% variance")
print(f"  PCo3: {var_explained[2]:.2f}% variance")
print(f"  PCo4: {var_explained[3]:.2f}% variance")
print(f"  PCo5: {var_explained[4]:.2f}% variance")

# Compute coordinates: PCo_i = eigenvector_i * sqrt(eigenvalue_i)
n_axes = min(10, len(eigenvalues_pos))
coords = eigenvectors_pos[:, :n_axes] * np.sqrt(eigenvalues_pos[:n_axes])

# Build output dataframe
pcoa_df = meta[['sample_name', 'taxon_id', 'treatment', 'time_hours', 'replicate']].copy()
for i in range(n_axes):
    pcoa_df[f'PCo{i+1}'] = coords[:, i]

pcoa_df.to_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'), sep='\t', index=False)
print(f"  Saved bc_pcoa_coords.tsv ({len(pcoa_df)} samples × {n_axes} axes)")

# Save eigenvalues
eig_df = pd.DataFrame({
    'axis': [f'PCo{i+1}' for i in range(len(eigenvalues_pos))],
    'eigenvalue': eigenvalues_pos,
    'variance_explained_pct': var_explained,
    'cumulative_pct': np.cumsum(var_explained),
})
eig_df.to_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_eigenvalues.tsv'), sep='\t', index=False)
print(f"  Saved bc_pcoa_eigenvalues.tsv")

# ============================================================================
# 6. PERMANOVA on BC distance matrix
# ============================================================================
print("[6/7] Running PERMANOVA (treatment factor, 9999 permutations)...")

def permanova(dm, groups, n_perms=9999, seed=42):
    """
    PERMANOVA (Anderson 2001) on a distance matrix.
    Tests whether group centroids differ in multivariate space.

    Returns: F_obs, R2, p_value
    """
    rng = np.random.RandomState(seed)
    n = len(groups)
    unique_groups = np.unique(groups)

    # Total sum of squares
    ss_total = np.sum(dm ** 2) / (2 * n)

    # Within-group sum of squares
    def ss_within(dm, grps):
        ss_w = 0.0
        for g in unique_groups:
            mask = grps == g
            n_g = mask.sum()
            if n_g > 1:
                sub_dm = dm[np.ix_(mask, mask)]
                ss_w += np.sum(sub_dm ** 2) / (2 * n_g)
        return ss_w

    ss_w_obs = ss_within(dm, groups)
    ss_b_obs = ss_total - ss_w_obs

    k = len(unique_groups)
    f_obs = (ss_b_obs / (k - 1)) / (ss_w_obs / (n - k))
    r2 = ss_b_obs / ss_total

    # Permutation test
    n_ge = 1  # include observed
    for _ in range(n_perms):
        perm_groups = rng.permutation(groups)
        ss_w_perm = ss_within(dm, perm_groups)
        ss_b_perm = ss_total - ss_w_perm
        f_perm = (ss_b_perm / (k - 1)) / (ss_w_perm / (n - k))
        if f_perm >= f_obs:
            n_ge += 1

    p_value = n_ge / (n_perms + 1)
    return f_obs, r2, p_value

# Run on main treatments only (exclude basal + control for treatment comparison)
# But CLAUDE.md says "substrate × time factor" — let's test treatment across all samples
groups = meta['treatment'].values
f_obs, r2, p_val = permanova(bc_matrix, groups, n_perms=9999, seed=42)
print(f"  PERMANOVA (all samples, treatment factor):")
print(f"    F = {f_obs:.2f}, R² = {r2:.4f}, p = {p_val:.4f}")

# Also test on main treatments only (GLUCOSE, XYLOSE, CELLULOSE)
main_mask = meta['treatment'].isin(['GLUCOSE', 'XYLOSE', 'CELLULOSE']).values
bc_main = bc_matrix[np.ix_(main_mask, main_mask)]
groups_main = meta.loc[main_mask, 'treatment'].values
f_main, r2_main, p_main = permanova(bc_main, groups_main, n_perms=9999, seed=42)
print(f"  PERMANOVA (main treatments only):")
print(f"    F = {f_main:.2f}, R² = {r2_main:.4f}, p = {p_main:.4f}")

# Test substrate_group (Monomer vs Hemicellulose)
substrate_group = np.where(
    meta['treatment'].isin(['GLUCOSE', 'XYLOSE']), 'Monomer',
    np.where(meta['treatment'] == 'CELLULOSE', 'Hemicellulose', 'Control'))
sg_mask = substrate_group != 'Control'
bc_sg = bc_matrix[np.ix_(sg_mask, sg_mask)]
groups_sg = substrate_group[sg_mask]
f_sg, r2_sg, p_sg = permanova(bc_sg, groups_sg, n_perms=9999, seed=42)
print(f"  PERMANOVA (Monomer vs Hemicellulose):")
print(f"    F = {f_sg:.2f}, R² = {r2_sg:.4f}, p = {p_sg:.4f}")

# ============================================================================
# 7. Distance from baseline (each sample to each control_t0 sample)
# ============================================================================
print("[7/7] Computing distance from baseline...")

# Baseline pool: control samples (t=0, unamended)
control_mask = meta['treatment'] == 'control'
control_indices = np.where(control_mask)[0]
control_names = meta.loc[control_mask, 'sample_name'].values
n_ctrl = len(control_indices)
print(f"  Control samples (baseline): n={n_ctrl}")
print(f"    {list(control_names)}")

# For each sample, store BC distance to EACH control sample individually,
# plus the mean across all controls.
baseline_rows = []
for i in range(len(meta)):
    row = {
        'sample_name': meta.iloc[i]['sample_name'],
        'taxon_id': meta.iloc[i]['taxon_id'],
        'treatment': meta.iloc[i]['treatment'],
        'time_hours': meta.iloc[i]['time_hours'],
        'replicate': meta.iloc[i]['replicate'],
        'PCo1': pcoa_df.iloc[i]['PCo1'],
    }
    dists_to_ctrl = []
    for k, ci in enumerate(control_indices):
        d = bc_matrix[i, ci]
        row[f'bc_dist_to_{control_names[k]}'] = d
        dists_to_ctrl.append(d)
    row['bc_dist_mean'] = np.mean(dists_to_ctrl)
    row['bc_dist_sd'] = np.std(dists_to_ctrl, ddof=1) if len(dists_to_ctrl) > 1 else 0
    baseline_rows.append(row)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(os.path.join(DERIVED_DIR, 'bc_distance_from_baseline.tsv'),
                   sep='\t', index=False)
print(f"  Saved bc_distance_from_baseline.tsv ({len(baseline_df)} rows, "
      f"per-control + mean columns)")

# ============================================================================
# Summary and sanity checks
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Samples: {len(meta)} (74 after QC)")
print(f"  KO features (filtered): {counts_filt.shape[1]}")
print(f"  PCo1: {var_explained[0]:.2f}% variance")
print(f"  PCo2: {var_explained[1]:.2f}% variance")
print(f"  PERMANOVA (all treatments): F={f_obs:.2f}, R²={r2:.4f}, p={p_val:.4f}")
print(f"  PERMANOVA (Monomer vs Hemicellulose): F={f_sg:.2f}, R²={r2_sg:.4f}, p={p_sg:.4f}")

# Sanity check: does sample45 (xylose 336h rep1) still appear as outlier?
s45 = meta[meta.sample_name.str.contains('sample45')]
if len(s45):
    s45_idx = s45.index[0]
    print(f"\n  Outlier check — sample45 (xylose 336h rep1):")
    print(f"    PCo1 = {pcoa_df.loc[s45_idx, 'PCo1']:.2f}")
    # Compare to other xylose 336h samples
    xy336 = meta[(meta.treatment == 'XYLOSE') & (meta.time_hours == 336)]
    xy336_pco1 = pcoa_df.loc[xy336.index, 'PCo1'].values
    print(f"    Other xylose 336h PCo1: {xy336_pco1}")

# Key framing check: do monomer and hemicellulose occupy distinguishable
# regions at most timepoints?
print("\n  Framing check — PCo1 means by treatment × timepoint:")
for treat in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
    sub = meta[meta.treatment == treat]
    for tp in sorted(sub.time_hours.unique()):
        tp_sub = sub[sub.time_hours == tp]
        pco1_vals = pcoa_df.loc[tp_sub.index, 'PCo1'].values
        print(f"    {treat:12s} t={tp:4d}h: PCo1 mean={np.mean(pco1_vals):7.3f} (n={len(pco1_vals)})")

# Distance from baseline by treatment × timepoint
print("\n  Distance from baseline (BC to control centroid):")
for treat in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
    sub = baseline_df[baseline_df.treatment == treat]
    for tp in sorted(sub.time_hours.unique()):
        tp_sub = sub[sub.time_hours == tp]
        d = tp_sub.bc_dist_mean.values
        print(f"    {treat:12s} t={tp:4d}h: dist={np.mean(d):.4f} ± {np.std(d, ddof=1)/np.sqrt(len(d)):.4f} (n={len(d)})")

print("\nDone.")
