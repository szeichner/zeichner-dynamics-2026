#!/usr/bin/env python3
"""
compute_multivariate_treatment_tests.py
========================================

Three multivariate analyses of treatment differences over time:

2. Pairwise PERMANOVA (Monomer vs Hemicellulose) at each timepoint
4. Treatment × time interaction PERMANOVA (full BC distance matrix)
6. Centroid-to-centroid BC distances between treatments over time

All use the BC distance matrix from compute_bc_pcoa.py.

Outputs:
  - data/derived/permanova_per_timepoint.tsv
  - data/derived/permanova_treatment_time_interaction.tsv
  - data/derived/centroid_distances_over_time.tsv
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')

# ============================================================================
# Load data
# ============================================================================
print("Loading BC distance matrix and metadata...")

npz = np.load(os.path.join(DERIVED_DIR, 'bc_distance_matrix.npz'), allow_pickle=True)
bc_full = npz['distances']
sample_ids_full = npz['sample_ids'].astype(str)

coords = pd.read_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'), sep='\t')
coords['taxon_id'] = coords['taxon_id'].astype(str)

# Ensure alignment
assert list(sample_ids_full) == list(coords.taxon_id.values), "Sample order mismatch"

meta = coords[['taxon_id', 'sample_name', 'treatment', 'time_hours', 'replicate']].copy()
meta['substrate_group'] = np.where(
    meta.treatment.isin(['GLUCOSE', 'XYLOSE']), 'Monomer',
    np.where(meta.treatment == 'CELLULOSE', 'Hemicellulose',
    np.where(meta.treatment == 'control', 'Control', 'Basal')))

n = len(meta)
print(f"  {n} samples, BC matrix {bc_full.shape}")


def permanova(dm, groups, n_perms=9999, seed=42):
    """PERMANOVA returning F, R², p."""
    rng = np.random.RandomState(seed)
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)
    if k < 2 or n < k + 1:
        return np.nan, np.nan, np.nan

    ss_total = np.sum(dm ** 2) / (2 * n)

    def ss_within(dm, grps):
        ss_w = 0.0
        for g in unique_groups:
            mask = grps == g
            n_g = mask.sum()
            if n_g > 1:
                ss_w += np.sum(dm[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        return ss_w

    ss_w_obs = ss_within(dm, groups)
    ss_b_obs = ss_total - ss_w_obs
    if ss_w_obs == 0:
        return np.nan, np.nan, np.nan
    f_obs = (ss_b_obs / (k - 1)) / (ss_w_obs / (n - k))
    r2 = ss_b_obs / ss_total

    n_ge = 1
    for _ in range(n_perms):
        perm = rng.permutation(groups)
        ss_w_p = ss_within(dm, perm)
        ss_b_p = ss_total - ss_w_p
        f_p = (ss_b_p / (k - 1)) / (ss_w_p / (n - k))
        if f_p >= f_obs:
            n_ge += 1
    p = n_ge / (n_perms + 1)
    return f_obs, r2, p


# ============================================================================
# Analysis 2: Pairwise PERMANOVA at each timepoint
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Pairwise PERMANOVA (Monomer vs Hemicellulose) per timepoint")
print("=" * 70)

# Timepoints where both Monomer and Hemicellulose have samples
main_mask = meta.substrate_group.isin(['Monomer', 'Hemicellulose'])
main_meta = meta[main_mask]
shared_tps = sorted(set(main_meta[main_meta.substrate_group == 'Monomer'].time_hours) &
                    set(main_meta[main_meta.substrate_group == 'Hemicellulose'].time_hours))

print(f"  Shared timepoints: {shared_tps}")

tp_results = []
for tp in shared_tps:
    tp_mask = (meta.time_hours == tp) & main_mask
    indices = np.where(tp_mask)[0]
    if len(indices) < 4:
        continue
    dm_tp = bc_full[np.ix_(indices, indices)]
    groups_tp = meta.loc[tp_mask, 'substrate_group'].values

    n_mono = (groups_tp == 'Monomer').sum()
    n_hemi = (groups_tp == 'Hemicellulose').sum()

    f_val, r2, p_val = permanova(dm_tp, groups_tp, n_perms=9999, seed=42)

    tp_results.append({
        'timepoint_hours': tp,
        'n_monomer': n_mono,
        'n_hemicellulose': n_hemi,
        'n_total': n_mono + n_hemi,
        'F': f_val,
        'R2': r2,
        'p_value': p_val,
    })
    print(f"  t={tp:4d}h: n_mono={n_mono}, n_hemi={n_hemi}, "
          f"F={f_val:.2f}, R²={r2:.3f}, p={p_val:.4f}")

tp_df = pd.DataFrame(tp_results)

# BH correction across timepoints
from statsmodels.stats.multitest import multipletests
if len(tp_df) > 0:
    reject, padj, _, _ = multipletests(tp_df.p_value.values, method='fdr_bh')
    tp_df['padj'] = padj
    tp_df['significant'] = reject
    print(f"\n  After BH correction:")
    for _, r in tp_df.iterrows():
        sig = '*' if r.significant else ' '
        print(f"  {sig} t={int(r.timepoint_hours):4d}h: R²={r.R2:.3f}, "
              f"p={r.p_value:.4f}, padj={r.padj:.4f}")

tp_df.to_csv(os.path.join(DERIVED_DIR, 'permanova_per_timepoint.tsv'),
             sep='\t', index=False)
print(f"\n  Saved permanova_per_timepoint.tsv")

# Also run pairwise: Glucose vs Xylose at each timepoint
print("\n  --- Glucose vs Xylose per timepoint (monomer pooling check) ---")
gx_results = []
for tp in sorted(meta[meta.treatment.isin(['GLUCOSE', 'XYLOSE'])].time_hours.unique()):
    tp_mask = (meta.time_hours == tp) & meta.treatment.isin(['GLUCOSE', 'XYLOSE'])
    indices = np.where(tp_mask)[0]
    if len(indices) < 4:
        continue
    dm_tp = bc_full[np.ix_(indices, indices)]
    groups_tp = meta.loc[tp_mask, 'treatment'].values
    n_g = (groups_tp == 'GLUCOSE').sum()
    n_x = (groups_tp == 'XYLOSE').sum()
    f_val, r2, p_val = permanova(dm_tp, groups_tp, n_perms=9999, seed=42)
    gx_results.append({
        'timepoint_hours': tp, 'n_glucose': n_g, 'n_xylose': n_x,
        'F': f_val, 'R2': r2, 'p_value': p_val,
    })
    print(f"    t={tp:4d}h: n_G={n_g}, n_X={n_x}, F={f_val:.2f}, R²={r2:.3f}, p={p_val:.4f}")

# ============================================================================
# Analysis 4: Treatment × Time interaction PERMANOVA
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Treatment × Time interaction PERMANOVA")
print("=" * 70)

# Use the approach: compare full model R² vs marginal models
# Full: PERMANOVA with treatment*time_category as groups
# Test interaction by comparing SS explained

# Simpler approach: sequential PERMANOVA
# Model 1: treatment only
# Model 2: time only
# Model 3: treatment + time (additive)
# The interaction is assessed by checking if treatment effects differ by timepoint

# First: omnibus treatment effect (all timepoints pooled)
main_indices = np.where(main_mask)[0]
dm_main = bc_full[np.ix_(main_indices, main_indices)]

# Treatment effect
groups_treat = meta.loc[main_mask, 'substrate_group'].values
f_treat, r2_treat, p_treat = permanova(dm_main, groups_treat, n_perms=9999, seed=42)
print(f"\n  Treatment effect (Monomer vs Hemicellulose, all timepoints pooled):")
print(f"    F={f_treat:.2f}, R²={r2_treat:.3f}, p={p_treat:.4f}")

# Time effect
groups_time = meta.loc[main_mask, 'time_hours'].astype(str).values
f_time, r2_time, p_time = permanova(dm_main, groups_time, n_perms=9999, seed=42)
print(f"\n  Time effect (timepoint as factor, treatments pooled):")
print(f"    F={f_time:.2f}, R²={r2_time:.3f}, p={p_time:.4f}")

# Treatment × Time (each treatment-timepoint combo as a separate group)
groups_txn = (meta.loc[main_mask, 'substrate_group'].values + '_' +
              meta.loc[main_mask, 'time_hours'].astype(str).values)
f_txn, r2_txn, p_txn = permanova(dm_main, groups_txn, n_perms=9999, seed=42)
print(f"\n  Treatment × Time (each treatment-timepoint as group):")
print(f"    F={f_txn:.2f}, R²={r2_txn:.3f}, p={p_txn:.4f}")

# Interaction R² = full model R² - (treatment R² + time R²)
r2_interaction = r2_txn - r2_treat - r2_time
print(f"\n  Interaction R² (approximate): {r2_interaction:.3f}")
print(f"    Full model R²:     {r2_txn:.3f}")
print(f"    Treatment marginal: {r2_treat:.3f}")
print(f"    Time marginal:      {r2_time:.3f}")
print(f"    Interaction:        {r2_interaction:.3f}")

interaction_results = pd.DataFrame([{
    'model': 'treatment', 'F': f_treat, 'R2': r2_treat, 'p': p_treat,
}, {
    'model': 'time', 'F': f_time, 'R2': r2_time, 'p': p_time,
}, {
    'model': 'treatment_x_time', 'F': f_txn, 'R2': r2_txn, 'p': p_txn,
}, {
    'model': 'interaction_residual', 'F': np.nan,
    'R2': r2_interaction, 'p': np.nan,
}])
interaction_results.to_csv(os.path.join(DERIVED_DIR, 'permanova_treatment_time_interaction.tsv'),
                           sep='\t', index=False)
print(f"\n  Saved permanova_treatment_time_interaction.tsv")

# ============================================================================
# Analysis 6: Centroid-to-centroid BC distances over time
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 6: Centroid-to-centroid BC distances over time")
print("=" * 70)

from scipy.spatial.distance import braycurtis

# Load relative abundance for centroid computation
# We need the original relative abundance vectors, not just the distance matrix
# Recompute from the count matrix
import json
METAT_DIR = os.path.join(BASE, 'data', 'metatranscriptomics', 'img data')
with open(os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')) as f:
    ko_data = json.load(f)

QC_EXCLUDED = [
    'sample20_GLUCOSE_xylose_168hrs_rep4',
    'sample25_glucose_XYLOSE_2hrs_rep1',
    'sample71_CELLULOSE_336hrs_rep3',
]
meta_full = pd.read_csv(os.path.join(METAT_DIR, 'IMG_metadata.tsv'), sep='\t')
meta_full['taxon_id'] = meta_full['taxon_id'].astype(str)
meta_full = meta_full[~meta_full.sample_name.isin(QC_EXCLUDED)].reset_index(drop=True)

keep = set(meta_full.taxon_id)
rows = {}
for ko_id, tc in ko_data.items():
    for t, c in tc.items():
        if t in keep:
            rows.setdefault(t, {})[ko_id] = c
counts = pd.DataFrame(rows).T.fillna(0).astype(int)
counts = counts.reindex(meta_full.taxon_id.values)

# Filter same as PCoA
feat_mask = (counts > 0).sum(axis=0) >= 3
counts_filt = counts.loc[:, feat_mask]
rel_abund = counts_filt.div(counts_filt.sum(axis=1), axis=0)

meta_full['substrate_group'] = np.where(
    meta_full.treatment.isin(['GLUCOSE', 'XYLOSE']), 'Monomer',
    np.where(meta_full.treatment == 'CELLULOSE', 'Hemicellulose',
    np.where(meta_full.treatment == 'control', 'Control', 'Basal')))

# Compute centroids (mean relative abundance) per treatment × timepoint
# Then BC distance between centroids
treatments_pairs = [
    ('GLUCOSE', 'CELLULOSE', 'Glucose vs Hemicellulose'),
    ('XYLOSE', 'CELLULOSE', 'Xylose vs Hemicellulose'),
    ('GLUCOSE', 'XYLOSE', 'Glucose vs Xylose'),
]

centroid_results = []
for t1, t2, label in treatments_pairs:
    # Timepoints shared between both treatments
    tps1 = set(meta_full[meta_full.treatment == t1].time_hours)
    tps2 = set(meta_full[meta_full.treatment == t2].time_hours)
    shared = sorted(tps1 & tps2)

    for tp in shared:
        ids1 = meta_full[(meta_full.treatment == t1) & (meta_full.time_hours == tp)].taxon_id.values
        ids2 = meta_full[(meta_full.treatment == t2) & (meta_full.time_hours == tp)].taxon_id.values

        centroid1 = rel_abund.loc[ids1].mean(axis=0).values
        centroid2 = rel_abund.loc[ids2].mean(axis=0).values

        bc_dist = braycurtis(centroid1, centroid2)

        centroid_results.append({
            'treatment_1': t1,
            'treatment_2': t2,
            'comparison': label,
            'timepoint_hours': tp,
            'bc_distance': bc_dist,
            'n_treat1': len(ids1),
            'n_treat2': len(ids2),
        })

cent_df = pd.DataFrame(centroid_results)
cent_df.to_csv(os.path.join(DERIVED_DIR, 'centroid_distances_over_time.tsv'),
               sep='\t', index=False)

print("\n  Centroid-to-centroid BC distances:")
for label in ['Glucose vs Hemicellulose', 'Xylose vs Hemicellulose', 'Glucose vs Xylose']:
    sub = cent_df[cent_df.comparison == label]
    print(f"\n  {label}:")
    for _, r in sub.iterrows():
        print(f"    t={int(r.timepoint_hours):4d}h: BC={r.bc_distance:.4f}")

print(f"\n  Saved centroid_distances_over_time.tsv")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY FOR MANUSCRIPT TEXT")
print("=" * 70)

print(f"""
Treatment explains {r2_treat:.1%} of community variance (PERMANOVA F={f_treat:.1f},
p={p_treat:.4f}), time explains {r2_time:.1%} (F={f_time:.1f}, p={p_time:.4f}),
and treatment × time jointly explains {r2_txn:.1%} (F={f_txn:.1f}, p={p_txn:.4f}).

Monomer and hemicellulose communities are distinguishable at:""")

for _, r in tp_df.iterrows():
    sig = "YES" if r.significant else "no"
    print(f"  t={int(r.timepoint_hours):4d}h: R²={r.R2:.3f}, padj={r.padj:.4f} [{sig}]")

print(f"""
Centroid-to-centroid BC distances show:
  Glucose vs Hemicellulose: range {cent_df[cent_df.comparison=='Glucose vs Hemicellulose'].bc_distance.min():.3f} to {cent_df[cent_df.comparison=='Glucose vs Hemicellulose'].bc_distance.max():.3f}
  Xylose vs Hemicellulose:  range {cent_df[cent_df.comparison=='Xylose vs Hemicellulose'].bc_distance.min():.3f} to {cent_df[cent_df.comparison=='Xylose vs Hemicellulose'].bc_distance.max():.3f}
  Glucose vs Xylose:        range {cent_df[cent_df.comparison=='Glucose vs Xylose'].bc_distance.min():.3f} to {cent_df[cent_df.comparison=='Glucose vs Xylose'].bc_distance.max():.3f}
""")

print("Done.")
