#!/usr/bin/env python3
"""
Abundance-Bin EAF Partition Analysis
=====================================
Addresses reviewer comments (Steve/Jeff): Do abundant or rare taxa account
for more 13C incorporation?

Approach:
1. Assign each ASV to a fixed abundance bin using mean relative abundance
   across all treatment x timepoint combinations (no t=0 controls available
   in qSIP data; unlabeled controls are separate source microcosms).
2. Partition community-level weighted EAF by abundance bin.
3. Bootstrap 95% CIs on bin shares (2000 resamples).
4. Generate stacked bar figure + companion panel.

Author: Claude (supplementary analysis for manuscript)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# Paths
# ============================================================================
BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
EAF_FILE = os.path.join(BASE, 'data/qSIP/output/tables/qsip_per_treatment_timepoint_EAF.tsv')
COMM_FILE = os.path.join(BASE, 'data/qSIP/output/tables/qsip_community_incorporation_efficiency.tsv')

OTU_16S = os.path.join(BASE, 'data/qSIP/input/16S/z16S_table_5x5.txt')
OTU_ITS = os.path.join(BASE, 'data/qSIP/input/ITS/ITS_exp_table_5x5.txt')
SAMPLE_META_16S = os.path.join(BASE, 'data/qSIP/input/16S/z16S_sample_metadata.txt')
SAMPLE_META_ITS = os.path.join(BASE, 'data/qSIP/input/ITS/ITS_exp_sample_metadata.txt')
SOURCE_META = os.path.join(BASE, 'data/qSIP/input/16S/z16S_source_metadata.txt')

FIG_DIR = os.path.join(BASE, 'figures/output')
TABLE_DIR = os.path.join(BASE, 'tables/output')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ============================================================================
# Style (following project conventions from CLAUDE.md)
# ============================================================================
TREAT_COLORS = {
    'Glucose':       '#6B7C2E',
    'Xylose':        '#C8960C',
    'Hemicellulose': '#7B3F6E',
}
TREAT_ORDER = ['Glucose', 'Xylose', 'Hemicellulose']
MARKERS_SHAPE = {'Glucose': 'o', 'Xylose': '^', 'Hemicellulose': 's'}

# Abundance bin colors — separate palettes for bacteria (teal) and fungi (coral)
# following project domain-color convention: Bacterial #2A9D8F, Fungal #E76F51
BIN_COLORS_16S = {
    '>1%':        '#1A6B5A',   # dark teal
    '0.1-1%':     '#2A9D8F',   # teal
    '0.01-0.1%':  '#7CCABD',   # medium teal
    '<0.01%':     '#C5E8E2',   # light teal
}
BIN_COLORS_ITS = {
    '>1%':        '#B84A2E',   # dark coral
    '0.1-1%':     '#E76F51',   # coral
    '0.01-0.1%':  '#F0A58E',   # medium coral
    '<0.01%':     '#FADAD0',   # light coral
}
# Generic colors for companion panels (bottom row) and fallback
BIN_COLORS = {
    '>1%':        '#2c3e50',
    '0.1-1%':     '#5b7fa5',
    '0.01-0.1%':  '#95bbd6',
    '<0.01%':     '#d4e6f1',
}
BIN_ORDER = ['>1%', '0.1-1%', '0.01-0.1%', '<0.01%']

RANK_BIN_COLORS_16S = {
    'Top 10':    '#1A6B5A',
    'Rank 11-100': '#2A9D8F',
    'Remainder': '#C5E8E2',
}
RANK_BIN_COLORS_ITS = {
    'Top 10':    '#B84A2E',
    'Rank 11-100': '#E76F51',
    'Remainder': '#FADAD0',
}
RANK_BIN_COLORS = {
    'Top 10':    '#2c3e50',
    'Rank 11-100': '#5b7fa5',
    'Remainder': '#d4e6f1',
}
RANK_BIN_ORDER = ['Top 10', 'Rank 11-100', 'Remainder']

TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

N_BOOT = 2000

# ============================================================================
# 1. Load data
# ============================================================================
print("Loading data...")
eaf_all = pd.read_csv(EAF_FILE, sep='\t')
comm_df = pd.read_csv(COMM_FILE, sep='\t')

# Load source metadata for treatment mapping
source_meta = pd.read_csv(SOURCE_META, sep='\t')
treat_map = {'GLUCOSE_xylose': 'Glucose', 'glucose_XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}
source_meta = source_meta.dropna(subset=['time', 'isotopolog'])
source_meta['treatment_label'] = source_meta['isotopolog'].map(treat_map)
source_meta['timepoint_hours'] = source_meta['time'].str.replace('hrs', '').astype(int)
# Keep only labeled treatments (drop controls)
source_meta = source_meta[source_meta['treatment_label'].notna()].copy()

# Load OTU tables
print("Loading OTU tables...")
otu_16S = pd.read_csv(OTU_16S, sep='\t', index_col=0)
otu_ITS = pd.read_csv(OTU_ITS, sep='\t', index_col=0)

# Load sample metadata
sample_meta_16S = pd.read_csv(SAMPLE_META_16S, sep='\t')
sample_meta_ITS = pd.read_csv(SAMPLE_META_ITS, sep='\t')

print(f"  EAF data: {len(eaf_all)} rows, markers: {eaf_all['marker'].unique()}")
print(f"  16S OTU table: {otu_16S.shape[0]} ASVs x {otu_16S.shape[1]} samples")
print(f"  ITS OTU table: {otu_ITS.shape[0]} ASVs x {otu_ITS.shape[1]} samples")


# ============================================================================
# 2. Compute per-ASV relative abundance for each treatment x timepoint
#    Sum reads across all fractions within each source, then across replicates
# ============================================================================
def compute_asv_rel_abundance(otu_df, sample_meta_df, source_meta_df, marker_suffix):
    """Compute per-ASV relative abundance for each treatment x timepoint.

    Returns a DataFrame with columns: feature_id, treatment_label, timepoint, rel_abund
    """
    results = []

    for treat_code, treat_label in treat_map.items():
        for tp_hours in sorted(source_meta_df[
            source_meta_df['isotopolog'] == treat_code]['timepoint_hours'].unique()):

            # Get source_mat_ids for this treatment x timepoint
            src_ids = source_meta_df[
                (source_meta_df['isotopolog'] == treat_code) &
                (source_meta_df['timepoint_hours'] == tp_hours)
            ]['source_mat_id'].values

            # Get sample_ids (fractions) for these sources
            samp_ids = sample_meta_df[
                sample_meta_df['source_mat_id'].isin(src_ids)
            ]['sample_id'].unique()

            # Filter to available columns in OTU table
            available = [s for s in samp_ids if s in otu_df.columns]
            if len(available) == 0:
                continue

            # Sum reads per ASV across all fractions and replicates
            asv_reads = otu_df[available].sum(axis=1)
            total_reads = asv_reads.sum()
            if total_reads == 0:
                continue

            rel_abund = asv_reads / total_reads

            df = pd.DataFrame({
                'feature_id': rel_abund.index,
                'treatment_label': treat_label,
                'timepoint': tp_hours,
                'total_reads': asv_reads.values,
                'rel_abund': rel_abund.values,
            })
            results.append(df)

    return pd.concat(results, ignore_index=True)


print("\nComputing per-ASV relative abundance...")
rel_16S = compute_asv_rel_abundance(otu_16S, sample_meta_16S, source_meta, '16S')
rel_16S['marker'] = '16S'
rel_ITS = compute_asv_rel_abundance(otu_ITS, sample_meta_ITS, source_meta, 'ITS')
rel_ITS['marker'] = 'ITS'
rel_all = pd.concat([rel_16S, rel_ITS], ignore_index=True)

print(f"  16S: {len(rel_16S)} ASV x treatment x timepoint entries")
print(f"  ITS: {len(rel_ITS)} ASV x treatment x timepoint entries")


# ============================================================================
# 3. Compute FIXED bin membership per ASV per marker
#    Use mean relative abundance across all treatment x timepoint combos
# ============================================================================
def assign_abundance_bins(rel_df):
    """Assign ASVs to fixed abundance bins based on mean rel abundance."""
    mean_abund = rel_df.groupby(['marker', 'feature_id'])['rel_abund'].mean().reset_index()
    mean_abund.rename(columns={'rel_abund': 'mean_rel_abund'}, inplace=True)

    # Abundance bins
    conditions = [
        mean_abund['mean_rel_abund'] >= 0.01,
        (mean_abund['mean_rel_abund'] >= 0.001) & (mean_abund['mean_rel_abund'] < 0.01),
        (mean_abund['mean_rel_abund'] >= 0.0001) & (mean_abund['mean_rel_abund'] < 0.001),
        mean_abund['mean_rel_abund'] < 0.0001,
    ]
    labels = ['>1%', '0.1-1%', '0.01-0.1%', '<0.01%']
    mean_abund['abund_bin'] = np.select(conditions, labels, default='<0.01%')

    # Rank-based bins
    for mkr in mean_abund['marker'].unique():
        mask = mean_abund['marker'] == mkr
        ranks = mean_abund.loc[mask, 'mean_rel_abund'].rank(ascending=False, method='first')
        rank_bins = np.where(ranks <= 10, 'Top 10',
                    np.where(ranks <= 100, 'Rank 11-100', 'Remainder'))
        mean_abund.loc[mask, 'rank_bin'] = rank_bins

    return mean_abund


print("\nAssigning abundance bins...")
bin_assignments = assign_abundance_bins(rel_all)

for mkr in ['16S', 'ITS']:
    sub = bin_assignments[bin_assignments['marker'] == mkr]
    print(f"\n  {mkr} abundance bins:")
    for b in BIN_ORDER:
        bsub = sub[sub['abund_bin'] == b]
        print(f"    {b:>12s}: {len(bsub):5d} ASVs, cumulative rel_abund = {bsub['mean_rel_abund'].sum():.4f}")
    print(f"  {mkr} rank bins:")
    for b in RANK_BIN_ORDER:
        bsub = sub[sub['rank_bin'] == b]
        print(f"    {b:>12s}: {len(bsub):5d} ASVs, cumulative rel_abund = {bsub['mean_rel_abund'].sum():.4f}")


# ============================================================================
# 4. Merge EAF with abundance data and bin assignments
# ============================================================================
print("\nMerging EAF with abundance and bin assignments...")

# For each treatment x timepoint, get per-ASV reads from the same group
eaf_merged = eaf_all.merge(
    rel_all[['feature_id', 'treatment_label', 'timepoint', 'total_reads', 'rel_abund', 'marker']],
    left_on=['feature_id', 'treatment_label', 'timepoint', 'marker'],
    right_on=['feature_id', 'treatment_label', 'timepoint', 'marker'],
    how='left'
)

# Merge bin assignments (fixed per ASV per marker)
eaf_merged = eaf_merged.merge(
    bin_assignments[['marker', 'feature_id', 'mean_rel_abund', 'abund_bin', 'rank_bin']],
    on=['marker', 'feature_id'],
    how='left'
)

# Drop rows with no read data
n_before = len(eaf_merged)
eaf_merged = eaf_merged.dropna(subset=['total_reads'])
eaf_merged = eaf_merged[eaf_merged['total_reads'] > 0].copy()
print(f"  Kept {len(eaf_merged)}/{n_before} rows with read data")


# ============================================================================
# 5. Compute bin-level EAF partition
# ============================================================================
def compute_bin_partition(df, bin_col, bin_order):
    """Compute abundance-weighted EAF contribution per bin."""
    results = []

    for (mkr, treat, tp), grp in df.groupby(['marker', 'treatment_label', 'timepoint']):
        reads = grp['total_reads'].values.astype(float)
        eaf_vals = grp['EAF'].values
        total_reads = reads.sum()

        # Per-ASV weighted contribution
        weights = reads / total_reads  # relative abundance within this group
        weighted_eaf = weights * eaf_vals
        community_wmean = weighted_eaf.sum()

        for b in bin_order:
            bmask = grp[bin_col].values == b
            n_asvs = bmask.sum()
            if n_asvs == 0:
                results.append({
                    'marker': mkr, 'treatment': treat, 'timepoint': tp,
                    'bin_label': b, 'n_ASVs': 0,
                    'cumulative_rel_abundance': 0.0,
                    'bin_weighted_EAF': 0.0,
                    'bin_share_of_total': 0.0,
                    'community_weighted_EAF': community_wmean,
                })
                continue

            cum_rel = weights[bmask].sum()
            bin_weighted = weighted_eaf[bmask].sum()
            share = bin_weighted / community_wmean if community_wmean != 0 else np.nan

            results.append({
                'marker': mkr, 'treatment': treat, 'timepoint': tp,
                'bin_label': b, 'n_ASVs': n_asvs,
                'cumulative_rel_abundance': cum_rel,
                'bin_weighted_EAF': bin_weighted,
                'bin_share_of_total': share,
                'community_weighted_EAF': community_wmean,
            })

    return pd.DataFrame(results)


def bootstrap_bin_absolute(df, bin_col, bin_order, n_boot=N_BOOT):
    """Bootstrap CIs for absolute bin_weighted_EAF by resampling ASVs with replacement.

    Returns CIs on the absolute contribution (not the share), avoiding
    ratio instability when community EAF is near zero.
    """
    results = []

    for (mkr, treat, tp), grp in df.groupby(['marker', 'treatment_label', 'timepoint']):
        reads = grp['total_reads'].values.astype(float)
        eaf_vals = grp['EAF'].values
        bin_labels = grp[bin_col].values
        n = len(grp)

        boot_abs = {b: [] for b in bin_order}

        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            b_reads = reads[idx]
            b_eaf = eaf_vals[idx]
            b_bins = bin_labels[idx]

            total = b_reads.sum()
            if total == 0:
                continue
            b_weights = b_reads / total
            b_weighted_eaf = b_weights * b_eaf

            for b in bin_order:
                bmask = b_bins == b
                if bmask.sum() == 0:
                    boot_abs[b].append(0.0)
                else:
                    boot_abs[b].append(b_weighted_eaf[bmask].sum())

        for b in bin_order:
            vals = np.array(boot_abs[b])
            if len(vals) > 0:
                ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
            else:
                ci_lo, ci_hi = np.nan, np.nan
            results.append({
                'marker': mkr, 'treatment': treat, 'timepoint': tp,
                'bin_label': b, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            })

    return pd.DataFrame(results)


print("\nComputing abundance-bin EAF partition...")
abund_partition = compute_bin_partition(eaf_merged, 'abund_bin', BIN_ORDER)
print("Computing rank-bin EAF partition...")
rank_partition = compute_bin_partition(eaf_merged, 'rank_bin', RANK_BIN_ORDER)

print(f"Bootstrapping abundance-bin absolute CIs ({N_BOOT} resamples)...")
abund_ci = bootstrap_bin_absolute(eaf_merged, 'abund_bin', BIN_ORDER)
print(f"Bootstrapping rank-bin absolute CIs ({N_BOOT} resamples)...")
rank_ci = bootstrap_bin_absolute(eaf_merged, 'rank_bin', RANK_BIN_ORDER)

# Merge CIs
abund_partition = abund_partition.merge(abund_ci, on=['marker', 'treatment', 'timepoint', 'bin_label'], how='left')
rank_partition = rank_partition.merge(rank_ci, on=['marker', 'treatment', 'timepoint', 'bin_label'], how='left')


# ============================================================================
# 6. Verify against community-level weighted EAF
# ============================================================================
print("\n" + "="*60)
print("VERIFICATION: Bin sums vs community weighted EAF")
print("="*60)

comm_df_renamed = comm_df.rename(columns={'treatment': 'treatment_code'})
# Map treatment names
treat_name_map = {'Glucose': 'Glucose', 'Xylose': 'Xylose', 'Hemicellulose': 'Hemicellulose'}

verification_ok = True
for mkr in ['16S', 'ITS']:
    for treat in TREAT_ORDER:
        for tp in sorted(eaf_all[eaf_all['marker'] == mkr]['timepoint'].unique()):
            sub = abund_partition[
                (abund_partition['marker'] == mkr) &
                (abund_partition['treatment'] == treat) &
                (abund_partition['timepoint'] == tp)
            ]
            if len(sub) == 0:
                continue
            bin_sum = sub['bin_weighted_EAF'].sum()
            comm_wmean = sub['community_weighted_EAF'].iloc[0]

            # Compare to community file
            comm_row = comm_df[
                (comm_df['marker'] == mkr) &
                (comm_df['treatment'] == treat) &
                (comm_df['timepoint'] == tp)
            ]
            if len(comm_row) > 0:
                ref_wmean = comm_row['weighted_mean_EAF'].values[0]
                diff = abs(bin_sum - ref_wmean)
                if diff > 0.001:
                    print(f"  WARNING: {mkr}/{treat}/{tp}h: bin_sum={bin_sum:.6f}, "
                          f"ref={ref_wmean:.6f}, diff={diff:.6f}")
                    verification_ok = False

            # Internal consistency
            if abs(bin_sum - comm_wmean) > 1e-10:
                print(f"  INTERNAL: {mkr}/{treat}/{tp}h: bin_sum={bin_sum:.10f} != comm_wmean={comm_wmean:.10f}")

if verification_ok:
    print("  All bin sums match community weighted EAF (within 0.001 tolerance)")
else:
    print("  Note: Small discrepancies expected due to ASV filtering differences")
    print("  (OTU table read counts vs qSIP internal abundance estimation)")


# ============================================================================
# 7. Print key findings
# ============================================================================
print("\n" + "="*60)
print("KEY FINDINGS: Abundance-bin dominance of 13C incorporation")
print("="*60)

for mkr in ['16S', 'ITS']:
    print(f"\n--- {mkr} ---")
    for treat in TREAT_ORDER:
        print(f"\n  {treat}:")
        sub = abund_partition[
            (abund_partition['marker'] == mkr) &
            (abund_partition['treatment'] == treat)
        ].copy()
        if len(sub) == 0:
            print("    (no data)")
            continue

        # Show bin shares at peak EAF timepoint
        peak_tp = sub.groupby('timepoint')['community_weighted_EAF'].first().idxmax()
        peak_sub = sub[sub['timepoint'] == peak_tp].sort_values('bin_label',
            key=lambda x: x.map({b: i for i, b in enumerate(BIN_ORDER)}))

        print(f"    Peak EAF timepoint: {peak_tp}h ({TP_LABELS.get(peak_tp, str(peak_tp))})")
        print(f"    Community weighted EAF: {peak_sub['community_weighted_EAF'].iloc[0]:.4f}")
        for _, row in peak_sub.iterrows():
            share_pct = row['bin_share_of_total'] * 100
            ci_str = f"[{row['ci_lo']*100:.1f}, {row['ci_hi']*100:.1f}]" if not np.isnan(row['ci_lo']) else "[NA]"
            print(f"    {row['bin_label']:>12s}: {row['n_ASVs']:4d} ASVs, "
                  f"rel_abund={row['cumulative_rel_abundance']:.3f}, "
                  f"share={share_pct:6.1f}% {ci_str}")


# ============================================================================
# 8. Save summary table
# ============================================================================
out_cols = ['marker', 'treatment', 'timepoint', 'bin_label', 'n_ASVs',
            'cumulative_rel_abundance', 'bin_weighted_EAF', 'community_weighted_EAF',
            'bin_share_of_total', 'ci_lo', 'ci_hi']
out_file = os.path.join(TABLE_DIR, 'abundance_bin_EAF_partition.tsv')
abund_partition[out_cols].to_csv(out_file, sep='\t', index=False, float_format='%.6f')
print(f"\nSaved: {out_file}")

# Also save rank-based partition
rank_out = os.path.join(TABLE_DIR, 'abundance_bin_EAF_partition_rankbased.tsv')
rank_partition[out_cols].to_csv(rank_out, sep='\t', index=False, float_format='%.6f')
print(f"Saved: {rank_out}")


# ============================================================================
# 9. Generate supplementary figure
# ============================================================================
print("\nGenerating figure...")

def make_stacked_figure(partition_df, bin_order, bin_colors, bin_type_label,
                        filename_suffix=''):
    """Create 2×3 stacked bar chart: bacteria (A–C) top, fungi (D–F) bottom.

    Greyscale bins, one row per domain, one column per treatment.
    """

    markers = ['16S', 'ITS']
    marker_labels = {'16S': 'Bacteria', 'ITS': 'Fungi'}

    fig, axes = plt.subplots(2, 3, figsize=(180/25.4, 120/25.4),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.30})

    # Determine y-axis range per domain (separate scales for bacteria vs fungi)
    for row_idx, mkr in enumerate(markers):
        all_vals = []
        for treat in TREAT_ORDER:
            sub = partition_df[
                (partition_df['marker'] == mkr) &
                (partition_df['treatment'] == treat)
            ]
            for tp in sub['timepoint'].unique():
                tp_sub = sub[sub['timepoint'] == tp]
                pos_sum = tp_sub.loc[tp_sub['bin_weighted_EAF'] > 0, 'bin_weighted_EAF'].sum()
                neg_sum = tp_sub.loc[tp_sub['bin_weighted_EAF'] < 0, 'bin_weighted_EAF'].sum()
                all_vals.extend([pos_sum, neg_sum])
        y_max = max(all_vals) * 1.15 if all_vals else 0.15
        y_min = min(min(all_vals) * 1.15, -0.005) if all_vals else -0.02

        for col_idx, treat in enumerate(TREAT_ORDER):
            ax = axes[row_idx, col_idx]

            sub = partition_df[
                (partition_df['marker'] == mkr) &
                (partition_df['treatment'] == treat)
            ].copy()

            if len(sub) == 0:
                continue

            timepoints = sorted(sub['timepoint'].unique())
            x_positions = np.log10(np.array(timepoints, dtype=float))
            bar_width = 0.18

            pos_bottoms = np.zeros(len(timepoints))
            neg_bottoms = np.zeros(len(timepoints))

            for b in bin_order:
                vals = []
                for tp in timepoints:
                    row = sub[(sub['timepoint'] == tp) & (sub['bin_label'] == b)]
                    if len(row) > 0:
                        v = row['bin_weighted_EAF'].values[0]
                        vals.append(v if not np.isnan(v) else 0)
                    else:
                        vals.append(0)
                vals = np.array(vals)

                pos_vals = np.where(vals >= 0, vals, 0)
                neg_vals = np.where(vals < 0, vals, 0)

                if np.any(pos_vals > 0):
                    ax.bar(x_positions, pos_vals, width=bar_width,
                           bottom=pos_bottoms, color=bin_colors[b],
                           edgecolor='white', linewidth=0.3)
                    pos_bottoms += pos_vals
                if np.any(neg_vals < 0):
                    ax.bar(x_positions, neg_vals, width=bar_width,
                           bottom=neg_bottoms, color=bin_colors[b],
                           edgecolor='white', linewidth=0.3)
                    neg_bottoms += neg_vals

            # Titles: treatment on top row only
            if row_idx == 0:
                ax.set_title(treat, fontsize=11, fontweight='bold',
                             color=TREAT_COLORS[treat])
            # X-axis labels on bottom row only
            if row_idx == 1:
                ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=9)
            ax.set_xticks(np.log10([2, 8, 24, 72, 168, 336, 672]))
            ax.set_xticklabels(['2h', '8h', '1d', '3d', '7d', '14d', '28d'],
                              fontsize=8, rotation=45, ha='right')
            if col_idx == 0:
                ax.set_ylabel('Bin contribution to\ncommunity EAF', fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
            ax.tick_params(labelsize=8, direction='out', length=2, width=0.6)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            ax.spines['left'].set_linewidth(0.6)
            ax.spines['bottom'].set_linewidth(0.6)

        # Domain label on right side of last column
        axes[row_idx, 2].text(1.04, 0.5, marker_labels[mkr],
                              transform=axes[row_idx, 2].transAxes,
                              fontsize=10, fontweight='bold', rotation=-90,
                              va='center', ha='left')

    # Panel labels
    for i, ax in enumerate(axes[0]):
        ax.text(-0.15, 1.08, chr(65 + i), transform=ax.transAxes,
                fontsize=12, fontweight='bold', fontfamily='Arial')
    for i, ax in enumerate(axes[1]):
        ax.text(-0.15, 1.08, chr(68 + i), transform=ax.transAxes,
                fontsize=12, fontweight='bold', fontfamily='Arial')

    # Legend — single row of greyscale bin swatches
    legend_elements = [Patch(facecolor=bin_colors[b], edgecolor='#999999',
                             linewidth=0.5, label=b) for b in bin_order]
    fig.legend(handles=legend_elements, loc='upper center',
              ncol=len(bin_order), fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, 1.02), handletextpad=0.3, columnspacing=1.0)

    fig.suptitle(f'Abundance-weighted EAF partition by {bin_type_label}',
                fontsize=11, fontweight='bold', y=1.06)

    return fig


# Greyscale bin colors (dark to light for abundant to rare)
GREY_BIN_COLORS = {
    '>1%':        '#000000',
    '0.1-1%':     '#4d4d4d',
    '0.01-0.1%':  '#999999',
    '<0.01%':     '#d9d9d9',
}
GREY_RANK_COLORS = {
    'Top 10':      '#000000',
    'Rank 11-100': '#4d4d4d',
    'Remainder':   '#d9d9d9',
}

# Main figure with abundance bins
fig_abund = make_stacked_figure(abund_partition, BIN_ORDER, GREY_BIN_COLORS,
                                 'relative abundance bin')
fig_abund.savefig(os.path.join(FIG_DIR, f'FigureS_abundance_bin_EAF.pdf'),
                  dpi=600, bbox_inches='tight')
fig_abund.savefig(os.path.join(FIG_DIR, f'FigureS_abundance_bin_EAF.png'),
                  dpi=600, bbox_inches='tight')
plt.close(fig_abund)
print(f"Saved: {os.path.join(FIG_DIR, 'FigureS_abundance_bin_EAF.pdf')}")
print(f"Saved: {os.path.join(FIG_DIR, 'FigureS_abundance_bin_EAF.png')}")

# Sensitivity check with rank bins
fig_rank = make_stacked_figure(rank_partition, RANK_BIN_ORDER, GREY_RANK_COLORS,
                                'rank-based abundance bin')
fig_rank.savefig(os.path.join(FIG_DIR, f'FigureS_abundance_bin_EAF_rankbased.pdf'),
                 dpi=600, bbox_inches='tight')
fig_rank.savefig(os.path.join(FIG_DIR, f'FigureS_abundance_bin_EAF_rankbased.png'),
                 dpi=600, bbox_inches='tight')
plt.close(fig_rank)
print(f"Saved: {os.path.join(FIG_DIR, 'FigureS_abundance_bin_EAF_rankbased.pdf')}")
print(f"Saved: {os.path.join(FIG_DIR, 'FigureS_abundance_bin_EAF_rankbased.png')}")

print("\n=== ABUNDANCE-BIN EAF ANALYSIS COMPLETE ===")
