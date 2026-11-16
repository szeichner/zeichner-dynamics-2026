#!/usr/bin/env python3
"""
compute_pairwise_permanova_all.py
==================================

Pairwise PERMANOVA on every combination of treatment × timepoint groups.
Each group is e.g. "GLUCOSE_2h", "CELLULOSE_168h", etc.

Also generates summary figures:
  1. Heatmap of pairwise R² values
  2. Heatmap of pairwise p-values (BH-corrected)
  3. Per-timepoint PERMANOVA R² bar chart (Monomer vs Hemicellulose)
  4. Centroid-to-centroid distances over time

Outputs:
  - data/derived/permanova_all_pairwise.tsv
  - figures/output/FigureS_pairwise_permanova_heatmap.pdf/.png
  - figures/output/FigureS_multivariate_summary.pdf/.png
"""

import os, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

TC = {'GLUCOSE': '#6B7C2E', 'XYLOSE': '#C8960C', 'CELLULOSE': '#7B3F6E',
      'Monomer': '#6B7C2E', 'Hemicellulose': '#7B3F6E'}

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
npz = np.load(os.path.join(DERIVED_DIR, 'bc_distance_matrix.npz'), allow_pickle=True)
bc_full = npz['distances']
sample_ids = npz['sample_ids'].astype(str)

coords = pd.read_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'), sep='\t')
coords['taxon_id'] = coords['taxon_id'].astype(str)

meta = coords[['taxon_id', 'sample_name', 'treatment', 'time_hours', 'replicate']].copy()

# Create group labels
meta['group'] = meta['treatment'] + '_' + meta['time_hours'].astype(str) + 'h'

# Only main treatments
main_mask = meta.treatment.isin(['GLUCOSE', 'XYLOSE', 'CELLULOSE'])
main_meta = meta[main_mask].copy()

groups_unique = sorted(main_meta.group.unique())
print(f"  {len(main_meta)} samples in {len(groups_unique)} treatment×timepoint groups")
for g in groups_unique:
    n = (main_meta.group == g).sum()
    print(f"    {g}: n={n}")


def permanova_fast(dm, groups, n_perms=999, seed=42):
    """PERMANOVA with fewer perms for the large pairwise matrix."""
    rng = np.random.RandomState(seed)
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)
    if k < 2 or n < 3:
        return np.nan, np.nan, np.nan

    ss_total = np.sum(dm ** 2) / (2 * n)

    def ss_within(grps):
        ss_w = 0.0
        for g in unique_groups:
            mask = grps == g
            n_g = mask.sum()
            if n_g > 1:
                ss_w += np.sum(dm[np.ix_(mask, mask)] ** 2) / (2 * n_g)
        return ss_w

    ss_w_obs = ss_within(groups)
    ss_b_obs = ss_total - ss_w_obs
    if ss_w_obs == 0:
        return np.nan, np.nan, np.nan
    f_obs = (ss_b_obs / (k - 1)) / (ss_w_obs / (n - k))
    r2 = ss_b_obs / ss_total

    n_ge = 1
    for _ in range(n_perms):
        perm = rng.permutation(groups)
        ss_w_p = ss_within(perm)
        ss_b_p = ss_total - ss_w_p
        f_p = (ss_b_p / (k - 1)) / (ss_w_p / (n - k))
        if f_p >= f_obs:
            n_ge += 1
    return f_obs, r2, n_ge / (n_perms + 1)


# ============================================================================
# All pairwise PERMANOVA
# ============================================================================
print("\nRunning all pairwise PERMANOVA (999 perms each)...")

pairs = list(itertools.combinations(groups_unique, 2))
print(f"  {len(pairs)} pairwise comparisons")

results = []
for i, (g1, g2) in enumerate(pairs):
    idx1 = np.where(main_meta.group == g1)[0]
    idx2 = np.where(main_meta.group == g2)[0]
    all_idx = np.concatenate([idx1, idx2])

    # Map to positions in the full BC matrix
    global_idx = main_meta.index[all_idx].values
    # Need to find positions in sample_ids
    pos = [np.where(sample_ids == meta.loc[gi, 'taxon_id'])[0][0] for gi in global_idx]
    dm_pair = bc_full[np.ix_(pos, pos)]

    grp_labels = np.array([g1] * len(idx1) + [g2] * len(idx2))
    f_val, r2, p_val = permanova_fast(dm_pair, grp_labels, n_perms=999, seed=42)

    results.append({
        'group1': g1, 'group2': g2,
        'n1': len(idx1), 'n2': len(idx2),
        'F': f_val, 'R2': r2, 'p_value': p_val,
    })
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(pairs)} done...")

print(f"  All {len(pairs)} comparisons done.")

res_df = pd.DataFrame(results)

# BH correction
valid = ~res_df.p_value.isna()
reject = np.full(len(res_df), False)
padj = np.full(len(res_df), np.nan)
if valid.sum() > 0:
    rej, pa, _, _ = multipletests(res_df.loc[valid, 'p_value'].values, method='fdr_bh')
    reject[valid] = rej
    padj[valid] = pa
res_df['padj'] = padj
res_df['significant'] = reject

n_sig = reject.sum()
print(f"  Significant after BH correction: {n_sig}/{len(res_df)}")

res_df.to_csv(os.path.join(DERIVED_DIR, 'permanova_all_pairwise.tsv'),
              sep='\t', index=False)
print(f"  Saved permanova_all_pairwise.tsv")

# ============================================================================
# Figure 1: Pairwise R² heatmap
# ============================================================================
print("\nGenerating pairwise PERMANOVA heatmap...")

# Order groups by treatment then timepoint
def sort_key(g):
    treat, tp = g.rsplit('_', 1)
    tp_num = int(tp.replace('h', ''))
    treat_order = {'GLUCOSE': 0, 'XYLOSE': 1, 'CELLULOSE': 2}
    return (treat_order.get(treat, 9), tp_num)

groups_ordered = sorted(groups_unique, key=sort_key)
n_groups = len(groups_ordered)

# Build R² and significance matrices
r2_mat = np.full((n_groups, n_groups), np.nan)
sig_mat = np.full((n_groups, n_groups), False)

g_to_i = {g: i for i, g in enumerate(groups_ordered)}
for _, row in res_df.iterrows():
    i = g_to_i.get(row.group1)
    j = g_to_i.get(row.group2)
    if i is not None and j is not None:
        r2_mat[i, j] = row.R2
        r2_mat[j, i] = row.R2
        sig_mat[i, j] = row.significant
        sig_mat[j, i] = row.significant

# Diagonal = 0
np.fill_diagonal(r2_mat, 0)

fig_heat, ax_heat = plt.subplots(figsize=(180 * MM, 160 * MM))

im = ax_heat.imshow(r2_mat, cmap='YlOrRd', interpolation='nearest',
                     vmin=0, vmax=0.8, aspect='equal')

# Labels
short_labels = []
for g in groups_ordered:
    treat, tp = g.rsplit('_', 1)
    tp_num = int(tp.replace('h', ''))
    tp_lab = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}
    t_short = {'GLUCOSE': 'Glu', 'XYLOSE': 'Xyl', 'CELLULOSE': 'Hemi'}
    short_labels.append(f"{t_short[treat]} {tp_lab.get(tp_num, tp)}")

ax_heat.set_xticks(range(n_groups))
ax_heat.set_xticklabels(short_labels, fontsize=5.5, fontfamily=FF, rotation=90)
ax_heat.set_yticks(range(n_groups))
ax_heat.set_yticklabels(short_labels, fontsize=5.5, fontfamily=FF)

# Color y/x labels by treatment
for i, g in enumerate(groups_ordered):
    treat = g.rsplit('_', 1)[0]
    color = TC.get(treat, 'black')
    ax_heat.get_yticklabels()[i].set_color(color)
    ax_heat.get_xticklabels()[i].set_color(color)

# Mark significant cells with dots
for i in range(n_groups):
    for j in range(n_groups):
        if i != j and sig_mat[i, j]:
            ax_heat.plot(j, i, 'k.', markersize=2, zorder=5)

# Annotate R² values
for i in range(n_groups):
    for j in range(n_groups):
        if i != j and not np.isnan(r2_mat[i, j]):
            val = r2_mat[i, j]
            txt_color = 'white' if val > 0.5 else 'black'
            ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=3.5, fontfamily=FF, color=txt_color)

# Treatment block dividers
# Find where treatment changes
treat_breaks = []
prev_treat = None
for i, g in enumerate(groups_ordered):
    treat = g.rsplit('_', 1)[0]
    if prev_treat and treat != prev_treat:
        treat_breaks.append(i - 0.5)
    prev_treat = treat

for b in treat_breaks:
    ax_heat.axhline(b, color='white', linewidth=1.5, zorder=4)
    ax_heat.axvline(b, color='white', linewidth=1.5, zorder=4)

cb = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.02)
cb.set_label('PERMANOVA R²', fontsize=7, fontfamily=FF)
cb.ax.tick_params(labelsize=6)

ax_heat.set_title('Pairwise PERMANOVA R² (all treatment × timepoint pairs)',
                   fontsize=8, fontfamily=FF, fontweight='bold', pad=8)
fig_heat.text(0.5, 0.01, 'Black dots = significant after BH correction (padj < 0.05)',
              fontsize=5.5, fontfamily=FF, ha='center', color='#666666')

for sp in ax_heat.spines.values():
    sp.set_visible(False)

for ext in ['pdf', 'png']:
    fig_heat.savefig(os.path.join(FIGURES_DIR, f'FigureS_pairwise_permanova_heatmap.{ext}'),
                     format=ext, dpi=600, bbox_inches='tight')
plt.close(fig_heat)
print("  Saved FigureS_pairwise_permanova_heatmap.pdf/.png")

# ============================================================================
# Figure 2: Multi-panel summary figure
# ============================================================================
print("\nGenerating multivariate summary figure...")

fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 65 * MM))

# --- Panel A: Per-timepoint PERMANOVA R² (Monomer vs Hemicellulose) ---
ax_a = axes[0]
tp_perm = pd.read_csv(os.path.join(DERIVED_DIR, 'permanova_per_timepoint.tsv'), sep='\t')

tp_labels_map = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d'}
x_pos = range(len(tp_perm))
bars = ax_a.bar(x_pos, tp_perm.R2.values, color='#8B7BB5', edgecolor='black',
                linewidth=0.5, width=0.7, zorder=3)

# Stars for significance
for i, (_, row) in enumerate(tp_perm.iterrows()):
    if row.significant:
        ax_a.text(i, row.R2 + 0.015, '*', ha='center', va='bottom',
                  fontsize=8, fontfamily=FF, fontweight='bold', color='black')

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([tp_labels_map.get(int(t), str(int(t))) for t in tp_perm.timepoint_hours],
                      fontsize=6, fontfamily=FF)
ax_a.set_xlabel('Time', fontsize=7, fontfamily=FF)
ax_a.set_ylabel('PERMANOVA R²\n(Monomer vs Hemicellulose)', fontsize=6.5, fontfamily=FF)
ax_a.set_ylim(0, 0.7)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.tick_params(labelsize=6)
ax_a.text(-0.2, 1.05, 'A', transform=ax_a.transAxes, fontsize=9,
          fontweight='bold', va='top', fontfamily=FF)

# --- Panel B: Centroid-to-centroid BC distances over time ---
ax_b = axes[1]
cent = pd.read_csv(os.path.join(DERIVED_DIR, 'centroid_distances_over_time.tsv'), sep='\t')

tp_x = {2: 0, 8: 1, 24: 2, 72: 3, 168: 4, 336: 5, 672: 6}
tp_labs = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']

for label, color, ls in [
    ('Glucose vs Hemicellulose', '#6B7C2E', '-'),
    ('Xylose vs Hemicellulose', '#C8960C', '-'),
    ('Glucose vs Xylose', '#888888', '--'),
]:
    sub = cent[cent.comparison == label].sort_values('timepoint_hours')
    xs = [tp_x[int(t)] for t in sub.timepoint_hours]
    ax_b.plot(xs, sub.bc_distance.values, color=color, linewidth=1.5,
              linestyle=ls, marker='o', markersize=4, markeredgecolor='black',
              markeredgewidth=0.3, zorder=3, label=label.replace(' vs ', '\nvs '))

ax_b.set_xticks(range(len(tp_labs)))
ax_b.set_xticklabels(tp_labs, fontsize=6, fontfamily=FF)
ax_b.set_xlabel('Time', fontsize=7, fontfamily=FF)
ax_b.set_ylabel('BC distance\nbetween centroids', fontsize=6.5, fontfamily=FF)
ax_b.legend(fontsize=4.5, loc='upper left', frameon=False, handletextpad=0.3)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(labelsize=6)
ax_b.text(-0.2, 1.05, 'B', transform=ax_b.transAxes, fontsize=9,
          fontweight='bold', va='top', fontfamily=FF)

# --- Panel C: Variance partitioning pie/bar ---
ax_c = axes[2]
inter = pd.read_csv(os.path.join(DERIVED_DIR, 'permanova_treatment_time_interaction.tsv'), sep='\t')

treat_r2 = inter.loc[inter.model == 'treatment', 'R2'].values[0]
time_r2 = inter.loc[inter.model == 'time', 'R2'].values[0]
interact_r2 = inter.loc[inter.model == 'interaction_residual', 'R2'].values[0]
residual_r2 = 1.0 - treat_r2 - time_r2 - interact_r2

components = ['Treatment', 'Time', 'Treatment\n× Time', 'Residual']
values = [treat_r2, time_r2, interact_r2, residual_r2]
colors_bar = ['#7B3F6E', '#C8960C', '#B8A9C9', '#DDDDDD']

bars = ax_c.barh(range(len(components)), values, color=colors_bar,
                 edgecolor='black', linewidth=0.5, height=0.6, zorder=3)

for i, (v, label) in enumerate(zip(values, components)):
    ax_c.text(v + 0.01, i, f'{v:.1%}', va='center', ha='left',
              fontsize=6, fontfamily=FF, fontweight='bold')

ax_c.set_yticks(range(len(components)))
ax_c.set_yticklabels(components, fontsize=6.5, fontfamily=FF)
ax_c.set_xlabel('Proportion of variance (R²)', fontsize=7, fontfamily=FF)
ax_c.set_xlim(0, 0.55)
ax_c.invert_yaxis()
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.tick_params(labelsize=6)
ax_c.text(-0.25, 1.05, 'C', transform=ax_c.transAxes, fontsize=9,
          fontweight='bold', va='top', fontfamily=FF)

fig.tight_layout(w_pad=3)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS_multivariate_summary.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("  Saved FigureS_multivariate_summary.pdf/.png")

print("\nDone.")
