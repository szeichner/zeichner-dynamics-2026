#!/usr/bin/env python3
"""
Supplementary Figure — k=5 functional gene bubble heatmap.

Shows dominant functional categories (subcategories) shaping each k=5 program.
Bubble color = mean log2FC vs control (RdBu_r diverging).
Bubble size = number of genes in that subcategory-program combination.

Output: FigureS_k5_functional_bubble.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

TP_HOURS_MONO = [2, 8, 24, 72, 168, 336]
TP_HOURS_HEMI = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

PROG_COLORS_K5 = {
    0: '#D95F02',
    1: '#1B9E77',
    2: '#7570B3',
    3: '#E7298A',
    4: '#66A61E',
}

DESCRIPTIVE_LABELS_K5 = [
    'Early growth burst',
    'Mid-phase plateau',
    'Late succession rise',
    'Polymer-responsive',
    'Baseline suppressed',
]

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')

# Load subcategory assignments
subcat = pd.read_csv(os.path.join(TABLES_DIR, 'TableS9_program_subcategory_assignments_v2.tsv'), sep='\t')
subcat_map = subcat[['ko_id', 'subcategory_flat']].drop_duplicates()
assign = assign.merge(subcat_map, on='ko_id', how='left')
assign['subcategory_flat'] = assign['subcategory_flat'].fillna('Other/diverse')

lfc_cols = [c for c in assign.columns if c.startswith('MONOMER_') or c.startswith('HEMICELLULOSE_')]
mono_cols = [c for c in lfc_cols if c.startswith('MONOMER')]
hemi_cols = [c for c in lfc_cols if c.startswith('HEMICELLULOSE')]

# Center and cluster at k=5
lfc = assign[lfc_cols].copy()
lfc_centered = lfc.subtract(lfc.mean(axis=1), axis=0)
X = lfc_centered.values

km5 = KMeans(n_clusters=5, random_state=42, n_init=20)
labels_k5 = km5.fit_predict(X)
assign['k5_cluster'] = labels_k5

# Sort clusters by monomer peak time
cluster_peaks = []
for c in range(5):
    mask = labels_k5 == c
    traj = assign.loc[mask, mono_cols].mean(axis=0)
    peak_idx = traj.values.argmax()
    cluster_peaks.append((c, peak_idx, mask.sum()))
cluster_peaks.sort(key=lambda x: x[1])

# Create sorted cluster mapping
sorted_clusters = [cp[0] for cp in cluster_peaks]
cluster_name_map = {sorted_clusters[i]: DESCRIPTIVE_LABELS_K5[i] for i in range(5)}
cluster_color_map = {sorted_clusters[i]: PROG_COLORS_K5[i] for i in range(5)}

assign['k5_label'] = assign['k5_cluster'].map(cluster_name_map)

# ============================================================================
# For each k=5 cluster, find top subcategories
# ============================================================================
TOP_CATS_PER_CLUSTER = 5
exclude_cats = ['Other/diverse']

all_rows = []
for ci, orig_c in enumerate(sorted_clusters):
    csub = assign[assign['k5_cluster'] == orig_c]
    cat_counts = csub['subcategory_flat'].value_counts()
    # Remove excluded categories
    cat_counts = cat_counts[~cat_counts.index.isin(exclude_cats)]
    top_cats = cat_counts.head(TOP_CATS_PER_CLUSTER).index.tolist()

    for cat in top_cats:
        cat_sub = csub[csub['subcategory_flat'] == cat]
        n_genes = len(cat_sub)
        # Compute mean LFC per timepoint (monomer)
        for tp_col in mono_cols:
            tp_h = int(tp_col.replace('MONOMER_t', ''))
            mean_lfc = cat_sub[tp_col].mean()
            all_rows.append({
                'cluster_idx': ci,
                'cluster_label': DESCRIPTIVE_LABELS_K5[ci],
                'subcategory': cat,
                'treatment': 'Monomer',
                'timepoint': tp_h,
                'mean_lfc': mean_lfc,
                'n_genes': n_genes,
            })
        for tp_col in hemi_cols:
            tp_h = int(tp_col.replace('HEMICELLULOSE_t', ''))
            mean_lfc = cat_sub[tp_col].mean()
            all_rows.append({
                'cluster_idx': ci,
                'cluster_label': DESCRIPTIVE_LABELS_K5[ci],
                'subcategory': cat,
                'treatment': 'Hemicellulose',
                'timepoint': tp_h,
                'mean_lfc': mean_lfc,
                'n_genes': n_genes,
            })

bubble_df = pd.DataFrame(all_rows)

# ============================================================================
# Generate figure: 5 rows (one per k=5 cluster), 2 cols (mono, hemi)
# ============================================================================
print("Generating k=5 functional bubble heatmap...")

n_programs = 5
fig, axes = plt.subplots(n_programs, 2, figsize=(200 * MM, n_programs * 50 * MM))
fig.subplots_adjust(hspace=0.50, wspace=0.30, left=0.38, right=0.93,
                    top=0.94, bottom=0.11)

norm = mcolors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
cmap = plt.get_cmap('RdBu_r')
max_n = bubble_df['n_genes'].max()
min_n = bubble_df['n_genes'].min()
# Scale bubble area linearly with n_genes for clear size differentiation
# Range: 15 (smallest) to 350 (largest)
SIZE_MIN = 15
SIZE_MAX = 350

for ci in range(n_programs):
    cluster_label = DESCRIPTIVE_LABELS_K5[ci]
    cluster_data = bubble_df[bubble_df['cluster_idx'] == ci]
    cats = cluster_data['subcategory'].unique().tolist()

    for ti, (treat, tp_list) in enumerate([
        ('Monomer', TP_HOURS_MONO),
        ('Hemicellulose', TP_HOURS_HEMI),
    ]):
        ax = axes[ci, ti]
        treat_data = cluster_data[cluster_data['treatment'] == treat]

        for ri, cat in enumerate(cats):
            cat_data = treat_data[treat_data['subcategory'] == cat]
            for _, row in cat_data.iterrows():
                tp = row['timepoint']
                xi = tp_list.index(tp) if tp in tp_list else -1
                if xi < 0:
                    continue
                # Linear area scaling for clear size differentiation
                frac = (row['n_genes'] - min_n) / (max_n - min_n) if max_n > min_n else 0.5
                size = SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)
                color = cmap(norm(row['mean_lfc']))
                ax.scatter(xi, ri, s=size, c=[color], edgecolors='black',
                           linewidth=0.3, zorder=3, clip_on=False)

        ax.set_yticks(range(len(cats)))
        if ti == 0:
            # Truncate long category names
            ylabels = [c[:30] + '...' if len(c) > 30 else c for c in cats]
            ax.set_yticklabels(ylabels, fontsize=7, fontfamily=FF)
        else:
            ax.set_yticklabels([])

        ax.set_xticks(range(len(tp_list)))
        if ci == n_programs - 1:
            ax.set_xticklabels([TP_LABELS[t] for t in tp_list], fontsize=8, fontfamily=FF)
            ax.set_xlabel('Time', fontsize=10, fontfamily=FF)
        else:
            ax.set_xticklabels([])

        ax.set_ylim(-0.7, len(cats) - 0.3)
        ax.set_xlim(-0.7, len(tp_list) - 0.3)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for _sp in ['left', 'bottom']:
            ax.spines[_sp].set_linewidth(0.6)
        ax.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)

        if ci == 0:
            title_color = '#6B7C2E' if ti == 0 else '#7B3F6E'
            ax.set_title('Monomers' if ti == 0 else 'Hemicellulose',
                         fontsize=11, fontfamily=FF, fontweight='bold', color=title_color)

        if ti == 0:
            # Program label on far left
            n_in_cluster = cluster_peaks[ci][2]
            ax.text(-1.0, 0.5, f'{cluster_label}\n(n={n_in_cluster})',
                    transform=ax.transAxes, fontsize=7, fontfamily=FF,
                    fontweight='bold', color=PROG_COLORS_K5[ci],
                    ha='center', va='center', rotation=90, linespacing=1.2)

# Colorbar
cax = fig.add_axes([0.30, 0.03, 0.30, 0.012])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('Mean $\\log_2$FC vs control', fontsize=8, fontfamily=FF)
cb.ax.tick_params(labelsize=8)

# Size legend — use actual representative values from the data (range: 15–119)
size_legend_ax = fig.add_axes([0.68, 0.015, 0.24, 0.055])
size_legend_ax.set_xlim(0, 1)
size_legend_ax.set_ylim(0, 1)
size_legend_ax.axis('off')
size_legend_ax.text(0.5, 0.95, 'Genes per subcategory', ha='center', va='top', fontsize=7,
                    fontfamily=FF, fontweight='bold')
for i, n_ex in enumerate([15, 50, 119]):
    frac = (n_ex - min_n) / (max_n - min_n) if max_n > min_n else 0.5
    sz = SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)
    x_pos = 0.15 + i * 0.35
    size_legend_ax.scatter(x_pos, 0.45, s=sz, c='white', edgecolors='black', linewidth=0.5)
    size_legend_ax.text(x_pos, 0.05, f'n={n_ex}', ha='center', va='top', fontsize=7, fontfamily=FF)

fig.suptitle('k = 5 programs: top functional categories per cluster',
             fontsize=11, fontfamily=FF, fontweight='bold', y=0.97)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS_k5_functional_bubble.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS_k5_functional_bubble.pdf/.png")
