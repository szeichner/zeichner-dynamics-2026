#!/usr/bin/env python3
"""
Supplementary Figure — Genus-level qSIP EAF heatmaps.

Shows top ~20 genera by abundance-weighted EAF for 16S and ITS.
Uses sequential Reds colormap (negative EAF clipped to 0). Weighted by n_ASVs.

Output: FigureS_qsip_genus_heatmaps.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
QSIP_DIR = os.path.join(BASE, 'data', 'qSIP')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

QSIP_TC = {'Glucose': '#6B7C2E', 'Xylose': '#C8960C', 'Hemicellulose': '#7B3F6E'}

TOP_N_16S = 100
TOP_N_ITS = 50


def plot_genus_heatmap(ax, marker, panel_label, title, genus_df, top_n, vmax=0.17):
    gen = genus_df[genus_df['marker'] == marker].copy()

    # Compute abundance-weighted mean EAF across all treatments/timepoints
    gen['weighted_eaf'] = gen['mean_EAF'] * gen['n_ASVs']
    genus_score = gen.groupby('genus_label').agg(
        total_weighted_eaf=('weighted_eaf', 'sum'),
        total_asvs=('n_ASVs', 'sum'),
        mean_eaf=('mean_EAF', 'mean'),
    )
    genus_score['abund_weighted_eaf'] = genus_score['total_weighted_eaf'] / genus_score['total_asvs']

    # Filter: need at least some ASVs
    min_asvs = 5 if marker == 'ITS' else 10
    genus_score = genus_score[genus_score['total_asvs'] >= min_asvs]

    # Sort by absolute abundance-weighted EAF, take top N
    genus_score['abs_score'] = genus_score['abund_weighted_eaf'].abs()
    top_genera = genus_score.nlargest(top_n, 'abs_score').index.tolist()

    # Re-sort by mean EAF for display
    genera_ordered = genus_score.loc[top_genera].sort_values('abund_weighted_eaf', ascending=True).index.tolist()

    gen = gen[gen['genus_label'].isin(genera_ordered)]

    treats = ['Glucose', 'Xylose', 'Hemicellulose']
    treat_tps = {t: sorted(gen[gen['treatment_label'] == t]['timepoint'].unique()) for t in treats}

    all_cols = []
    for t in treats:
        for tp in treat_tps[t]:
            all_cols.append((tp, t))

    n_cols, n_rows = len(all_cols), len(genera_ordered)
    mat = np.full((n_rows, n_cols), np.nan)
    for ci, (tp, treat) in enumerate(all_cols):
        for ri, gname in enumerate(genera_ordered):
            vals = gen[(gen['genus_label'] == gname) & (gen['treatment_label'] == treat) &
                       (gen['timepoint'] == tp)]['mean_EAF']
            if len(vals) > 0:
                mat[ri, ci] = vals.iloc[0]

    # Clip negative EAF values to zero (negative values are measurement noise,
    # not biologically meaningful incorporation)
    mat = np.where(np.isnan(mat), np.nan, np.clip(mat, 0, None))

    # Auto-detect vmax from data
    actual_max = np.nanmax(mat)
    actual_vmax = max(vmax, actual_max * 1.05) if not np.isnan(actual_max) else vmax
    im = ax.imshow(mat, aspect='auto', cmap='Reds', vmin=0, vmax=actual_vmax,
                   interpolation='nearest')

    # Clean up genus labels: strip prefixes, simplify Incertae_sedis
    ylabels = []
    for g in genera_ordered:
        # Remove g__ and f__ prefixes
        g_clean = g.replace('g__', '').replace('f__', '')
        # Simplify "_gen_Incertae_sedis (xxx_fam_Incertae_sedis)" patterns
        g_clean = g_clean.replace('_gen_Incertae_sedis', ' (inc. sed.)')
        g_clean = g_clean.replace('_fam_Incertae_sedis', '')
        # Remove redundant parenthetical if it just repeats inc. sed.
        g_clean = g_clean.replace(' ( (inc. sed.))', '')
        g_clean = g_clean.replace(' ((inc. sed.))', '')
        # Clean up double spaces and trailing parens
        g_clean = g_clean.replace('(inc. sed.) ()', '(inc. sed.)')
        g_clean = g_clean.replace(' ()', '').strip()
        # Replace underscores with spaces
        g_clean = g_clean.replace('_', ' ')
        ylabels.append(g_clean)
    ax.set_yticks(range(n_rows))
    label_fontsize = 6.5 if marker == 'ITS' else 6
    ax.set_yticklabels(ylabels, fontsize=label_fontsize, fontfamily=FF)

    tp_map = {2:'2h', 8:'8h', 24:'1d', 72:'3d', 168:'7d', 336:'14d', 672:'28d'}
    xlabels = [tp_map.get(int(tp), str(int(tp))) for tp, _ in all_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, fontsize=8, fontfamily=FF, rotation=45, ha='right')

    col_offset = 0
    for ti, treat in enumerate(treats):
        n_tp = len(treat_tps[treat])
        center = col_offset + (n_tp - 1) / 2
        ax.text(center, -1.8, treat, ha='center', va='bottom', fontsize=8,
                fontfamily=FF, fontweight='bold', color=QSIP_TC[treat])
        if ti < len(treats) - 1:
            ax.axvline(col_offset + n_tp - 0.5, color='black', linewidth=1.0, zorder=5)
        col_offset += n_tp

    # Cell annotations
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = mat[ri, ci]
            if not np.isnan(val):
                color = 'white' if val > vmax * 0.5 else 'black'
                ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                        fontsize=2.5, color=color, fontfamily=FF)

    ax.set_title(title, fontsize=11, fontweight='bold', fontfamily=FF, pad=18)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', fontfamily=FF)
    ax.tick_params(direction='out', length=2.5, width=0.6)
    return im


# Load data
print("Loading qSIP genus data...")
genus_df = pd.read_csv(
    os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_by_genus.tsv'), sep='\t')

print(f"  Total genus records: {len(genus_df)}")
print(f"  16S genera: {genus_df[genus_df['marker']=='16S']['genus_label'].nunique()}")
print(f"  ITS genera: {genus_df[genus_df['marker']=='ITS']['genus_label'].nunique()}")

# Figure
row_h_mm = 6
fig_h = 60 + (TOP_N_16S + TOP_N_ITS) * row_h_mm
fig_h = min(fig_h, 600)

fig, axes = plt.subplots(2, 1, figsize=(180 * MM, fig_h * MM),
                          gridspec_kw={'height_ratios': [TOP_N_16S, TOP_N_ITS],
                                       'hspace': 0.20})

im_16s = plot_genus_heatmap(axes[0], '16S', 'A',
                             '16S (Bacteria/Archaea): top genera by abundance-weighted EAF',
                             genus_df, TOP_N_16S, vmax=0.17)

im_its = plot_genus_heatmap(axes[1], 'ITS', 'B',
                             'ITS (Fungi): top genera by abundance-weighted EAF',
                             genus_df, TOP_N_ITS, vmax=0.17)

# Shared colorbar
cax = fig.add_axes([0.30, 0.02, 0.35, 0.012])
cb = plt.colorbar(im_16s, cax=cax, orientation='horizontal')
cb.set_label('Mean excess atom fraction (EAF)', fontsize=8, fontfamily=FF)
cb.ax.tick_params(labelsize=8)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS_qsip_genus_heatmaps.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS_qsip_genus_heatmaps.pdf/.png")
