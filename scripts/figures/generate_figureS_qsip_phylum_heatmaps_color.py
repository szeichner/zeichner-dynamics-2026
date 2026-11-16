#!/usr/bin/env python3
"""
Supplementary Figure — qSIP phylum-level EAF heatmaps in COLOR (RdBu_r).

Shows BOTH unweighted and weighted (abundance-weighted) mean EAF side by side.

Panel A: 16S unweighted mean EAF
Panel B: 16S weighted mean EAF (EAF * n_ASVs / total_ASVs)
Panel C: ITS unweighted mean EAF
Panel D: ITS weighted mean EAF

Uses RdBu_r diverging colormap centered at 0. No line plots on top.
Output: FigureS3_qSIP_phylum_heatmaps_color.pdf/.png
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


def plot_phylum_heatmap_color(ax, marker, panel_label, title, phylum_df,
                               min_asv_total=100, vmax=0.17, weighted=False):
    phy = phylum_df[phylum_df['marker'] == marker].copy()
    phy_totals = phy.groupby('Phylum')['n_ASVs'].sum()
    threshold = 20 if marker == 'ITS' else min_asv_total
    major = phy_totals[phy_totals >= threshold].index.tolist()
    phy = phy[phy['Phylum'].isin(major)]

    # Compute weighted EAF if requested
    if weighted:
        # For each treatment x timepoint, compute total ASVs across all phyla
        total_asvs = phy.groupby(['treatment_label', 'timepoint'])['n_ASVs'].sum().reset_index()
        total_asvs.columns = ['treatment_label', 'timepoint', 'total_ASVs']
        phy = phy.merge(total_asvs, on=['treatment_label', 'timepoint'], how='left')
        phy['display_EAF'] = phy['mean_EAF'] * phy['n_ASVs'] / phy['total_ASVs']
    else:
        phy['display_EAF'] = phy['mean_EAF']

    phylum_mean = phy.groupby('Phylum')['display_EAF'].mean().sort_values(ascending=True)
    phyla_ordered = phylum_mean.index.tolist()

    treats = ['Glucose', 'Xylose', 'Hemicellulose']
    ALL_TPS = [2, 8, 24, 72, 168, 336, 672]
    treat_tps = {t: ALL_TPS for t in treats}

    all_cols = []
    for t in treats:
        for tp in treat_tps[t]:
            all_cols.append((tp, t))

    n_cols, n_rows = len(all_cols), len(phyla_ordered)
    mat = np.full((n_rows, n_cols), np.nan)
    for ci, (tp, treat) in enumerate(all_cols):
        for ri, pname in enumerate(phyla_ordered):
            vals = phy[(phy['Phylum'] == pname) & (phy['treatment_label'] == treat) &
                       (phy['timepoint'] == tp)]['display_EAF']
            if len(vals) > 0:
                mat[ri, ci] = vals.iloc[0]

    # Fill missing phylum-treatment-timepoint combos with 0 (no observed incorporation)
    # so they display as white (center of RdBu_r) instead of grey (NaN).
    # This matches Figure 1's visual behavior where Greys colormap renders NaN as white.
    mat = np.where(np.isnan(mat), 0.0, mat)

    # Symmetric diverging colormap: vmin = -vmax so colors are balanced around 0
    actual_vmax = vmax
    if weighted:
        max_val = np.nanmax(np.abs(mat))
        actual_vmax = max(max_val * 1.1, 0.01)

    norm = mcolors.TwoSlopeNorm(vmin=-actual_vmax, vcenter=0, vmax=actual_vmax)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

    # Black gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    ylabels = [p.replace('Candidatus_', 'Ca. ').replace('p__', '') for p in phyla_ordered]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ylabels, fontsize=8, fontfamily=FF)

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

    ax.set_title(title, fontsize=11, fontweight='bold', fontfamily=FF, pad=20)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', fontfamily=FF)
    ax.tick_params(direction='out', length=2.5, width=0.6)
    return im


# Load data
print("Loading qSIP phylum data...")
phylum_df = pd.read_csv(
    os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_by_phylum.tsv'), sep='\t')

n_16s = len(phylum_df[phylum_df['marker'] == '16S'].groupby('Phylum')['n_ASVs'].sum().pipe(
    lambda s: s[s >= 100]))
n_its = len(phylum_df[phylum_df['marker'] == 'ITS'].groupby('Phylum')['n_ASVs'].sum().pipe(
    lambda s: s[s >= 20]))

print(f"  16S: {n_16s} phyla, ITS: {n_its} phyla")

# 2x2 layout: (16S unweighted, 16S weighted) on top, (ITS unweighted, ITS weighted) below
row_h_mm = 12
fig_h = 80 + (n_16s + n_its) * row_h_mm
fig_h = min(fig_h, 350)

fig, axes = plt.subplots(2, 2, figsize=(180 * MM, fig_h * MM),
                          gridspec_kw={'height_ratios': [n_16s, n_its],
                                       'hspace': 0.25, 'wspace': 0.40})

im_16s_u = plot_phylum_heatmap_color(axes[0, 0], '16S', 'A',
                                      '16S: Unweighted mean EAF',
                                      phylum_df, min_asv_total=100, vmax=0.17, weighted=False)
im_16s_w = plot_phylum_heatmap_color(axes[0, 1], '16S', 'B',
                                      '16S: Abundance-weighted EAF',
                                      phylum_df, min_asv_total=100, vmax=0.17, weighted=True)
im_its_u = plot_phylum_heatmap_color(axes[1, 0], 'ITS', 'C',
                                      'ITS: Unweighted mean EAF',
                                      phylum_df, min_asv_total=20, vmax=0.17, weighted=False)
im_its_w = plot_phylum_heatmap_color(axes[1, 1], 'ITS', 'D',
                                      'ITS: Abundance-weighted EAF',
                                      phylum_df, min_asv_total=20, vmax=0.17, weighted=True)

# Shared colorbar (for unweighted)
cax = fig.add_axes([0.15, 0.01, 0.30, 0.012])
cb = plt.colorbar(im_16s_u, cax=cax, orientation='horizontal')
cb.set_label('Mean EAF (unweighted)', fontsize=8, fontfamily=FF)
cb.ax.tick_params(labelsize=8)

cax2 = fig.add_axes([0.55, 0.01, 0.30, 0.012])
cb2 = plt.colorbar(im_16s_w, cax=cax2, orientation='horizontal')
cb2.set_label('Abundance-weighted EAF', fontsize=8, fontfamily=FF)
cb2.ax.tick_params(labelsize=8)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS3_qSIP_phylum_heatmaps_color.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS3_qSIP_phylum_heatmaps_color.pdf/.png")
