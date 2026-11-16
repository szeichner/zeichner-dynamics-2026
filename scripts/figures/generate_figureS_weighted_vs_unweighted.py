#!/usr/bin/env python3
"""
Supplementary Figure: Weighted vs Unweighted EAF
3x2 grid (rows: Glucose, Xylose, Hemicellulose; cols: 16S, ITS)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
qsip = pd.read_csv(
    os.path.join(BASE, 'data/qSIP/output/tables/qsip_community_incorporation_efficiency.tsv'),
    sep='\t',
)

OUT_DIR = os.path.join(BASE, 'figures/output')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TREATMENTS = ['Glucose', 'Xylose', 'Hemicellulose']
MARKERS = ['16S', 'ITS']
COL_TITLES = {'16S': 'Bacteria (16S)', 'ITS': 'Fungi (ITS)'}

COLORS = {
    'Glucose':       '#6B7C2E',
    'Xylose':        '#C8960C',
    'Hemicellulose': '#7B3F6E',
}
MARKERS_SHAPE = {
    'Glucose':       'o',
    'Xylose':        '^',
    'Hemicellulose': 's',
}

TIMEPOINTS = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']
RATIO_TPS = [72, 336, 672]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

mm = 1 / 25.4  # mm to inches
fig, axes = plt.subplots(3, 2, figsize=(180 * mm, 180 * mm),
                         sharex=True, sharey=False)

panel_labels = list('ABCDEF')

for row_i, treatment in enumerate(TREATMENTS):
    for col_j, marker in enumerate(MARKERS):
        ax = axes[row_i, col_j]
        color = COLORS[treatment]
        mk = MARKERS_SHAPE[treatment]

        sub = qsip[(qsip['treatment'] == treatment) & (qsip['marker'] == marker)].copy()
        sub = sub.sort_values('timepoint')

        tp = sub['timepoint'].values
        w_eaf = sub['weighted_mean_EAF'].values
        u_eaf = sub['unweighted_mean_EAF'].values

        # Lines
        ax.plot(tp, w_eaf, '-', color=color, marker=mk, markersize=4,
                linewidth=1.2, label='Weighted EAF', zorder=3)
        ax.plot(tp, u_eaf, '--', color=color, marker=mk, markersize=4,
                linewidth=1.2, label='Unweighted EAF', zorder=3)
        ax.fill_between(tp, w_eaf, u_eaf, color=color, alpha=0.10, zorder=1)

        # Ratio annotations at selected timepoints
        for rtp in RATIO_TPS:
            idx = np.where(tp == rtp)[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            if u_eaf[i] != 0:
                ratio = w_eaf[i] / u_eaf[i]
            else:
                ratio = np.nan
            y_top = max(w_eaf[i], u_eaf[i])
            ax.annotate(f'{ratio:.1f}x',
                        xy=(rtp, y_top),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=color)

        # X-axis
        ax.set_xticks(TIMEPOINTS)
        if row_i == 2:
            ax.set_xticklabels(TP_LABELS, fontsize=8, fontfamily='Arial')
            ax.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=10, fontfamily='Arial')
        else:
            ax.set_xticklabels([])

        ax.set_xscale('log')
        ax.set_xlim(1.5, 900)

        # Y-axis
        if col_j == 0:
            ax.set_ylabel('Mean EAF', fontsize=10, fontfamily='Arial')

        # Column titles
        if row_i == 0:
            ax.set_title(COL_TITLES[marker], fontsize=11, fontweight='bold', fontfamily='Arial')

        # Row labels (right side)
        if col_j == 1:
            ax.annotate(treatment, xy=(1.06, 0.5), xycoords='axes fraction',
                        fontsize=10, fontweight='bold', fontfamily='Arial',
                        rotation=270, ha='left', va='center')

        # Panel label
        label_idx = row_i * 2 + col_j
        ax.text(-0.15, 1.08, panel_labels[label_idx],
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                fontfamily='Arial', va='top', ha='left')

        # Spine offset
        ax.spines['left'].set_position(('outward', 4))
        ax.spines['bottom'].set_position(('outward', 4))

# Shared legend at bottom
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='k', linewidth=1.2, linestyle='-',  label='Weighted EAF'),
    Line2D([0], [0], color='k', linewidth=1.2, linestyle='--', label='Unweighted EAF'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02),
           prop={'family': 'Arial'})

plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save
fig.savefig(os.path.join(OUT_DIR, 'FigureS_weighted_vs_unweighted.pdf'),
            dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'FigureS_weighted_vs_unweighted.png'),
            dpi=600, bbox_inches='tight')
print('Saved FigureS_weighted_vs_unweighted.pdf and .png')
plt.close()
