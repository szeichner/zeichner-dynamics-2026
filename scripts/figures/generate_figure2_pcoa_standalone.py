#!/usr/bin/env python3
"""
Figure 2 — Bray-Curtis PCoA ordination (2 panels).

Panel A: BC-PCoA scatter (all 74 samples, time-gradient coloring, centroid arrows)
Panel B: PCo1 score vs time (timepoint means + SEM)

Output: Figure2_PCoA.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from scipy.interpolate import splprep, splev
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

# ---------- Shared palette ----------
TC = {'GLUCOSE': '#6B7C2E', 'XYLOSE': '#C8960C', 'CELLULOSE': '#7B3F6E'}
TL = {'GLUCOSE': 'Glucose', 'XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}
TM = {'GLUCOSE': 'o', 'XYLOSE': '^', 'CELLULOSE': 's'}

# ---------- Shared sizing ----------
AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
MARKER_SIZE = 5
LINE_WIDTH = 1.5
ANNO_SIZE = 8

ALL_TPS = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']

def make_gradient(hex_color, n=7):
    r, g, b = mcolors.hex2color(hex_color)
    colors = []
    for i in range(n):
        frac = i / (n - 1)
        w = 1.0 - 0.1 - 0.9 * frac
        colors.append((r + w * (1 - r), g + w * (1 - g), b + w * (1 - b)))
    return colors

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(direction='out', length=3, width=0.8, labelsize=TICK_LABEL)

# ============================================================================
# Load data
# ============================================================================
print("Loading PCoA data...")
coords = pd.read_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'), sep='\t')
eig = pd.read_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_eigenvalues.tsv'), sep='\t')
pco1_var = eig.loc[0, 'variance_explained_pct']
pco2_var = eig.loc[1, 'variance_explained_pct']

# ============================================================================
# Figure layout: 1 row × 2 cols
# ============================================================================
print("Generating Figure 2 PCoA panels...")
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(180 * MM, 90 * MM))
fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.20, wspace=0.45)

# ============================================================================
# Panel A: BC-PCoA scatter
# ============================================================================
main_mask = coords.treatment.isin(['GLUCOSE', 'XYLOSE', 'CELLULOSE'])
ctrl_mask = coords.treatment.isin(['control', 'basal'])

# Control points
ctrl = coords[ctrl_mask]
ax_a.scatter(ctrl.PCo1, ctrl.PCo2, c='#9B9B9B', marker='D', s=20,
             edgecolors='black', linewidths=0.3, alpha=0.6, zorder=2)

tp_label_map = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

for treat in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
    sub = coords[coords.treatment == treat].copy()
    timepoints = sorted(sub.time_hours.unique())
    gradient = make_gradient(TC[treat], len(timepoints))
    tp_to_idx = {tp: i for i, tp in enumerate(timepoints)}

    for _, row in sub.iterrows():
        idx = tp_to_idx[row.time_hours]
        ax_a.scatter(row.PCo1, row.PCo2, c=[gradient[idx]],
                     marker=TM[treat], s=35, edgecolors='black',
                     linewidths=0.3, zorder=3)

    # Smooth curved arrow through centroids
    centroids_x, centroids_y = [], []
    for tp in timepoints:
        tp_sub = sub[sub.time_hours == tp]
        centroids_x.append(tp_sub.PCo1.mean())
        centroids_y.append(tp_sub.PCo2.mean())

    cx, cy = np.array(centroids_x), np.array(centroids_y)

    if len(cx) >= 4:
        tck, u = splprep([cx, cy], s=0, k=3)
        u_fine = np.linspace(0, 1, 300)
        sx, sy = splev(u_fine, tck)
    else:
        u_fine = np.linspace(0, 1, 300)
        sx = np.interp(u_fine, np.linspace(0, 1, len(cx)), cx)
        sy = np.interp(u_fine, np.linspace(0, 1, len(cy)), cy)

    # Draw the smooth curve (without last few points — arrowhead covers the tip)
    ax_a.plot(sx[:-8], sy[:-8], color=TC[treat], linewidth=1.8,
              alpha=0.8, solid_capstyle='round', zorder=5)

    # Arrowhead at the end
    ax_a.annotate('', xy=(sx[-1], sy[-1]), xytext=(sx[-12], sy[-12]),
                  arrowprops=dict(arrowstyle='->', color=TC[treat],
                                  linewidth=1.8, shrinkA=0, shrinkB=1,
                                  mutation_scale=12), zorder=5)

    # Time labels at centroids
    for k, tp in enumerate(timepoints):
        label = tp_label_map.get(tp, str(tp))
        offset_y = 0.006 if k % 2 == 0 else -0.008
        ax_a.text(centroids_x[k], centroids_y[k] + offset_y, label,
                  fontsize=7, fontfamily=FF, ha='center', va='center',
                  color=TC[treat], fontweight='bold', zorder=6,
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='none', alpha=0.7))

ax_a.set_xlabel(f'PCo1 ({pco1_var:.1f}% variance)', fontsize=AXIS_LABEL, fontfamily=FF)
ax_a.set_ylabel(f'PCo2 ({pco2_var:.1f}% variance)', fontsize=AXIS_LABEL, fontfamily=FF)
style_ax(ax_a)
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes, fontsize=PANEL_LABEL,
          fontweight='bold', va='top', fontfamily=FF)

# Time gradient bars below Panel A
for ti, treat in enumerate(['GLUCOSE', 'XYLOSE', 'CELLULOSE']):
    timepoints = sorted(coords[coords.treatment == treat].time_hours.unique())
    gradient = make_gradient(TC[treat], len(timepoints))
    bar_y = -0.25 - ti * 0.08
    for k, c in enumerate(gradient):
        ax_a.add_patch(plt.Rectangle(
            (k / len(gradient), bar_y), 1 / len(gradient), 0.05,
            transform=ax_a.transAxes, facecolor=c, edgecolor='none', clip_on=False))
    ax_a.text(-0.02, bar_y + 0.025, TL[treat], fontsize=7, fontfamily=FF,
              ha='right', va='center', color=TC[treat],
              transform=ax_a.transAxes, fontweight='bold')
ax_a.text(0.0, -0.25 - 3 * 0.08 + 0.01, 'Early', fontsize=7.5, fontfamily=FF,
          ha='left', va='top', color='#888888', transform=ax_a.transAxes)
ax_a.text(1.0, -0.25 - 3 * 0.08 + 0.01, 'Late', fontsize=7.5, fontfamily=FF,
          ha='right', va='top', color='#888888', transform=ax_a.transAxes)

# ============================================================================
# Panel B: PCo1 score vs time
# ============================================================================
tp_xpos = {tp: i for i, tp in enumerate(ALL_TPS)}

# Control baseline
ctrl_pco1 = coords[coords.treatment == 'control'].PCo1.values
ctrl_mean = np.mean(ctrl_pco1)
ax_b.axhline(ctrl_mean, color='#CCCCCC', linewidth=0.8, linestyle=':', zorder=0)
ax_b.text(len(ALL_TPS) - 0.5, ctrl_mean + 0.004, 'Control', fontsize=ANNO_SIZE,
          fontfamily=FF, color='#AAAAAA', va='bottom', ha='right')

for treat in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
    sub = coords[(coords.treatment == treat) & (coords.time_hours > 0)].copy()
    tp_means = sub.groupby('time_hours').PCo1.agg(['mean', 'sem']).reset_index()
    xs = [tp_xpos[tp] for tp in tp_means.time_hours]
    means = tp_means['mean'].values
    sems = tp_means['sem'].values

    ax_b.plot(xs, means, color=TC[treat], linewidth=LINE_WIDTH, zorder=3)
    ax_b.fill_between(xs, means - sems, means + sems,
                      color=TC[treat], alpha=0.15, zorder=2)
    ax_b.errorbar(xs, means, yerr=sems, fmt=TM[treat], color=TC[treat],
                  markersize=MARKER_SIZE, markeredgecolor='black', markeredgewidth=0.3,
                  elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=4)

ax_b.axhline(0, color='#EEEEEE', linewidth=0.5, zorder=0)
ax_b.set_xticks(range(len(ALL_TPS)))
ax_b.set_xticklabels(TP_LABELS, fontsize=TICK_LABEL, fontfamily=FF)
ax_b.set_xlabel('Time (h)', fontsize=AXIS_LABEL, fontfamily=FF)
ax_b.set_ylabel(f'PCo1 score ({pco1_var:.1f}% variance)', fontsize=AXIS_LABEL, fontfamily=FF)
style_ax(ax_b)
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes, fontsize=PANEL_LABEL,
          fontweight='bold', va='top', fontfamily=FF)

# ============================================================================
# Shared legend at bottom
# ============================================================================
legend_handles = [
    Line2D([0], [0], marker='o', color=TC['GLUCOSE'], markerfacecolor=TC['GLUCOSE'],
           markeredgecolor='black', markeredgewidth=0.3, markersize=6,
           linewidth=LINE_WIDTH, label='Glucose'),
    Line2D([0], [0], marker='^', color=TC['XYLOSE'], markerfacecolor=TC['XYLOSE'],
           markeredgecolor='black', markeredgewidth=0.3, markersize=6,
           linewidth=LINE_WIDTH, label='Xylose'),
    Line2D([0], [0], marker='s', color=TC['CELLULOSE'], markerfacecolor=TC['CELLULOSE'],
           markeredgecolor='black', markeredgewidth=0.3, markersize=6,
           linewidth=LINE_WIDTH, label='Hemicellulose'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor='#9B9B9B',
           markeredgecolor='black', markeredgewidth=0.3, markersize=5,
           label='Control'),
]
fig.legend(handles=legend_handles, fontsize=LEGEND_SIZE, loc='lower center',
           ncol=4, frameon=False, bbox_to_anchor=(0.53, 0.0),
           handletextpad=0.3, columnspacing=1.2)

# ============================================================================
# Save
# ============================================================================
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'Figure2_PCoA.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved Figure2_PCoA.pdf/.png")
