#!/usr/bin/env python3
"""
Generate supplementary figure — k-selection and program trajectory shapes.

Panel A: Elbow plot (inertia vs k) + silhouette score
Panel B: Mean LFC trajectories for k=5 programs (monomer and hemicellulose)

Also shows k=3 and k=4 for comparison.
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

TP_HOURS_MONO = [2, 8, 24, 72, 168, 336]
TP_HOURS_HEMI = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

PROG_COLORS_K5 = {
    0: '#D95F02',   # Early monomer burst — warm orange
    1: '#1B9E77',   # Mid monomer plateau — teal
    2: '#7570B3',   # Late succession — muted purple
    3: '#E7298A',   # Polymer-responsive — magenta
    4: '#66A61E',   # Baseline suppressed — green
}

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')
ksel = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_k_selection.tsv'), sep='\t')

lfc_cols = [c for c in assign.columns if c.startswith('MONOMER_') or c.startswith('HEMICELLULOSE_')]
mono_cols = [c for c in lfc_cols if c.startswith('MONOMER')]
hemi_cols = [c for c in lfc_cols if c.startswith('HEMICELLULOSE')]

lfc = assign[lfc_cols].copy()
lfc_centered = lfc.subtract(lfc.mean(axis=1), axis=0)
X = lfc_centered.values

# ============================================================================
# Figure: 3 rows
# Row 1: k-selection (elbow + silhouette)
# Row 2: k=5 program trajectories (monomer left, hemi right)
# Row 3: k=3 and k=4 for comparison
# ============================================================================
print("Generating figure...")

fig = plt.figure(figsize=(180 * MM, 240 * MM))
gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35,
                      left=0.12, right=0.95, top=0.96, bottom=0.07,
                      height_ratios=[0.7, 1, 1])

# --- Row 1: k-selection ---
ax_elbow = fig.add_subplot(gs[0, 0])
ax_elbow.plot(ksel.k, ksel.inertia, 'ko-', markersize=5, linewidth=1.5)
ax_elbow.axvline(5, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax_elbow.set_xlabel('k (number of clusters)', fontsize=10, fontfamily=FF)
ax_elbow.set_ylabel('Within-cluster\nsum of squares', fontsize=10, fontfamily=FF)
ax_elbow.set_title('Elbow plot', fontsize=10, fontfamily=FF, fontweight='bold')
ax_elbow.spines['top'].set_visible(False)
ax_elbow.spines['right'].set_visible(False)
for _sp in ['left', 'bottom']:
    ax_elbow.spines[_sp].set_linewidth(0.6)
ax_elbow.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)
ax_elbow.text(-0.15, 1.08, 'A', transform=ax_elbow.transAxes, fontsize=12,
              fontweight='bold', va='top', fontfamily=FF)

ax_sil = fig.add_subplot(gs[0, 1])
ax_sil.plot(ksel.k, ksel.silhouette, 's-', color='#7B3F6E', markersize=5, linewidth=1.5)
ax_sil.axvline(5, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax_sil.set_xlabel('k (number of clusters)', fontsize=10, fontfamily=FF)
ax_sil.set_ylabel('Silhouette score', fontsize=10, fontfamily=FF)
ax_sil.set_title('Silhouette score', fontsize=10, fontfamily=FF, fontweight='bold')
ax_sil.spines['top'].set_visible(False)
ax_sil.spines['right'].set_visible(False)
for _sp in ['left', 'bottom']:
    ax_sil.spines[_sp].set_linewidth(0.6)
ax_sil.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)

# --- Row 2: k=5 trajectories (computed on the fly) ---
km5 = KMeans(n_clusters=5, random_state=42, n_init=20)
labels_k5 = km5.fit_predict(X)

# Sort clusters by monomer peak time for consistent ordering
cluster_peaks_k5 = []
for c in range(5):
    mask = labels_k5 == c
    traj = lfc.loc[mask, mono_cols].mean(axis=0)
    peak_idx = traj.values.argmax()
    cluster_peaks_k5.append((c, peak_idx, mask.sum()))
cluster_peaks_k5.sort(key=lambda x: x[1])

palette_k5 = [PROG_COLORS_K5[i] for i in range(5)]

# Assign descriptive labels based on trajectory shape after peak-time sorting
# (cluster_peaks_k5 is sorted by monomer peak index)
DESCRIPTIVE_LABELS_K5 = [
    'Early growth burst',
    'Mid-phase plateau',
    'Late succession rise',
    'Polymer-responsive',
    'Baseline suppressed',
]

# k=5 Monomers
ax_k5m = fig.add_subplot(gs[1, 0])
for ci, (c, _, n) in enumerate(cluster_peaks_k5):
    mask = labels_k5 == c
    traj = lfc.loc[mask, mono_cols].mean(axis=0)
    xs = TP_HOURS_MONO
    ax_k5m.plot(xs, traj.values, color=palette_k5[ci], linewidth=1.8,
                label=f'{DESCRIPTIVE_LABELS_K5[ci]} (n={n})')
    # SEM shading
    sem = lfc.loc[mask, mono_cols].std(axis=0) / np.sqrt(n)
    ax_k5m.fill_between(xs, traj.values - sem.values, traj.values + sem.values,
                        color=palette_k5[ci], alpha=0.1)

ax_k5m.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='--')
ax_k5m.set_xscale('log')
ax_k5m.set_xticks(TP_HOURS_MONO)
ax_k5m.set_xlim(1.5, 900)
ax_k5m.set_xticklabels([TP_LABELS[t] for t in TP_HOURS_MONO], fontsize=8, fontfamily=FF)
ax_k5m.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_k5m.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=10, fontfamily=FF)
ax_k5m.set_ylabel('Mean $\\log_2$FC vs control', fontsize=10, fontfamily=FF)
ax_k5m.set_title('k = 5: Monomer treatments', fontsize=10, fontfamily=FF, fontweight='bold')
ax_k5m.spines['top'].set_visible(False)
ax_k5m.spines['right'].set_visible(False)
for _sp in ['left', 'bottom']:
    ax_k5m.spines[_sp].set_linewidth(0.6)
ax_k5m.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)
ax_k5m.legend(fontsize=6.5, loc='upper left', frameon=True, fancybox=False,
              edgecolor='#CCCCCC', facecolor='white', framealpha=0.9,
              handletextpad=0.3, handlelength=1.5, borderpad=0.3, labelspacing=0.3)
ax_k5m.text(-0.15, 1.08, 'B', transform=ax_k5m.transAxes, fontsize=10,
            fontweight='bold', va='top', fontfamily=FF)

# k=5 Hemicellulose
ax_k5h = fig.add_subplot(gs[1, 1])
for ci, (c, _, n) in enumerate(cluster_peaks_k5):
    mask = labels_k5 == c
    traj = lfc.loc[mask, hemi_cols].mean(axis=0)
    xs = TP_HOURS_HEMI
    ax_k5h.plot(xs, traj.values, color=palette_k5[ci], linewidth=1.8,
                label=f'{DESCRIPTIVE_LABELS_K5[ci]}')
    sem = lfc.loc[mask, hemi_cols].std(axis=0) / np.sqrt(n)
    ax_k5h.fill_between(xs, traj.values - sem.values, traj.values + sem.values,
                        color=palette_k5[ci], alpha=0.1)

ax_k5h.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='--')
ax_k5h.set_xscale('log')
ax_k5h.set_xticks(TP_HOURS_HEMI)
ax_k5h.set_xlim(1.5, 900)
ax_k5h.set_xticklabels([TP_LABELS[t] for t in TP_HOURS_HEMI], fontsize=8, fontfamily=FF)
ax_k5h.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_k5h.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=10, fontfamily=FF)
ax_k5h.set_ylabel('Mean $\\log_2$FC vs control', fontsize=10, fontfamily=FF)
ax_k5h.set_title('k = 5: Hemicellulose treatment', fontsize=10, fontfamily=FF, fontweight='bold')
ax_k5h.spines['top'].set_visible(False)
ax_k5h.spines['right'].set_visible(False)
for _sp in ['left', 'bottom']:
    ax_k5h.spines[_sp].set_linewidth(0.6)
ax_k5h.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)

# --- Row 3: k=3 and k=4 for comparison ---
palette_k3 = ['#6B7C2E', '#A2C060', '#7B3F6E']  # match k=3 program colors
palette_k4 = ['#D95F02', '#1B9E77', '#7570B3', '#E7298A']

DESCRIPTIVE_LABELS_K3 = [
    'P1: Monomer growth response',
    'P2: Late succession',
    'P3: Hemicellulose-associated',
]
DESCRIPTIVE_LABELS_K4 = [
    'Early monomer burst',
    'Mid monomer response',
    'Late succession rise',
    'Polymer-associated / suppressed',
]

for ki, (k_val, palette, desc_labels) in enumerate([
    (3, palette_k3, DESCRIPTIVE_LABELS_K3),
    (4, palette_k4, DESCRIPTIVE_LABELS_K4),
]):
    ax_comp = fig.add_subplot(gs[2, ki])
    km = KMeans(n_clusters=k_val, random_state=42, n_init=20)
    labels = km.fit_predict(X)

    # Sort clusters by monomer peak time for consistent ordering
    cluster_peaks = []
    for c in range(k_val):
        mask = labels == c
        traj = lfc.loc[mask, mono_cols].mean(axis=0)
        peak_idx = traj.values.argmax()
        cluster_peaks.append((c, peak_idx, mask.sum()))
    cluster_peaks.sort(key=lambda x: x[1])

    for ci, (c, _, n) in enumerate(cluster_peaks):
        mask = labels == c
        # Plot monomer trajectory (solid)
        traj_m = lfc.loc[mask, mono_cols].mean(axis=0)
        xs_m = TP_HOURS_MONO
        ax_comp.plot(xs_m, traj_m.values, color=palette[ci], linewidth=1.8,
                     label=f'{desc_labels[ci]} (n={n})')
        # Plot hemicellulose trajectory (dashed)
        traj_h = lfc.loc[mask, hemi_cols].mean(axis=0)
        xs_h = TP_HOURS_HEMI
        ax_comp.plot(xs_h, traj_h.values, color=palette[ci], linewidth=1.2,
                     linestyle='--', alpha=0.7)

    ax_comp.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='--')
    ax_comp.set_xscale('log')
    ax_comp.set_xticks(TP_HOURS_HEMI)
    ax_comp.set_xlim(1.5, 900)
    ax_comp.set_xticklabels([TP_LABELS[t] for t in TP_HOURS_HEMI], fontsize=8, fontfamily=FF)
    ax_comp.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_comp.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=10, fontfamily=FF)
    ax_comp.set_ylabel('Mean $\\log_2$FC vs control', fontsize=10, fontfamily=FF)
    ax_comp.set_title(f'k = {k_val} (comparison)', fontsize=10, fontfamily=FF, fontweight='bold')
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)
    for _sp in ['left', 'bottom']:
        ax_comp.spines[_sp].set_linewidth(0.6)
    ax_comp.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)
    ax_comp.legend(fontsize=6.5, loc='upper left', frameon=True, fancybox=False,
                   edgecolor='#CCCCCC', facecolor='white', framealpha=0.9,
                   handletextpad=0.3, handlelength=1.5, borderpad=0.3, labelspacing=0.3)
    letter = 'C' if ki == 0 else 'D'
    ax_comp.text(-0.15, 1.08, letter, transform=ax_comp.transAxes, fontsize=12,
                 fontweight='bold', va='top', fontfamily=FF)

# Add note about solid vs dashed — placed below bottom margin
fig.text(0.5, 0.015, 'Solid lines = monomer treatments; dashed lines = hemicellulose treatment',
         fontsize=7, fontfamily=FF, ha='center', color='#666666')

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS_k_selection_trajectories.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS_k_selection_trajectories.pdf/.png")
