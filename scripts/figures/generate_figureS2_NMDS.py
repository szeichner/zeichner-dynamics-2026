#!/usr/bin/env python3
"""
Generate Figure S2 -- NMDS ordination of control-standardized VST KO profiles.

Same data as Figure 1 Panel B (PCA), but using non-metric MDS to provide a
method-independent confirmation of treatment overlap and temporal structure.

Uses sklearn.manifold.MDS with metric=False (NMDS) and Bray-Curtis dissimilarity.

Style: CLAUDE.md (PNAS format, double-column 180 mm wide)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from pydeseq2.dds import DeseqDataSet

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAT_DIR = os.path.join(BASE_DIR, 'data', 'metatranscriptomics', 'img data')
IMG_METADATA_TSV = os.path.join(METAT_DIR, 'IMG_metadata.tsv')
KO_OUTPUT_JSON = os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')
QC_OUTPUT_STATS_CSV = os.path.join(BASE_DIR, 'data', 'metatranscriptomics',
                                    'qc', 'output_stat_summary.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'output')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================
TREATMENT_COLORS = {
    'GLUCOSE':   '#6B7C2E',
    'XYLOSE':    '#C8960C',
    'CELLULOSE': '#7B3F6E',
}
TREATMENT_LABELS = {
    'GLUCOSE':   'Glucose',
    'XYLOSE':    'Xylose',
    'CELLULOSE': 'Hemicellulose',
}
TREATMENT_MARKERS = {'GLUCOSE': 'o', 'XYLOSE': '^', 'CELLULOSE': 's'}
MAIN_TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']

FONT_FAMILY = 'Arial'
FONT_AXIS_LABEL = 10
FONT_TICK_LABEL = 8
FONT_PANEL_LABEL = 12
FONT_LEGEND = 8
FONT_ANNOTATION = 8
MARKER_SIZE = 40
DPI_PNG = 600
MM_TO_INCH = 1 / 25.4


def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=2.5, width=0.6, labelsize=FONT_TICK_LABEL)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


# ============================================================================
# QC + DATA LOAD (same as Figure 1)
# ============================================================================
print("Running QC and loading data...")
qc = pd.read_csv(QC_OUTPUT_STATS_CSV)
qc['qc_pass_rate'] = (qc['reads_passed_qc'] / qc['raw_read_count']) * 100
qc['total_flags'] = (
    (qc['qc_pass_rate'] < 40).astype(int) +
    (qc['pct_reads_assembled'] < 70).astype(int) +
    (qc['contig_n50'] < 200).astype(int) +
    (qc['pct_fully_stranded'] < 85).astype(int) +
    (qc['raw_read_count'] < 50e6).astype(int)
)
qc_passed = set(qc[qc['total_flags'] == 0]['sample_name'].values)

img_metadata = pd.read_csv(IMG_METADATA_TSV, sep='\t')
img_metadata = img_metadata[img_metadata['sample_name'].isin(qc_passed)].copy()

with open(KO_OUTPUT_JSON, 'r') as f:
    ko_data = json.load(f)

taxon_ids = img_metadata['taxon_id'].astype(str).tolist()
ko_matrix = []
for tid in taxon_ids:
    ko_matrix.append([ko_data[ko].get(tid, 0) for ko in ko_data.keys()])
ko_df = pd.DataFrame(ko_matrix, columns=list(ko_data.keys()), index=taxon_ids)

ko_presence = (ko_df > 1).sum(axis=0)
ko_filt = ko_df.loc[:, ko_presence >= 5]
ko_filt = ko_filt.loc[:, ko_filt.var(axis=0) > 0]

counts = ko_filt.astype(int)
meta = img_metadata.set_index(img_metadata['taxon_id'].astype(str))
counts.index = counts.index.astype(str)
common = counts.index.intersection(meta.index)
counts = counts.loc[common]
meta = meta.loc[common]

# ============================================================================
# VST + CONTROL STANDARDIZATION
# ============================================================================
print("Running DESeq2 VST...")
dds = DeseqDataSet(counts=counts, metadata=meta,
                   design_factors=['treatment', 'time_hours'], n_cpus=4)
dds.deseq2()
dds.vst_fit()
vst = pd.DataFrame(dds.vst_transform(),
                    index=counts.index, columns=counts.columns)

ctrl = meta['treatment'] == 'control'
vst_cs = (vst - vst.loc[ctrl].mean(axis=0)) / vst.loc[ctrl].std(axis=0).replace(0, 1)

# Filter to treated samples only
treated = meta['treatment'].isin(MAIN_TREATMENTS)
vst_treated = vst_cs.loc[treated]
meta_treated = meta.loc[treated]

print(f"  {len(vst_treated)} treated samples for NMDS")

# ============================================================================
# NMDS with Bray-Curtis dissimilarity
# ============================================================================
print("Computing Bray-Curtis dissimilarity...")
# Shift data to non-negative for Bray-Curtis (add minimum value)
vst_shifted = vst_treated - vst_treated.min().min()
bc_dist = pdist(vst_shifted.values, metric='braycurtis')
bc_matrix = squareform(bc_dist)

print("Running NMDS...")
nmds = MDS(n_components=2, metric=False, dissimilarity='precomputed',
           random_state=42, n_init=10, max_iter=1000, normalized_stress='auto')
coords = nmds.fit_transform(bc_matrix)
stress = nmds.stress_
print(f"  NMDS stress = {stress:.4f}")

# ============================================================================
# PLOT
# ============================================================================
print("Generating Figure S2...")
fig_w = 120 * MM_TO_INCH  # single-column-ish
fig_h = 140 * MM_TO_INCH  # taller to fit gradient legend below
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0.12, right=0.92, top=0.93, bottom=0.28)

# Time gradient normalization
time_vals = meta_treated['time_hours'].astype(float).values
log_time = np.log10(time_vals + 1)
time_norm = Normalize(vmin=log_time.min(), vmax=log_time.max())

for treatment in MAIN_TREATMENTS:
    mask = meta_treated['treatment'].values == treatment
    t_coords = coords[mask]
    t_times = log_time[mask]
    color = TREATMENT_COLORS[treatment]
    marker = TREATMENT_MARKERS[treatment]

    # Use alpha gradient for time
    for j in range(t_coords.shape[0]):
        alpha = 0.3 + 0.7 * time_norm(t_times[j])
        ax.scatter(t_coords[j, 0], t_coords[j, 1],
                   c=color, marker=marker, s=MARKER_SIZE,
                   alpha=alpha, edgecolors='black', linewidths=0.3,
                   zorder=3)

# Stress annotation
ax.text(0.03, 0.97, f'Stress = {stress:.2f}',
        transform=ax.transAxes, fontsize=FONT_ANNOTATION,
        fontfamily=FONT_FAMILY, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.8))

ax.set_xlabel('NMDS1', fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax.set_ylabel('NMDS2', fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
apply_style(ax)

# Legend
legend_handles = []
for treatment in MAIN_TREATMENTS:
    legend_handles.append(
        Line2D([0], [0], marker=TREATMENT_MARKERS[treatment],
               color='none',
               markerfacecolor=TREATMENT_COLORS[treatment],
               markeredgecolor='black', markeredgewidth=0.3,
               markersize=7, linestyle='none',
               label=TREATMENT_LABELS[treatment]))
ax.legend(handles=legend_handles, fontsize=FONT_LEGEND,
          loc='upper right', frameon=True, fancybox=False,
          edgecolor='#CCCCCC', framealpha=0.9,
          handletextpad=0.4)

# Time-gradient legend: three horizontal gradient bars below the plot
from matplotlib.colors import to_rgba

n_steps = 100
bar_height = 0.025  # fraction of figure height
y_base = 0.04       # bottom of lowest bar
y_gap = 0.005       # gap between bars
x_left = 0.25       # left edge of gradient bars (figure coords)
x_width = 0.55      # width of gradient bars
label_x = x_left - 0.02  # right edge of treatment labels

for ti, treatment in enumerate(MAIN_TREATMENTS):
    base_color = to_rgba(TREATMENT_COLORS[treatment])
    gradient = np.zeros((1, n_steps, 4))
    for si in range(n_steps):
        frac = si / (n_steps - 1)
        alpha = 0.3 + 0.7 * frac
        gradient[0, si, :3] = base_color[:3]
        gradient[0, si, 3] = alpha

    y_bot = y_base + ti * (bar_height + y_gap)
    bar_ax = fig.add_axes([x_left, y_bot, x_width, bar_height])
    bar_ax.imshow(gradient, aspect='auto', extent=[0, 1, 0, 1],
                  interpolation='bicubic')
    bar_ax.set_yticks([])
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_visible(False)

    # Time tick labels only on the bottom bar
    if ti == 0:
        time_ticks = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]
        time_labels_list = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']
        bar_ax.set_xticks(time_ticks)
        bar_ax.set_xticklabels(time_labels_list, fontsize=8, fontfamily=FONT_FAMILY)
        bar_ax.tick_params(axis='x', length=2, width=0.5, pad=1)
    else:
        bar_ax.set_xticks([])
        bar_ax.spines['bottom'].set_visible(False)

    # Treatment label to the left
    fig.text(label_x, y_bot + bar_height / 2, TREATMENT_LABELS[treatment],
             fontsize=8, fontfamily=FONT_FAMILY, ha='right', va='center')

# "Time" label centered above the bars
fig.text(x_left + x_width / 2, y_base + 3 * (bar_height + y_gap) + 0.01,
         'Time', fontsize=8, fontfamily=FONT_FAMILY, ha='center', va='bottom',
         fontweight='bold')

out_pdf = os.path.join(FIGURES_DIR, 'FigureS2_NMDS.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS2_NMDS.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
