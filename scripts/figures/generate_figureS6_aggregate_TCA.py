#!/usr/bin/env python3
"""
Generate Figure S17 — Aggregate mean fractional ¹³C labeling across TCA
intermediates and amino acids.

Single-column figure (88 mm wide) showing mean fractional ¹³C labeling
Σ(i × fᵢ) / nC averaged across all TCA intermediates (Cit, aKG, Suc, Fum,
Mal) plus amino acids (Asp, GLUM, AGLUM), plotted over time.

3 treatment lines with SEM shading. Categorical x-axis.

Style: CLAUDE.md (PNAS format, single-column 88 mm wide)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METABOLOMICS_DIR = os.path.join(BASE_DIR, 'data', 'metabolomics')
GLUCOSE_CORR_AREA_CSV = (
    '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity '
    '/data/Metabolomics/Data/GlucoseResults_ForIsoCor_res_corrected_area.csv')
XYLOSE_ISO_FRAC_CSV = os.path.join(METABOLOMICS_DIR,
    'XyloseResults_ForIsoCor_res_isotopologue_fraction.csv')
CELLULOSE_ISO_FRAC_CSV = os.path.join(METABOLOMICS_DIR,
    'CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv')
GLUCOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'GlucoseMetadata.csv')
XYLOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'XyloseMetadata.csv')
CELLULOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'CelluloseMetadata.csv')
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
LINEWIDTH_DATA = 1.5
LINEWIDTH_ERROR = 1.0
MARKER_SIZE = 5
CI_ALPHA = 0.20
DPI_PNG = 600
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 88 * MM_TO_INCH
FIG_HEIGHT = 70 * MM_TO_INCH

# Metabolites to include in aggregate
ALL_METS = ['Cit', 'aKG', 'Suc', 'Fum', 'Mal', 'Asp', 'GLUM', 'AGLUM']

# Carbon atoms per metabolite (for fractional labeling)
N_CARBONS = {
    'Cit': 6, 'aKG': 5, 'Suc': 4, 'Fum': 4, 'Mal': 4,
    'Asp': 4, 'GLUM': 5, 'AGLUM': 7,
    'PYR': 3, 'LAT': 3,
}

# Categorical timepoints (hours) and display labels
# Metadata timepoints are in days: 0.08, 0.33, 1, 2, 7, 14, 28
TP_HOURS = [1.92, 7.92, 24, 48, 168, 336, 672]
TP_LABELS = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']


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
# LOAD METABOLOMICS AND COMPUTE LABELED FRACTIONS
# ============================================================================
print("Loading metabolomics data...")


def compute_labeling(df, metadata, substrate, from_areas=False):
    """Compute fractional ¹³C labeling Σ(i × fᵢ) / nC per metabolite per sample."""
    results = []
    sample_cols = [c for c in df.columns if c not in ['metabolite', 'isotopologue']]
    for met in df['metabolite'].unique():
        if met not in ALL_METS:
            continue
        nc = N_CARBONS.get(met)
        if nc is None:
            continue
        met_data = df[df['metabolite'] == met]
        isotopologues = met_data['isotopologue'].values.astype(int)
        for sample in sample_cols:
            meta_row = metadata[metadata['sample'] == sample]
            if len(meta_row) == 0:
                continue
            tp_d = meta_row['timepoint'].values[0]
            tp_h = tp_d * 24
            rep = meta_row['replicate'].values[0]
            if from_areas:
                areas = met_data[sample].values.astype(float)
                total = np.nansum(areas)
                if total == 0 or np.isnan(total):
                    continue
                fractions = areas / total
            else:
                fractions = met_data[sample].values.astype(float)
                if np.any(np.isnan(fractions)):
                    continue
            frac_label = np.sum(isotopologues * fractions) / nc
            results.append({
                'substrate': substrate, 'metabolite': met,
                'sample': sample, 'timepoint_hours': tp_h,
                'replicate': rep, 'labeled_fraction': frac_label,
            })
    return pd.DataFrame(results)


# Load metadata
gluc_meta = pd.read_csv(GLUCOSE_META_CSV)
gluc_meta['sample'] = gluc_meta['sample'].astype(str)
xyl_meta = pd.read_csv(XYLOSE_META_CSV)
xyl_meta['sample'] = xyl_meta['sample'].astype(str)
cell_meta = pd.read_csv(CELLULOSE_META_CSV)
cell_meta['sample'] = cell_meta['sample'].astype(str)

# Load data and compute labeled fractions
all_lab = pd.concat([
    compute_labeling(pd.read_csv(GLUCOSE_CORR_AREA_CSV), gluc_meta, 'GLUCOSE',
                     from_areas=True),
    compute_labeling(pd.read_csv(XYLOSE_ISO_FRAC_CSV), xyl_meta, 'XYLOSE',
                     from_areas=False),
    compute_labeling(pd.read_csv(CELLULOSE_ISO_FRAC_CSV), cell_meta, 'CELLULOSE',
                     from_areas=False),
], ignore_index=True)

# Exclude baseline (timepoint == 0)
all_lab = all_lab[all_lab['timepoint_hours'] > 0].copy()
print(f"  {len(all_lab)} observations across {all_lab['metabolite'].nunique()} metabolites")

# ============================================================================
# COMPUTE AGGREGATE LABELED FRACTION
# For each sample x timepoint x substrate: average labeled fraction across
# all metabolites, then aggregate across replicates
# ============================================================================
print("Computing aggregate labeled fraction...")

# Round timepoints to nearest categorical value for grouping
tp_map = {}
for tp in all_lab['timepoint_hours'].unique():
    diffs = [abs(tp - tph) for tph in TP_HOURS]
    tp_map[tp] = TP_HOURS[np.argmin(diffs)]
all_lab['tp_cat'] = all_lab['timepoint_hours'].map(tp_map)

# Step 1: Average across metabolites within each sample (substrate x replicate x timepoint)
per_sample_agg = all_lab.groupby(
    ['substrate', 'replicate', 'tp_cat']
)['labeled_fraction'].mean().reset_index()
per_sample_agg.columns = ['substrate', 'replicate', 'tp_cat', 'agg_labeled_frac']

# Step 2: Average across replicates to get treatment means and SEM
final_agg = per_sample_agg.groupby(['substrate', 'tp_cat']).agg(
    mean=('agg_labeled_frac', 'mean'),
    std=('agg_labeled_frac', 'std'),
    n=('agg_labeled_frac', 'count')
).reset_index()
final_agg['sem'] = final_agg['std'] / np.sqrt(final_agg['n'])

print(f"  Final aggregate: {len(final_agg)} rows")
for treatment in MAIN_TREATMENTS:
    td = final_agg[final_agg['substrate'] == treatment]
    print(f"  {TREATMENT_LABELS[treatment]}: {len(td)} timepoints, "
          f"range [{td['mean'].min():.3f}, {td['mean'].max():.3f}]")

# ============================================================================
# GENERATE FIGURE S6
# ============================================================================
print("\nGenerating Figure S6...")

fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT),
                        constrained_layout=True)

# Map categorical timepoints to x positions
# Use actual hours for log x-axis

for treatment in MAIN_TREATMENTS:
    td = final_agg[final_agg['substrate'] == treatment].sort_values('tp_cat')
    if td.empty:
        continue

    color = TREATMENT_COLORS[treatment]
    marker = TREATMENT_MARKERS[treatment]
    label = TREATMENT_LABELS[treatment]

    x = td['tp_cat'].values
    y = td['mean'].values
    sem = td['sem'].values

    ax.plot(x, y, marker=marker, color=color, markersize=MARKER_SIZE,
            linewidth=LINEWIDTH_DATA, markeredgecolor='black',
            markeredgewidth=0.3, label=label, zorder=3)
    ax.fill_between(x, y - sem, y + sem, color=color, alpha=CI_ALPHA, zorder=2)

# Log x-axis
ax.set_xscale('log')
ax.set_xticks(TP_HOURS)
ax.set_xlim(1.5, 900)
ax.set_xticklabels(TP_LABELS, fontsize=FONT_TICK_LABEL, fontfamily=FONT_FAMILY)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

# Y-axis
ax.set_ylim(bottom=0)
ax.set_ylabel('Mean fractional $^{13}$C labeling, $\\Sigma(i \\times f_i)/n_C$',
              fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)

# Legend
ax.legend(fontsize=FONT_LEGEND, frameon=False, loc='upper left',
          handletextpad=0.3)

apply_style(ax)

# Save
out_pdf = os.path.join(FIGURES_DIR, 'FigureS6_aggregate_TCA_labeling.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS6_aggregate_TCA_labeling.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
