#!/usr/bin/env python3
"""
Generate Figure S0 — Metatranscriptomic sequencing quality control summary.

Six-panel figure showing QC metric distributions for all sequencing libraries,
colored by treatment, with threshold lines and flagged samples labeled.

Panels:
  A: Raw read count per sample
  B: QC pass rate (%)              — dashed threshold at 40%
  C: Reads assembled (%)           — dashed threshold at 70%
  D: Contig N50 (bp)               — dashed threshold at 200 bp
  E: GC content (%)
  F: Library strandedness (%)      — dashed threshold at 85%

Data: data/metatranscriptomics/qc/output_stat_summary.csv
      data/metatranscriptomics/qc/metat_readme_metadata_summary.csv

Style: CLAUDE.md (PNAS format, double-column 180 mm wide)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
QC_DIR = os.path.join(BASE_DIR, 'data', 'metatranscriptomics', 'qc')
QC_OUTPUT_STATS_CSV = os.path.join(QC_DIR, 'output_stat_summary.csv')
QC_METADATA_CSV = os.path.join(QC_DIR, 'metat_readme_metadata_summary.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'output')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================
TREATMENT_COLORS = {
    'GLUCOSE':   '#6B7C2E',
    'XYLOSE':    '#C8960C',
    'CELLULOSE': '#7B3F6E',
    'control':   '#9B9B9B',
    'basal':     '#444444',
    'other':     '#BBBBBB',
}
TREATMENT_LABELS = {
    'GLUCOSE':   'Glucose',
    'XYLOSE':    'Xylose',
    'CELLULOSE': 'Hemicellulose',
    'control':   'Control',
    'basal':     'Basal',
}
TREATMENT_MARKERS = {
    'GLUCOSE':   'o',
    'XYLOSE':    '^',
    'CELLULOSE': 's',
    'control':   'D',
    'basal':     'P',
}
TREATMENT_ORDER = ['basal', 'control', 'GLUCOSE', 'XYLOSE', 'CELLULOSE']

FONT_FAMILY = 'Arial'
FONT_AXIS_LABEL = 10
FONT_TICK_LABEL = 8
FONT_PANEL_LABEL = 12
FONT_LEGEND = 8
FONT_ANNOTATION = 8
LINEWIDTH_DATA = 1.5
LINEWIDTH_THRESHOLD = 1.0
MARKER_SIZE = 5
DPI_PNG = 600

MM_TO_INCH = 1 / 25.4
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 130 * MM_TO_INCH

# QC thresholds (must match figure-generation pipelines)
THR_QC_PASS = 40        # %
THR_ASSEMBLY = 70       # %
THR_N50 = 200           # bp
THR_STRANDED = 85       # %
THR_RAW_READS = 50e6    # 50M


def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=2.5, width=0.6, labelsize=FONT_TICK_LABEL)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontsize(FONT_TICK_LABEL)


def add_panel_label(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes,
            fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top',
            fontfamily=FONT_FAMILY)


# ============================================================================
# LOAD AND ANNOTATE DATA
# ============================================================================
print("Loading QC data...")
qc = pd.read_csv(QC_OUTPUT_STATS_CSV)
print(f"  {len(qc)} libraries")

# GC content from the metadata file (joined by sample_name)
meta = pd.read_csv(QC_METADATA_CSV)
gc_lookup = {}
if 'sample_name' in meta.columns and 'gc_content' in meta.columns:
    for _, row in meta.iterrows():
        gc_lookup[row['sample_name']] = row['gc_content']
elif 'gc_content' in meta.columns:
    # Fallback: positional join if no sample_name column
    for i, val in enumerate(meta['gc_content']):
        if i < len(qc):
            gc_lookup[qc['sample_name'].iloc[i]] = val
qc['gc_content'] = qc['sample_name'].map(gc_lookup)
# Convert from fraction (0-1) to percent (0-100) if needed
if qc['gc_content'].max() <= 1.0:
    qc['gc_content'] = qc['gc_content'] * 100


# Extract treatment from sample name
def extract_treatment(name):
    if pd.isna(name):
        return 'other'
    if name.lower().startswith('basal'):
        return 'basal'
    if 'control' in name.lower():
        return 'control'
    if 'CELLULOSE' in name:
        return 'CELLULOSE'
    if 'GLUCOSE_xylose' in name:
        return 'GLUCOSE'
    if 'glucose_XYLOSE' in name:
        return 'XYLOSE'
    return 'other'


# Add sample63 from MRA table (not in QC pipeline — processed separately)
if 'sample63_CELLULOSE_72hrs_rep3' not in qc['sample_name'].values:
    sample63 = pd.DataFrame([{
        'sample_name': 'sample63_CELLULOSE_72hrs_rep3',
        'raw_read_count': 191533472,
        'reads_passed_qc': np.nan,  # not available — processed separately
        'total_bases_assembled': 319702498,
        'pct_reads_assembled': 75.5,
        'pct_fully_stranded': np.nan,  # not available
        'contig_n50': np.nan,  # not available
        'gc_content': 60.5,
    }])
    qc = pd.concat([qc, sample63], ignore_index=True)
    print(f"  Added sample63_CELLULOSE_72hrs_rep3 from MRA table (no QC pipeline data)")

qc['treatment'] = qc['sample_name'].apply(extract_treatment)
qc['qc_pass_rate'] = (qc['reads_passed_qc'] / qc['raw_read_count']) * 100

# Apply current QC flags
qc['flag_qc_pass'] = qc['qc_pass_rate'] < THR_QC_PASS
qc['flag_assembly'] = qc['pct_reads_assembled'] < THR_ASSEMBLY
qc['flag_n50'] = qc['contig_n50'] < THR_N50
qc['flag_stranded'] = qc['pct_fully_stranded'] < THR_STRANDED
qc['flag_reads'] = qc['raw_read_count'] < THR_RAW_READS

# Flag samples excluded for missing KO annotation (not a QC metric failure)
qc['flag_no_annotation'] = qc['sample_name'].isin([
    'sample51_CELLULOSE_2hrs_rep3',
    'sample63_CELLULOSE_72hrs_rep3',
])

qc['total_flags'] = (qc[['flag_qc_pass', 'flag_assembly', 'flag_n50',
                          'flag_stranded', 'flag_reads', 'flag_no_annotation']].sum(axis=1))
qc['flagged'] = qc['total_flags'] > 0

flagged_df = qc[qc['flagged']].copy()
print(f"  {len(flagged_df)} samples flagged for exclusion")
for _, row in flagged_df.iterrows():
    reasons = []
    if row.get('flag_qc_pass', False): reasons.append('QC pass rate')
    if row.get('flag_assembly', False): reasons.append('assembly rate')
    if row.get('flag_n50', False): reasons.append('N50')
    if row.get('flag_stranded', False): reasons.append('strandedness')
    if row.get('flag_reads', False): reasons.append('read count')
    if row.get('flag_no_annotation', False): reasons.append('no KO annotation')
    print(f"    {row['sample_name']} ({', '.join(reasons)})")


# Order samples on x-axis: by treatment then by sample name
treatment_order_map = {t: i for i, t in enumerate(TREATMENT_ORDER)}
qc['_sort_key'] = qc['treatment'].map(treatment_order_map).fillna(99)
qc = qc.sort_values(['_sort_key', 'sample_name']).reset_index(drop=True)
qc['x_pos'] = np.arange(len(qc))


# ============================================================================
# Helper for plotting one metric
# ============================================================================

def short_label(name):
    """Extract just the sample number from a sample name (e.g., 'sample20')."""
    m = re.match(r'(sample\d+)', name)
    if m:
        return m.group(1)
    # Fallback for names like 'basal1', 'basal2', etc.
    return name.split('_')[0]


def plot_qc_panel(ax, ycol, ylabel, threshold=None, threshold_label=None,
                  log_y=False, scale=1.0, ylim=None):
    """
    Plot a QC metric scatter with treatment-colored markers.
    Annotates flagged samples with sample IDs.
    """
    for treatment in TREATMENT_ORDER:
        td = qc[qc['treatment'] == treatment]
        if td.empty:
            continue
        # Non-flagged points
        good = td[~td['flagged']]
        bad = td[td['flagged']]
        if len(good) > 0:
            ax.scatter(good['x_pos'], good[ycol] * scale,
                       c=TREATMENT_COLORS[treatment],
                       marker=TREATMENT_MARKERS[treatment],
                       s=MARKER_SIZE ** 2,
                       edgecolors='black', linewidths=0.3,
                       zorder=3)
        if len(bad) > 0:
            ax.scatter(bad['x_pos'], bad[ycol] * scale,
                       c=TREATMENT_COLORS[treatment],
                       marker=TREATMENT_MARKERS[treatment],
                       s=(MARKER_SIZE + 1) ** 2,
                       edgecolors='red', linewidths=0.8,
                       zorder=4)

    # Threshold line
    if threshold is not None:
        ax.axhline(threshold * scale, color='#888888',
                   linewidth=LINEWIDTH_THRESHOLD, linestyle='--', zorder=2)
        if threshold_label:
            ax.text(0.99, (threshold * scale),
                    threshold_label,
                    transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom',
                    fontsize=6.5, color='#666666',
                    fontfamily=FONT_FAMILY, fontstyle='italic')

    # Annotate flagged samples
    flagged_pts = qc[qc['flagged']]
    for _, row in flagged_pts.iterrows():
        ax.annotate(short_label(row['sample_name']),
                    xy=(row['x_pos'], row[ycol] * scale),
                    xytext=(3, -8), textcoords='offset points',
                    fontsize=6, fontfamily=FONT_FAMILY,
                    color='#B22222', rotation=0, ha='left')

    if log_y:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel('Sample (ordered by treatment)',
                  fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
    ax.set_xticks([])
    apply_style(ax)


# ============================================================================
# FIGURE
# ============================================================================
print("Generating Figure S0...")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=False)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.40,
                      left=0.07, right=0.97, top=0.88, bottom=0.08)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

# Panel A: raw read count (in millions)
plot_qc_panel(ax_a, 'raw_read_count', 'Raw read count (M)',
              threshold=THR_RAW_READS,
              threshold_label=f'{int(THR_RAW_READS/1e6)}M',
              scale=1e-6)
add_panel_label(ax_a, 'A')

# Panel B: QC pass rate
plot_qc_panel(ax_b, 'qc_pass_rate', 'QC pass rate (%)',
              threshold=THR_QC_PASS,
              threshold_label=f'{THR_QC_PASS}%')
add_panel_label(ax_b, 'B')

# Panel C: assembly rate
plot_qc_panel(ax_c, 'pct_reads_assembled', 'Reads assembled (%)',
              threshold=THR_ASSEMBLY,
              threshold_label=f'{THR_ASSEMBLY}%')
add_panel_label(ax_c, 'C')

# Panel D: contig N50
plot_qc_panel(ax_d, 'contig_n50', 'Contig N50 (bp)',
              threshold=THR_N50,
              threshold_label=f'{THR_N50} bp')
add_panel_label(ax_d, 'D')

# Panel E: GC content
plot_qc_panel(ax_e, 'gc_content', 'GC content (%)')
add_panel_label(ax_e, 'E')

# Panel F: strandedness
plot_qc_panel(ax_f, 'pct_fully_stranded', 'Fully stranded (%)',
              threshold=THR_STRANDED,
              threshold_label=f'{THR_STRANDED}%')
add_panel_label(ax_f, 'F')


# ============================================================================
# SHARED LEGEND
# ============================================================================
legend_handles = []
present_treatments = [t for t in TREATMENT_ORDER if t in qc['treatment'].values]
for treatment in present_treatments:
    legend_handles.append(
        Line2D([0], [0], marker=TREATMENT_MARKERS[treatment], color='w',
               markerfacecolor=TREATMENT_COLORS[treatment],
               markeredgecolor='black', markeredgewidth=0.3,
               markersize=5, label=TREATMENT_LABELS.get(treatment, treatment))
    )
# Flagged sample marker
legend_handles.append(
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='white', markeredgecolor='red',
           markeredgewidth=1.0, markersize=5, label='Flagged (excluded)')
)

fig.legend(handles=legend_handles, fontsize=FONT_LEGEND,
           loc='upper center', frameon=False,
           bbox_to_anchor=(0.5, 0.99),
           ncol=len(legend_handles),
           handletextpad=0.3, columnspacing=1.0,
           borderpad=0.2)


# ============================================================================
# SUMMARY ANNOTATION
# ============================================================================
n_total = len(qc)
n_flagged = qc['flagged'].sum()
n_passed = n_total - n_flagged
mean_reads = qc['raw_read_count'].mean() / 1e6
mean_pass = qc['qc_pass_rate'].mean()
mean_asm = qc['pct_reads_assembled'].mean()
mean_str = qc['pct_fully_stranded'].mean()

summary = (f'n = {n_total} libraries  |  '
           f'{n_passed} passed QC, {n_flagged} flagged for exclusion  |  '
           f'mean raw reads = {mean_reads:.0f}M, mean QC pass = {mean_pass:.1f}%, '
           f'mean assembly = {mean_asm:.1f}%, mean strandedness = {mean_str:.1f}%')
fig.text(0.5, 0.02, summary, fontsize=8, fontfamily=FONT_FAMILY,
         ha='center', va='bottom', color='#444444', fontstyle='italic')


# ============================================================================
# SAVE
# ============================================================================
out_pdf = os.path.join(FIGURES_DIR, 'FigureS1_metaT_QC.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS1_metaT_QC.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
