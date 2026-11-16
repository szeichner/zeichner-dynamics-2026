#!/usr/bin/env python3
"""
Generate Figure S8 — Bulk EA 13C data (two panels).

Panel A: Raw AT%13C over log-transformed time for all three treatments + control;
         points are treatment means with SEM error bars.

Panel B: Percentage of added 13C remaining in bulk soil over log-transformed
         time. Normalization uses the EA notebook pipeline (notebook 12):
           - Per-sample measured %C used to compute total soil C
           - Measured baseline AT%13C and %C from each treatment's t=0 controls
           - Substrate isotopic purity: 99% for monomers, 97% for cellulose
           - 100% defined as the expected post-addition mass-balance state
         Control is NOT shown in Panel B (normalization not meaningful).

Save as FigureS8_EA_13C_data.pdf and FigureS8_EA_13C_data.png.

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
EA_DIR = os.path.join(BASE_DIR, 'data', 'ea')
EA_GLUCOSE_CSV = os.path.join(EA_DIR, 'glucose_all.csv')
EA_XYLOSE_CSV = os.path.join(EA_DIR, 'xylose_all.csv')
EA_CELLULOSE_CSV = os.path.join(EA_DIR, 'cellulose_all.csv')
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
}
TREATMENT_LABELS = {
    'GLUCOSE':   'Glucose',
    'XYLOSE':    'Xylose',
    'CELLULOSE': 'Hemicellulose',
    'control':   'Control',
}
TREATMENT_MARKERS = {
    'GLUCOSE':   'o',
    'XYLOSE':    '^',
    'CELLULOSE': 's',
    'control':   'D',
}
MAIN_TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']

FONT_FAMILY = 'Arial'
FONT_AXIS_LABEL = 10
FONT_TICK_LABEL = 9
FONT_PANEL_LABEL = 12
FONT_LEGEND = 8
FONT_ANNOTATION = 8
LINEWIDTH_DATA = 1.5
LINEWIDTH_ERROR = 1.0
MARKER_SIZE = 5
CI_ALPHA = 0.15
DPI_PNG = 600

MM_TO_INCH = 1 / 25.4
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 80 * MM_TO_INCH

# ============================================================================
# EA NOTEBOOK PIPELINE PARAMETERS (from notebook 12 / carbon_addition.csv)
# ============================================================================
SOIL_MASS_G = 20.0
TOC_GLUCOSE_XYLOSE_MG = 6.940675119  # total C added (glucose+xylose, both)
C13_GLUCOSE_MG = 5.155555556         # labeled-glucose C
C13_XYLOSE_MG = 1.785119563          # labeled-xylose C
TOC_CELLULOSE_MG = 8.888888889       # total C in hemicellulose treatment
SUBSTRATE_PURITY_MONOMER = 99.0
SUBSTRATE_PURITY_CELLULOSE = 97.0

EA_TREATMENT_PARAMS = {
    'GLUCOSE':   (TOC_GLUCOSE_XYLOSE_MG, C13_GLUCOSE_MG, SUBSTRATE_PURITY_MONOMER),
    'XYLOSE':    (TOC_GLUCOSE_XYLOSE_MG, C13_XYLOSE_MG, SUBSTRATE_PURITY_MONOMER),
    'CELLULOSE': (TOC_CELLULOSE_MG,      TOC_CELLULOSE_MG, SUBSTRATE_PURITY_CELLULOSE),
}


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
    ax.text(-0.16, 1.10, label, transform=ax.transAxes,
            fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top',
            fontfamily=FONT_FAMILY)


# ============================================================================
# EA PIPELINE FUNCTIONS (ported from notebook 12)
# ============================================================================

def ea_load_treatment_file(filepath):
    """Load EA CSV and extract Timepoint from Sample ID."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['Timepoint'] = df['Sample ID'].str.extract(r'_(\d+)hrs').astype(int)
    return df


def ea_group_replicates(df):
    """Compute mean, SD, n per timepoint."""
    means = df.groupby('Timepoint').agg(
        **{
            'AT%13_mean': ('AT%13', 'mean'),
            'AT%13_SD':   ('AT%13', 'std'),
            '%C_mean':    ('%C',    'mean'),
            '%C_SD':      ('%C',    'std'),
            'n':          ('Sample ID', 'count'),
        }
    ).reset_index()
    return means.sort_values('Timepoint').reset_index(drop=True)


def ea_add_se(df):
    """Add SE = SD/sqrt(n) for *_mean cols."""
    out = df.copy()
    for col in [c for c in df.columns if c.endswith('_mean')]:
        base = col.replace('_mean', '')
        sd_col = f'{base}_SD'
        if sd_col in df.columns and 'n' in df.columns:
            out[f'{base}_SE'] = out[sd_col] / np.sqrt(out['n'])
    return out


def ea_normalize_to_100_percent_start(df_with_se, soil_mass_g,
                                       total_C_added_mg, labeled_C_added_mg,
                                       substrate_13C_percent):
    """
    Normalize so that t=0 = 100% of added 13C, with subsequent timepoints
    representing the fraction of added 13C still detectable in bulk soil.
    Mirrors notebook 12's normalize_to_100_percent_start().
    """
    df = df_with_se.copy()

    # Baseline: ORIGINAL t=0 (before substrate addition), measured from controls
    baseline_TOC = df.at[0, '%C_mean']
    baseline_atomic = df.at[0, 'AT%13_mean']
    baseline_TOC_SE = df.at[0, '%C_SE']
    baseline_atomic_SE = df.at[0, 'AT%13_SE']

    baseline_C_mg = (baseline_TOC / 100.0) * soil_mass_g * 1000.0
    baseline_C_SE_mg = (baseline_TOC_SE / 100.0) * soil_mass_g * 1000.0
    baseline_13C_mg = baseline_C_mg * (baseline_atomic / 100.0)
    baseline_13C_SE = baseline_atomic_SE * baseline_C_mg / 100.0

    # Expected post-addition state via mass balance
    initial_C_mg = baseline_C_mg
    soil_13C_mg = initial_C_mg * (baseline_atomic / 100.0)
    substrate_13C_mg = labeled_C_added_mg * (substrate_13C_percent / 100.0)
    total_13C_mg = soil_13C_mg + substrate_13C_mg
    total_C_for_isotope_mg = initial_C_mg + labeled_C_added_mg
    expected_t0_atomic = (total_13C_mg / total_C_for_isotope_mg) * 100.0
    expected_t0_TOC = ((initial_C_mg + total_C_added_mg) /
                       (soil_mass_g * 1000.0)) * 100.0
    expected_t0_C_mg = (expected_t0_TOC / 100.0) * soil_mass_g * 1000.0
    expected_t0_13C_mg = expected_t0_C_mg * (expected_t0_atomic / 100.0)

    added_C_mg = expected_t0_C_mg - baseline_C_mg
    added_13C_mg = expected_t0_13C_mg - baseline_13C_mg

    df['norm_TOC_mean'] = 0.0
    df['norm_TOC_SE'] = 0.0
    df['norm_13C_mean'] = 0.0
    df['norm_13C_SE'] = 0.0

    for idx in df.index:
        if idx == 0:
            df.at[idx, 'norm_TOC_mean'] = 100.0
            df.at[idx, 'norm_13C_mean'] = 100.0
            if added_C_mg > 0:
                df.at[idx, 'norm_TOC_SE'] = (baseline_C_SE_mg / added_C_mg) * 100.0
            if added_13C_mg > 0:
                expected_t0_13C_SE = baseline_atomic_SE * baseline_C_mg / 100.0
                df.at[idx, 'norm_13C_SE'] = (expected_t0_13C_SE / added_13C_mg) * 100.0
        else:
            current_C_mg = (df.at[idx, '%C_mean'] / 100.0) * soil_mass_g * 1000.0
            delta_C_mg = current_C_mg - baseline_C_mg
            df.at[idx, 'norm_TOC_mean'] = (delta_C_mg / added_C_mg * 100.0) \
                if added_C_mg > 0 else 0
            current_C_SE_mg = (df.at[idx, '%C_SE'] / 100.0) * soil_mass_g * 1000.0
            delta_C_SE = np.sqrt(current_C_SE_mg ** 2 + baseline_C_SE_mg ** 2)
            df.at[idx, 'norm_TOC_SE'] = (delta_C_SE / added_C_mg * 100.0) \
                if added_C_mg > 0 else 0

            current_total_C_mg = current_C_mg
            current_13C_mg = current_total_C_mg * (df.at[idx, 'AT%13_mean'] / 100.0)
            excess_13C_mg = current_13C_mg - baseline_13C_mg
            df.at[idx, 'norm_13C_mean'] = (excess_13C_mg / added_13C_mg * 100.0) \
                if added_13C_mg > 0 else 0
            current_13C_SE = df.at[idx, 'AT%13_SE'] * current_total_C_mg / 100.0
            excess_13C_SE = np.sqrt(current_13C_SE ** 2 + baseline_13C_SE ** 2)
            df.at[idx, 'norm_13C_SE'] = (excess_13C_SE / added_13C_mg * 100.0) \
                if added_13C_mg > 0 else 0

    return df


# ============================================================================
# RUN PIPELINE PER TREATMENT
# ============================================================================
print("Loading EA data and running normalization pipeline...")

ea_files = {
    'GLUCOSE':   EA_GLUCOSE_CSV,
    'XYLOSE':    EA_XYLOSE_CSV,
    'CELLULOSE': EA_CELLULOSE_CSV,
}

ea_normalized = {}
for treatment, filepath in ea_files.items():
    raw = ea_load_treatment_file(filepath)
    grouped = ea_group_replicates(raw)
    with_se = ea_add_se(grouped)
    total_C, labeled_C, purity = EA_TREATMENT_PARAMS[treatment]
    norm = ea_normalize_to_100_percent_start(
        with_se,
        soil_mass_g=SOIL_MASS_G,
        total_C_added_mg=total_C,
        labeled_C_added_mg=labeled_C,
        substrate_13C_percent=purity,
    )
    ea_normalized[treatment] = norm
    print(f"  {treatment}: t=0 = {norm.at[0, 'norm_13C_mean']:.0f}%; "
          f"final = {norm['norm_13C_mean'].iloc[-1]:.0f}%")


# ============================================================================
# 13C added per treatment (mg) — for normalization panels
# ============================================================================
C13_ADDED_MG = {
    'GLUCOSE':   5.16,
    'XYLOSE':    1.79,
    'CELLULOSE': 8.89,
}

# ============================================================================
# FIGURE S8 — Two-panel layout (1×2): raw AT%13C + normalized 13C recovery
# ============================================================================
print("Generating Figure S8...")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=False)
gs = fig.add_gridspec(1, 2, wspace=0.40,
                      left=0.08, right=0.96, top=0.90, bottom=0.15)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])


# ============================================================================
# Panel A: Raw AT%13C over time (all treatments + control)
# ============================================================================
# Plot the three treatments
for treatment in MAIN_TREATMENTS:
    td = ea_normalized[treatment].sort_values('Timepoint')
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]

    # Skip the t=0 row (which is the control); plot only t > 0 for treatments
    treat_data = td[td['Timepoint'] > 0]
    ax_a.errorbar(treat_data['Timepoint'], treat_data['AT%13_mean'],
                  yerr=treat_data['AT%13_SE'],
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2, capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)
    ax_a.fill_between(treat_data['Timepoint'],
                      treat_data['AT%13_mean'] - treat_data['AT%13_SE'],
                      treat_data['AT%13_mean'] + treat_data['AT%13_SE'],
                      color=color, alpha=CI_ALPHA, zorder=2)

# Control = t=0 row from glucose treatment (same controls in all 3 files)
ctrl_row = ea_normalized['GLUCOSE'].iloc[0]
# Place control at the leftmost x value for visibility (use 1h as anchor)
ax_a.errorbar([1.0], [ctrl_row['AT%13_mean']],
              yerr=[ctrl_row['AT%13_SE']],
              fmt='D', color=TREATMENT_COLORS['control'],
              markersize=MARKER_SIZE,
              markeredgecolor='black', markeredgewidth=0.3,
              elinewidth=LINEWIDTH_ERROR,
              capsize=2, capthick=LINEWIDTH_ERROR,
              label='Control (t=0)', zorder=3)

ax_a.set_xscale('log')
ax_a.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_a.set_ylabel('Bulk AT%\u00b9\u00b3C',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_a.legend(fontsize=FONT_LEGEND, loc='upper left', frameon=False,
            handletextpad=0.3)
apply_style(ax_a)
add_panel_label(ax_a, 'A')


# ============================================================================
# Panel B: Normalized 13C recovery (treatments only, no control)
# ============================================================================
for treatment in MAIN_TREATMENTS:
    td = ea_normalized[treatment]
    treat_data = td[td['Timepoint'] > 0].sort_values('Timepoint')
    if treat_data.empty:
        continue
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]

    ax_b.errorbar(treat_data['Timepoint'], treat_data['norm_13C_mean'],
                  yerr=treat_data['norm_13C_SE'],
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2, capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)
    ax_b.fill_between(treat_data['Timepoint'],
                      treat_data['norm_13C_mean'] - treat_data['norm_13C_SE'],
                      treat_data['norm_13C_mean'] + treat_data['norm_13C_SE'],
                      color=color, alpha=CI_ALPHA, zorder=2)

# 100% reference line (dashed)
ax_b.axhline(100, color='#999999', linewidth=0.8, linestyle='--', zorder=1)
ax_b.text(0.97, 0.96,
          '100% = expected \u00b9\u00b3C immediately\nafter substrate addition',
          transform=ax_b.transAxes, fontsize=FONT_ANNOTATION - 0.5,
          fontfamily=FONT_FAMILY, fontstyle='italic',
          color='#666666', ha='right', va='top')

ax_b.set_xscale('log')
ax_b.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_b.set_ylabel('\u00b9\u00b3C remaining\nin bulk soil (%)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
apply_style(ax_b)
add_panel_label(ax_b, 'B')


# ============================================================================
# SAVE
# ============================================================================
out_pdf = os.path.join(FIGURES_DIR, 'FigureS8_EA_13C_data.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS8_EA_13C_data.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
