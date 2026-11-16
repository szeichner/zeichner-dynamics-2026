#!/usr/bin/env python3
"""
Generate Figure S9 — Headspace CO2 respiration: δ13C and total CO2 flux over
time. Pipeline ported from notebooks/process_gas_bench_data_rates.R.

Two-panel figure:
  Panel A: δ13C (‰ VPDB) of respired CO2 over time for all three treatments
           plus control + basal; reference line at -8‰ (atmospheric)
  Panel B: Total CO2 flux (paired and tracked measurements) over time

Style: CLAUDE.md (PNAS format, double-column 180 mm wide)
"""

import os
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
CO2_CSV = os.path.join(BASE_DIR, 'data', 'co2',
                       'sugar_experiment_CO2_mgC_withJarInfo.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'output')
TABLES_DIR = os.path.join(BASE_DIR, 'tables', 'output')
os.makedirs(TABLES_DIR, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================
TREATMENT_COLORS = {
    'glucose':   '#6B7C2E', 'xylose':    '#C8960C',
    'cellulose': '#7B3F6E', 'control':   '#9B9B9B',
    'basal':     '#444444',
}
TREATMENT_LABELS = {
    'glucose':   'Glucose', 'xylose':    'Xylose',
    'cellulose': 'Hemicellulose', 'control':   'Control', 'basal':     'Basal',
}
TREATMENT_MARKERS = {
    'glucose':   'o', 'xylose':    '^', 'cellulose': 's',
    'control':   'D', 'basal':     'P',
}
PLOT_TREATMENTS = ['glucose', 'xylose', 'cellulose', 'control', 'basal']

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


def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=2.5, width=0.6, labelsize=FONT_TICK_LABEL)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def add_panel_label(ax, label):
    ax.text(-0.16, 1.10, label, transform=ax.transAxes,
            fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top',
            fontfamily=FONT_FAMILY)


# ============================================================================
# LOAD CO2 DATA
# ============================================================================
print("Loading CO2 data...")
df = pd.read_csv(CO2_CSV)
df = df.rename(columns={
    'jar number': 'jar_number',
    'grams in jar': 'grams_in_jar',
    'time point': 'time_point',
    'δ13CVPDB (‰)': 'delta13C',
    'at-% 13C': 'at_pct_13C',
    'Converted CO2 (mg)': 'converted_CO2_mg',
})

df['converted_CO2_mg'] = pd.to_numeric(df['converted_CO2_mg'], errors='coerce')
# Mass conversions
df['mg_total_C'] = df['converted_CO2_mg'] * (12.01 / 44.01)
df['mg_13C'] = df['mg_total_C'] * (df['at_pct_13C'] / 100.0)

print(f"  {len(df)} rows; treatments: {df['treatment'].value_counts().to_dict()}")
print(f"  Timepoints: {sorted(df['time_point'].unique())}")


# ============================================================================
# PAIRED FLUXES (T0/T1 pairs at 71-72, 167-168, 335-336, 671-672)
# ============================================================================
print("Computing paired fluxes...")
paired_pairs = {71: 72, 167: 168, 335: 336, 671: 672}
T0_TIMES = list(paired_pairs.keys())
T1_TIMES = list(paired_pairs.values())

paired_rows = []
for jar in df['jar_number'].unique():
    jar_data = df[df['jar_number'] == jar]
    for t0, t1 in paired_pairs.items():
        d0 = jar_data[jar_data['time_point'] == t0]
        d1 = jar_data[jar_data['time_point'] == t1]
        if len(d0) == 0 or len(d1) == 0:
            continue
        d0 = d0.iloc[0]
        d1 = d1.iloc[0]
        if pd.isna(d0['mg_total_C']) or pd.isna(d1['mg_total_C']):
            continue
        grams = d0['grams_in_jar']
        delta_t = 1.0
        flux_C = (d1['mg_total_C'] - d0['mg_total_C']) / (grams * delta_t)
        flux_13C = (d1['mg_13C'] - d0['mg_13C']) / (grams * delta_t)
        paired_rows.append({
            'jar': jar,
            'treatment': d0['treatment'],
            'replicate': d0['replicate'],
            'time_point_plot': t1,
            'flux_total_C': flux_C,
            'flux_13C': flux_13C,
            'delta13C_T1': d1['delta13C'],
            'at_pct_13C_T1': d1['at_pct_13C'],
        })
paired = pd.DataFrame(paired_rows)
print(f"  {len(paired)} paired flux measurements")


# ============================================================================
# JAR-TRACKED EARLY FLUXES (1, 4, 8 h)
# ============================================================================
print("Computing tracked early fluxes...")
EARLY_TIMES = [1, 4, 8]

# Find jars that have multiple early-time measurements
early = df[df['time_point'].isin(EARLY_TIMES)].copy()
jar_counts = early.groupby('jar_number')['time_point'].nunique()
tracked_jars = jar_counts[jar_counts > 1].index

tracked_rows = []
for jar in tracked_jars:
    jar_early = early[early['jar_number'] == jar].sort_values('time_point')
    prev_C = None
    prev_13C = None
    prev_t = None
    grams = jar_early['grams_in_jar'].iloc[0]
    treatment = jar_early['treatment'].iloc[0]
    rep = jar_early['replicate'].iloc[0]
    for _, row in jar_early.iterrows():
        if prev_C is not None and not pd.isna(row['mg_total_C']) and not pd.isna(prev_C):
            delta_t = row['time_point'] - prev_t
            flux_C = (row['mg_total_C'] - prev_C) / (grams * delta_t)
            flux_13C = (row['mg_13C'] - prev_13C) / (grams * delta_t)
            tracked_rows.append({
                'jar': jar, 'treatment': treatment, 'replicate': rep,
                'time_point_plot': row['time_point'],
                'flux_total_C': flux_C,
                'flux_13C': flux_13C,
                'delta13C': row['delta13C'],
                'at_pct_13C': row['at_pct_13C'],
            })
        prev_C = row['mg_total_C']
        prev_13C = row['mg_13C']
        prev_t = row['time_point']
tracked = pd.DataFrame(tracked_rows)
print(f"  {len(tracked)} jar-tracked early flux measurements")


# ============================================================================
# Combine flux data: paired + tracked
# ============================================================================
flux_long = pd.concat([
    paired[['treatment', 'replicate', 'time_point_plot', 'flux_total_C', 'flux_13C']],
    tracked[['treatment', 'replicate', 'time_point_plot', 'flux_total_C', 'flux_13C']],
], ignore_index=True)

flux_summary = flux_long.groupby(['treatment', 'time_point_plot']).agg(
    n=('flux_total_C', 'count'),
    mean_flux=('flux_total_C', 'mean'),
    sem_flux=('flux_total_C', 'sem'),
    mean_flux_13C=('flux_13C', 'mean'),
    sem_flux_13C=('flux_13C', 'sem'),
).reset_index()


# ============================================================================
# δ13C of respired CO2 (using all measurements, including singles)
# ============================================================================
print("Computing δ13C summary...")
delta_summary = df[df['time_point'] > 0].groupby(['treatment', 'time_point']).agg(
    n=('delta13C', 'count'),
    mean_delta13C=('delta13C', 'mean'),
    sem_delta13C=('delta13C', 'sem'),
    mean_at13=('at_pct_13C', 'mean'),
    sem_at13=('at_pct_13C', 'sem'),
).reset_index()


# ============================================================================
# 13C added per treatment (mg) — for normalization panels
# ============================================================================
C13_ADDED_MG = {
    'glucose':   5.16,
    'xylose':    1.79,
    'cellulose': 8.89,
}

# ============================================================================
# FIGURE S9 — four-panel layout (2x2): raw + normalized to 13C added
# ============================================================================
print("Generating Figure S9...")

FIG_HEIGHT_4 = 160 * MM_TO_INCH  # taller for 2 rows

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_4), constrained_layout=False)
gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.45,
                      left=0.08, right=0.97, top=0.94, bottom=0.08)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ---- Panel A: δ13C of respired CO2 ----
for treatment in PLOT_TREATMENTS:
    td = delta_summary[delta_summary['treatment'] == treatment].sort_values('time_point')
    if td.empty:
        continue
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]

    ax_a.errorbar(td['time_point'], td['mean_delta13C'],
                  yerr=td['sem_delta13C'],
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2,
                  capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)
    ax_a.fill_between(td['time_point'],
                      td['mean_delta13C'] - td['sem_delta13C'].fillna(0),
                      td['mean_delta13C'] + td['sem_delta13C'].fillna(0),
                      color=color, alpha=CI_ALPHA, zorder=2)

# Reference line at atmospheric ~−8‰
ax_a.axhline(-8, color='#999999', linewidth=0.8, linestyle='--', zorder=1)
ax_a.text(0.97, 0.04, 'Atmospheric (\u22128\u2030)',
          transform=ax_a.transAxes, fontsize=FONT_ANNOTATION - 0.5,
          fontfamily=FONT_FAMILY, fontstyle='italic', color='#666666',
          ha='right', va='bottom')

ax_a.set_xscale('log')
ax_a.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_a.set_ylabel(r'$\delta^{13}$C of respired CO$_2$ (‰ VPDB)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_a.legend(fontsize=FONT_LEGEND, loc='upper left', frameon=False,
            handletextpad=0.3, ncol=2)
apply_style(ax_a)
add_panel_label(ax_a, 'A')


# ---- Panel B: Total CO2 flux ----
flux_treatments = ['glucose', 'xylose', 'cellulose']

for treatment in flux_treatments:
    td = flux_summary[flux_summary['treatment'] == treatment].sort_values('time_point_plot')
    if td.empty:
        continue
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]

    ax_b.errorbar(td['time_point_plot'], td['mean_flux'],
                  yerr=td['sem_flux'],
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2,
                  capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)

ax_b.axhline(0, color='#CCCCCC', linewidth=0.5, zorder=1)
ax_b.set_xscale('log')
ax_b.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_b.set_ylabel(r'CO$_2$ flux' + '\n' +
                r'(mg C g$^{-1}$ soil h$^{-1}$)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_b.legend(fontsize=FONT_LEGEND, loc='upper right', frameon=False,
            handletextpad=0.3)
apply_style(ax_b)
add_panel_label(ax_b, 'B')

# ---- Panel C: δ13C normalized by 13C added ----
ax_c = fig.add_subplot(gs[1, 0])

# Compute baseline delta13C from control/basal average
baseline_rows = delta_summary[delta_summary['treatment'].isin(['control', 'basal'])]
baseline_delta = baseline_rows['mean_delta13C'].mean() if not baseline_rows.empty else -25.0

for treatment in ['glucose', 'xylose', 'cellulose']:
    td = delta_summary[delta_summary['treatment'] == treatment].sort_values('time_point')
    if td.empty:
        continue
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]
    c13_added = C13_ADDED_MG[treatment]

    norm_delta = (td['mean_delta13C'] - baseline_delta) / c13_added
    norm_sem = td['sem_delta13C'].fillna(0) / c13_added

    ax_c.errorbar(td['time_point'], norm_delta,
                  yerr=norm_sem,
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2,
                  capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)
    ax_c.fill_between(td['time_point'],
                      norm_delta - norm_sem,
                      norm_delta + norm_sem,
                      color=color, alpha=CI_ALPHA, zorder=2)

ax_c.set_xscale('log')
ax_c.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_c.set_ylabel(r'Excess $\delta^{13}$C per mg' + '\n' +
                r'$^{13}$C added (\u2030 mg$^{-1}$)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_c.legend(fontsize=FONT_LEGEND, loc='upper left', frameon=False,
            handletextpad=0.3)
apply_style(ax_c)
add_panel_label(ax_c, 'C')


# ---- Panel D: Total CO2 flux normalized by 13C added ----
ax_d = fig.add_subplot(gs[1, 1])

for treatment in ['glucose', 'xylose', 'cellulose']:
    td = flux_summary[flux_summary['treatment'] == treatment].sort_values('time_point_plot')
    if td.empty:
        continue
    color = TREATMENT_COLORS[treatment]
    label = TREATMENT_LABELS[treatment]
    marker = TREATMENT_MARKERS[treatment]
    c13_added = C13_ADDED_MG[treatment]

    norm_flux = td['mean_flux'] / c13_added
    norm_sem = td['sem_flux'] / c13_added

    ax_d.errorbar(td['time_point_plot'], norm_flux,
                  yerr=norm_sem,
                  fmt=marker + '-', color=color,
                  markersize=MARKER_SIZE, linewidth=LINEWIDTH_DATA,
                  elinewidth=LINEWIDTH_ERROR, capsize=2,
                  capthick=LINEWIDTH_ERROR,
                  markeredgecolor='black', markeredgewidth=0.3,
                  label=label, zorder=3)

ax_d.axhline(0, color='#CCCCCC', linewidth=0.5, zorder=1)
ax_d.set_xscale('log')
ax_d.set_xlabel(r'Time (log$_{10}$ scale)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_d.set_ylabel(r'CO$_2$ flux per mg $^{13}$C added' + '\n' +
                r'(mg C g$^{-1}$ h$^{-1}$ mg$^{-1}$)',
                fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
ax_d.legend(fontsize=FONT_LEGEND, loc='upper right', frameon=False,
            handletextpad=0.3)
apply_style(ax_d)
add_panel_label(ax_d, 'D')


out_pdf = os.path.join(FIGURES_DIR, 'FigureS9_CO2_respiration.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS9_CO2_respiration.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
