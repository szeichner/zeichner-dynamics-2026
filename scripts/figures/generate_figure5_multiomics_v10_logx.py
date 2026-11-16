#!/usr/bin/env python3
"""
Figure 5 v10 — Multiomics coupling + conceptual model — LOG X-AXIS variant.

Panels A–C use true log-scale x-axis (hours).
Panels D–E (conceptual model) remain as-is (schematic, not data-driven axes).

Output: figures/output/log_xaxis/Figure5_multiomics_v10_logx.pdf/.png
"""

import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude'
PROGRAM_FILE = os.path.join(BASE, 'data/derived/kmeans_program_assignments.tsv')
CO2_FILE = os.path.join(BASE, 'data/co2/sugar_experiment_CO2_mgC_withJarInfo.csv')
QSIP_FILE = os.path.join(BASE, 'data/qSIP/output/tables/qsip_community_incorporation_efficiency.tsv')
METAB_DIR = os.path.join(BASE, 'data/metabolomics')
OUT_DIR = os.path.join(BASE, 'figures/output/log_xaxis')
os.makedirs(OUT_DIR, exist_ok=True)

FF = 'Arial'
MM = 1 / 25.4

OLIVE = '#6B7C2E'; LIGHT_GREEN = '#A2C060'; PLUM = '#7B3F6E'
TEAL = '#2A9D8F'; CORAL = '#E76F51'; GOLD = '#C8960C'

C_GROWTH = '#000000'
C_TRANSCRIPT = '#888888'
C_METAB = '#457B9D'
C_CO2 = '#C0392B'

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}
MONO_HOURS = [2, 8, 24, 72, 168, 336]
MONO_LABELS = ['2h', '8h', '1d', '3d', '7d', '14d']
HEMI_HOURS = [2, 8, 24, 72, 168, 336, 672]
HEMI_LABELS = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']

TREAT_FILES = {
    'GLUCOSE': ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv', 'GlucoseMetadata.csv'),
    'XYLOSE': ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'CelluloseMetadata.csv'),
}
TREAT_SG = {'GLUCOSE': 'MONOMER', 'XYLOSE': 'MONOMER', 'CELLULOSE': 'HEMICELLULOSE'}
PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']

def minmax(vals):
    arr = np.array(vals, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)

# ============================================================================
# Load data
# ============================================================================
print("Loading programs...")
assign = pd.read_csv(PROGRAM_FILE, sep='\t')
lfc_cols = [c for c in assign.columns if c.startswith('MONOMER_') or c.startswith('HEMICELLULOSE_')]
progs = {}
for prog in PROG_ORDER:
    progs[prog] = assign[assign.mnemonic == prog][lfc_cols].mean(axis=0)

print("Loading CO2 and computing normalized excess 13C rates...")
R_VPDB = 0.0112372
C13_ADDED_MG = {'glucose': 5.155555556, 'xylose': 1.785119563, 'cellulose': 8.888888889}
RATE_CFG = {
    1: (1, 1), 2: (2, 2), 4: (4, 3), 8: (8, 4), 24: (24, 24),
    72: (72, 1), 168: (168, 1), 336: (336, 1), 672: (672, 1),
}

co2_raw = pd.read_csv(CO2_FILE)
co2_raw.columns = co2_raw.columns.str.strip()
co2_raw = co2_raw.rename(columns={
    'δ13CVPDB (‰)': 'delta13C', 'time point': 'time_h',
    'Converted CO2 (mg)': 'co2_mg', 'grams in jar': 'soil_g',
})
co2_raw['R_ratio'] = R_VPDB * (1 + co2_raw['delta13C'] / 1000)
co2_raw['F'] = co2_raw['R_ratio'] / (1 + co2_raw['R_ratio'])
F_baseline = co2_raw[co2_raw['treatment'].isin(['basal', 'control'])]['F'].mean()

co2_by_treat = {}
for treat in ['glucose', 'xylose', 'cellulose']:
    tdf = co2_raw[co2_raw['treatment'] == treat]
    c13_added = C13_ADDED_MG[treat]
    tp_rates = {}
    for nom_tp, (use_t, dt_h) in RATE_CFG.items():
        sub = tdf[tdf['time_h'] == use_t]
        if len(sub) == 0:
            continue
        rates = []
        for _, row in sub.iterrows():
            excess_F = row['F'] - F_baseline
            flux_per_g = (excess_F * row['co2_mg'] * 1000) / dt_h / row['soil_g']
            added_per_g = c13_added * 1000 / row['soil_g']
            rates.append(flux_per_g / added_per_g)
        tp_rates[nom_tp] = np.mean(rates)
    co2_by_treat[treat] = tp_rates

print("Loading qSIP...")
qsip = pd.read_csv(QSIP_FILE, sep='\t')
qsip_lookup = {}
for _, row in qsip.iterrows():
    key = (row['marker'], row['treatment'])
    qsip_lookup.setdefault(key, {})[int(row['timepoint'])] = float(row['weighted_mean_EAF'])

print("Loading metabolomics...")
def load_metab(treat, met):
    ff, mf = TREAT_FILES[treat]
    frac = pd.read_csv(os.path.join(METAB_DIR, ff))
    meta = pd.read_csv(os.path.join(METAB_DIR, mf))
    m0 = frac[(frac.metabolite == met) & (frac.isotopologue == 0)]
    if len(m0) == 0: return {}
    sc = [c for c in frac.columns if c not in ['metabolite', 'isotopologue']]
    vals = m0[sc].iloc[0]
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta.iterrows()}
    tp = {}
    for s in sc:
        if str(s) not in stp: continue
        td = stp[str(s)]
        if td == 0 or td not in METAB_TO_METAT: continue
        tp.setdefault(METAB_TO_METAT[td], []).append(1.0 - float(vals[s]))
    return {t: np.mean(v) for t, v in tp.items()}

def load_tca_agg(treat):
    agg = {}
    for met in ['Cit', 'aKG', 'Suc', 'Fum', 'Mal']:
        md = load_metab(treat, met)
        for t, v in md.items():
            agg.setdefault(t, []).append(v)
    return {t: np.mean(v) for t, v in agg.items()}

naa_g = load_metab('GLUCOSE', 'NAA')
suc_g = load_metab('GLUCOSE', 'Suc')
asp_g = load_metab('GLUCOSE', 'Asp')
tca_h = load_tca_agg('CELLULOSE')

# Colors for panel A metabolites
C_NAA = OLIVE     # NAA in program color (olive green)
C_SUC = C_METAB   # Succinate in standard metabolite blue

# ============================================================================
# Helpers — modified for log x-axis
# ============================================================================
def gather_and_normalize_logx(signals, hours_list):
    """Like gather_and_normalize but returns actual hour values as positions."""
    results = []
    for name, raw, color, ls, marker in signals:
        if isinstance(raw, dict):
            vals, positions = [], []
            for h in hours_list:
                if h in raw and not np.isnan(raw[h]):
                    vals.append(raw[h]); positions.append(float(h))
            if not vals:
                results.append((name, [], [], color, ls, marker)); continue
            results.append((name, positions, minmax(vals).tolist(), color, ls, marker))
        else:
            vals, positions = [], []
            for i, v in enumerate(raw):
                if not np.isnan(v): vals.append(v); positions.append(float(hours_list[i]))
            if not vals:
                results.append((name, [], [], color, ls, marker)); continue
            results.append((name, positions, minmax(vals).tolist(), color, ls, marker))
    return results

def style_panel_logx(ax, hours_list, labels, title, letter):
    ax.set_xscale('log')
    ax.set_xticks(hours_list)
    ax.set_xticklabels(labels, fontsize=9, fontfamily=FF)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(1.5, hours_list[-1] * 1.3)
    ax.set_ylabel('Normalized signal (0\u20131)', fontsize=9, fontfamily=FF)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=9, fontfamily=FF)
    ax.set_title(title, fontsize=11, fontweight='bold', fontfamily=FF, pad=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.8, labelsize=8)
    ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', fontfamily=FF)

# ============================================================================
# Build signals for panels A-C
# ============================================================================
p1_lfc = {h: progs['P_EARLY_MONO'][f'MONOMER_t{h}'] for h in MONO_HOURS}
p2_lfc = {h: progs['P_LATE_MONO'][f'MONOMER_t{h}'] for h in MONO_HOURS}
p3_lfc = {h: progs['P_HEMI'][f'HEMICELLULOSE_t{h}'] for h in HEMI_HOURS}
bact_g = qsip_lookup.get(('16S', 'Glucose'), {})
bact_h = qsip_lookup.get(('16S', 'Hemicellulose'), {})
its_g = qsip_lookup.get(('ITS', 'Glucose'), {})
co2_g = co2_by_treat.get('glucose', {})
co2_c = co2_by_treat.get('cellulose', {})

sigA = [('16S ¹³C assimilation', bact_g, C_GROWTH, '-', '^'),
        ('Transcription (LFC)', p1_lfc, C_TRANSCRIPT, '-', 'o'),
        ('NAA labeling', naa_g, C_NAA, ':', 'P'),
        ('Succinate labeling', suc_g, C_SUC, ':', 's'),
        ('$^{13}$C-CO$_2$ rate', co2_g, C_CO2, '--', 'D')]

sigB = [('ITS ¹³C assimilation', its_g, C_GROWTH, '-', '^'),
        ('Transcription (LFC)', p2_lfc, C_TRANSCRIPT, '-', 'o'),
        ('Metabolite labeling', asp_g, C_METAB, ':', 's')]

sigC = [('16S ¹³C assimilation', bact_h, C_GROWTH, '-', '^'),
        ('Transcription (LFC)', p3_lfc, C_TRANSCRIPT, '-', 'o'),
        ('Metabolite labeling', tca_h, C_METAB, ':', 's'),
        ('$^{13}$C-CO$_2$ rate', co2_c, C_CO2, '--', 'D')]

# ============================================================================
# Conceptual model helpers
# ============================================================================
def smooth(x, y, n=400):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 4:
        xi = np.linspace(x[0], x[-1], n)
        return xi, np.interp(xi, x, y)
    spl = make_interp_spline(x, y, k=3)
    xi = np.linspace(x[0], x[-1], n)
    return xi, np.clip(spl(xi), 0, None)

# ============================================================================
# Create figure
# ============================================================================
print("Creating figure...")
fig = plt.figure(figsize=(200 * MM, 240 * MM))

gs = GridSpec(3, 1, figure=fig, height_ratios=[0.30, 0.05, 0.66],
              hspace=0.30, left=0.09, right=0.96, top=0.96, bottom=0.04)
gs_top = gs[0].subgridspec(1, 3, wspace=0.50)
gs_bot = gs[2].subgridspec(2, 1, hspace=0.35, height_ratios=[1, 0.85])

# ---- Panel A — LOG X-AXIS ----
ax_a = fig.add_subplot(gs_top[0, 0])
for name, pos, nv, color, ls, marker in gather_and_normalize_logx(sigA, MONO_HOURS):
    if pos: ax_a.plot(pos, nv, color=color, linestyle=ls, marker=marker,
                      linewidth=2.0, markersize=6, markeredgecolor='black',
                      markeredgewidth=0.4, zorder=3, label=name, clip_on=False)
style_panel_logx(ax_a, MONO_HOURS, MONO_LABELS, 'P1: monomer assimilation (glucose)', 'A')
ax_a.set_title('P1: monomer assimilation (glucose)', fontsize=11, fontweight='bold',
               fontfamily=FF, pad=6, color=OLIVE)
# Metabolite annotations for panel A
ax_a.text(0.98, 0.52, 'NAA', fontsize=8, fontfamily=FF, color=C_NAA, fontweight='bold',
          ha='right', va='bottom', transform=ax_a.transAxes,
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)
ax_a.text(0.98, 0.38, 'Succinate', fontsize=8, fontfamily=FF, color=C_SUC, fontweight='bold',
          ha='right', va='bottom', transform=ax_a.transAxes,
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# ---- Panel B — LOG X-AXIS ----
ax_b = fig.add_subplot(gs_top[0, 1])
for name, pos, nv, color, ls, marker in gather_and_normalize_logx(sigB, MONO_HOURS):
    if pos: ax_b.plot(pos, nv, color=color, linestyle=ls, marker=marker,
                      linewidth=2.0, markersize=6, markeredgecolor='black',
                      markeredgewidth=0.4, zorder=3, label=name, clip_on=False)
style_panel_logx(ax_b, MONO_HOURS, MONO_LABELS, 'P2: late succession (glucose)', 'B')
ax_b.set_title('P2: late succession (glucose)', fontsize=11, fontweight='bold',
               fontfamily=FF, pad=6, color=LIGHT_GREEN)
# Metabolite annotation for panel B
ax_b.text(0.98, 0.38, 'Aspartate', fontsize=8, fontfamily=FF, color=C_SUC, fontweight='bold',
          ha='right', va='bottom', transform=ax_b.transAxes,
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# ---- Panel C — LOG X-AXIS ----
ax_c = fig.add_subplot(gs_top[0, 2])
for name, pos, nv, color, ls, marker in gather_and_normalize_logx(sigC, HEMI_HOURS):
    if pos: ax_c.plot(pos, nv, color=color, linestyle=ls, marker=marker,
                      linewidth=2.0, markersize=6, markeredgecolor='black',
                      markeredgewidth=0.4, zorder=3, label=name, clip_on=False)
# No gray band — removed per request
style_panel_logx(ax_c, HEMI_HOURS, HEMI_LABELS, 'P3: hemicellulose community', 'C')
ax_c.set_title('P3: hemicellulose community', fontsize=11, fontweight='bold',
               fontfamily=FF, pad=6, color=PLUM)
# Metabolite annotation for panel C
ax_c.text(0.98, 0.38, 'TCA aggregate', fontsize=8, fontfamily=FF, color=C_SUC, fontweight='bold',
          ha='right', va='bottom', transform=ax_c.transAxes,
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# ---- Shared legend ----
legend_handles = [
    Line2D([0], [0], color=C_GROWTH, linestyle='-', marker='^', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.5, label='¹³C assimilation (qSIP EAF)'),
    Line2D([0], [0], color=C_TRANSCRIPT, linestyle='-', marker='o', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.5, label='Transcription (LFC)'),
    Line2D([0], [0], color=C_NAA, linestyle=':', marker='P', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.5, label='NAA labeling'),
    Line2D([0], [0], color=C_SUC, linestyle=':', marker='s', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.5,
           label='Succinate/Asp/TCA labeling'),
    Line2D([0], [0], color=C_CO2, linestyle='--', marker='D', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.5,
           label='$^{13}$C-CO$_2$ rate'),
]
fig.legend(handles=legend_handles, fontsize=9, loc='center',
           ncol=5, frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fancybox=False,
           bbox_to_anchor=(0.52, 0.655), handletextpad=0.3, columnspacing=0.8)

# ============================================================================
# Panel D: Conceptual model — unchanged (schematic, not data-driven)
# ============================================================================

# ---- D top: Monomer consumers — LOG X-AXIS (hours) ----
ax_d1 = fig.add_subplot(gs_bot[0])
ax_d1.set_facecolor('#FAFAFA')

# Timepoints in actual hours
tp_a_hours = np.array([1.0, 8, 24, 72, 168, 336], dtype=float)
tp_a_labels = ['1h', '8h', '1d', '3d', '7d', '14d']

# Fine grid in log space for smooth curves
t_fine_log = np.logspace(np.log10(1), np.log10(336), 400)

# Phase dividers at 24h and 168h
for xp in [24, 168]:
    ax_d1.axvline(xp, color='#DDDDDD', linewidth=0.5, zorder=0)

# Phase labels using axes fraction coordinates
# early: 1-24h, mid: 24-168h, late: 168-336h
# In log space on [1,336]: log10(1)=0, log10(24)=1.38, log10(168)=2.225, log10(336)=2.526
# Fractional positions: early~0.27, mid~0.71, late~0.92
ax_d1.text(0.18, 1.03, 'early', fontsize=11, fontfamily=FF, ha='center',
           color='#BBBBBB', fontstyle='italic', transform=ax_d1.transAxes)
ax_d1.text(0.58, 1.03, 'mid', fontsize=11, fontfamily=FF, ha='center',
           color='#BBBBBB', fontstyle='italic', transform=ax_d1.transAxes)
ax_d1.text(0.88, 1.03, 'late', fontsize=11, fontfamily=FF, ha='center',
           color='#BBBBBB', fontstyle='italic', transform=ax_d1.transAxes)

# Glucose substrate (exponential decay in log-time)
glu_sub = 0.95 * np.exp(-0.015 * t_fine_log)
ax_d1.fill_between(t_fine_log, 0, glu_sub, color='#D4D9A0', alpha=0.35, zorder=1)
ax_d1.plot(t_fine_log, glu_sub, color=OLIVE, linewidth=1.2, zorder=2)
ax_d1.text(1.5, 0.92, 'glucose', fontsize=9, fontfamily=FF, color=OLIVE, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Xylose substrate (slower decay)
xyl_sub = 0.72 * np.exp(-0.010 * t_fine_log)
ax_d1.fill_between(t_fine_log, 0, xyl_sub, color=GOLD, alpha=0.08, zorder=1)
ax_d1.plot(t_fine_log, xyl_sub, color=GOLD, linewidth=1.2, zorder=2)
ax_d1.text(1.5, 0.60, 'xylose', fontsize=9, fontfamily=FF, color=GOLD, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Bacterial ¹³C assimilation (matches qSIP: rises to peak at 72h, dips 168h, recovers 336h)
y_bact = [0.01, 0.10, 0.30, 0.85, 0.25, 0.55]
xb, vb = smooth(np.log10(tp_a_hours), y_bact)
ax_d1.plot(10**xb, vb, color=TEAL, linewidth=2.0, zorder=5)
ax_d1.text(50, 0.78, 'Bacterial ¹³C\nassimilation', fontsize=8, fontfamily=FF,
           color=TEAL, fontstyle='italic', ha='left',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Fungal ¹³C assimilation (late rise, peaks 14d — matches ITS qSIP)
y_fung = [0.01, 0.02, 0.05, 0.15, 0.30, 0.45]
xf, vf = smooth(np.log10(tp_a_hours), y_fung)
ax_d1.plot(10**xf, vf, color=CORAL, linewidth=1.8, zorder=4)
ax_d1.text(250, 0.50, 'Fungal ¹³C\nassimilation', fontsize=8, fontfamily=FF,
           color=CORAL, fontstyle='italic', ha='center',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# P1: Transcription for monomer assimilation
y_p1 = [0.01, 0.08, 0.38, 0.55, 0.38, 0.18]
xp1, vp1 = smooth(np.log10(tp_a_hours), y_p1)
ax_d1.plot(10**xp1, vp1, color=OLIVE, linewidth=1.8, linestyle=(0, (2, 2)), zorder=4)
ax_d1.text(40, 0.50, 'P1 transcription', fontsize=8,
           fontfamily=FF, color=OLIVE, fontstyle='italic', ha='center',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# P2: Transcription for necromass recycling
y_p2 = [0.01, 0.02, 0.05, 0.14, 0.38, 0.55]
xp2, vp2 = smooth(np.log10(tp_a_hours), y_p2)
ax_d1.plot(10**xp2, vp2, color=LIGHT_GREEN, linewidth=1.8, linestyle=(0, (2, 2)), zorder=4)
ax_d1.text(300, 0.50, 'P2 transcription', fontsize=8,
           fontfamily=FF, color=LIGHT_GREEN, fontstyle='italic', ha='right',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Necromass pool (sigmoid in log space)
log_t = np.log10(t_fine_log)
log_midpoint = np.log10(50)
necro = 0.08 * (1 / (1 + np.exp(-8 * (log_t - log_midpoint))))
ax_d1.fill_between(t_fine_log, 0, necro, color='#CCCCCC', alpha=0.5, zorder=0)
ax_d1.text(250, 0.04, 'Necromass', fontsize=8, fontfamily=FF,
           color='#888888', fontstyle='italic', ha='center')

ax_d1.set_xscale('log')
ax_d1.set_xlim(1, 450)
ax_d1.set_ylim(0, 1.0)
ax_d1.set_xticks(tp_a_hours)
ax_d1.set_xticklabels(tp_a_labels, fontsize=9, fontfamily=FF)
ax_d1.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_d1.set_yticks([])
ax_d1.set_ylabel('Monomer\nconsumers', fontsize=10, fontfamily=FF, fontweight='bold',
                 rotation=0, labelpad=32, va='center')
ax_d1.spines['top'].set_visible(False)
ax_d1.spines['right'].set_visible(False)
ax_d1.spines['left'].set_visible(False)
ax_d1.text(-0.06, 1.08, 'D', transform=ax_d1.transAxes, fontsize=12,
           fontweight='bold', va='top', fontfamily=FF)

# ---- D bottom: Polymer consumers — LOG X-AXIS (hours) ----
ax_d2 = fig.add_subplot(gs_bot[1])
ax_d2.set_facecolor('#FAFAFA')

tp_b_hours = np.array([1.0, 8, 24, 72, 168, 336, 672], dtype=float)
tp_b_labels = ['1h', '8h', '1d', '3d', '7d', '14d', '28d']
t_fine_log_b = np.logspace(np.log10(1), np.log10(672), 400)

# Phase dividers at 24h and 168h
for xp in [24, 168]:
    ax_d2.axvline(xp, color='#DDDDDD', linewidth=0.5, zorder=0)

# Hemicellulose substrate (slow linear depletion in log-time)
log_frac = (np.log10(t_fine_log_b) - np.log10(1)) / (np.log10(672) - np.log10(1))
hemi_sub = 0.85 * (1 - 0.55 * log_frac)
ax_d2.fill_between(t_fine_log_b, 0, hemi_sub, color='#D4B8CC', alpha=0.18, zorder=1)
ax_d2.plot(t_fine_log_b, hemi_sub, color=PLUM, linewidth=1.2, zorder=2)
ax_d2.text(2, 0.88, 'hemicellulose', fontsize=9, fontfamily=FF,
           color=PLUM, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# P3: Transcription — sustained, slowly declining
y_p3 = [0.38, 0.35, 0.33, 0.28, 0.24, 0.20, 0.16]
xp3, vp3 = smooth(np.log10(tp_b_hours), y_p3)
ax_d2.plot(10**xp3, vp3, color=PLUM, linewidth=1.8, linestyle=(0, (2, 2)), zorder=4)
ax_d2.text(15, 0.38, 'P3 transcription',
           fontsize=8, fontfamily=FF, color=PLUM, fontstyle='italic', ha='left',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Bacterial ¹³C assimilation — gradual rise with 168h dip (matches hemicellulose qSIP)
y_bact_b = [0.02, 0.04, 0.07, 0.18, 0.08, 0.32, 0.42]
xbb, vbb = smooth(np.log10(tp_b_hours), y_bact_b)
ax_d2.plot(10**xbb, vbb, color=TEAL, linewidth=2.0, zorder=5)
ax_d2.text(500, 0.46, 'Bacterial ¹³C\nassimilation', fontsize=8, fontfamily=FF,
           color=TEAL, fontstyle='italic', ha='center',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

# Slow fungal ¹³C assimilation
y_fung_b = [0.01, 0.01, 0.02, 0.03, 0.04, 0.08, 0.15]
xfb, vfb = smooth(np.log10(tp_b_hours), y_fung_b)
ax_d2.plot(10**xfb, vfb, color=CORAL, linewidth=1.8, zorder=4)
ax_d2.text(600, 0.10, 'Fungal ¹³C\nassimilation', fontsize=8, fontfamily=FF,
           color=CORAL, fontstyle='italic', ha='center',
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1), zorder=10)

ax_d2.set_xscale('log')
ax_d2.set_xlim(1, 900)
ax_d2.set_ylim(0, 1.0)
ax_d2.set_xticks(tp_b_hours)
ax_d2.set_xticklabels(tp_b_labels, fontsize=9, fontfamily=FF)
ax_d2.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_d2.set_xlabel('Time (log$_{10}$ scale)', fontsize=10, fontfamily=FF)
ax_d2.set_yticks([])
ax_d2.set_ylabel('Polymer\nconsumers', fontsize=10, fontfamily=FF, fontweight='bold',
                 rotation=0, labelpad=32, va='center')
ax_d2.spines['top'].set_visible(False)
ax_d2.spines['right'].set_visible(False)
ax_d2.spines['left'].set_visible(False)

# Save
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(OUT_DIR, f'Figure5_multiomics_v10_logx.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved Figure5_multiomics_v10_logx.pdf/.png")
