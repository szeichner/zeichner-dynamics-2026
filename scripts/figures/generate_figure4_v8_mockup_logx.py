#!/usr/bin/env python3
"""
Figure 4 v8 — 10-panel metabolite layout (C-L) — LOG X-AXIS variant.

Same as generate_figure4_v8_mockup.py but x-axes use true log scale.

Output: figures/output/log_xaxis/Figure4_carbon_fate_metabolites_logx.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output', 'log_xaxis')
os.makedirs(FIGURES_DIR, exist_ok=True)

FF = 'Arial'
MM = 1 / 25.4

TC = {'GLUCOSE': '#6B7C2E', 'XYLOSE': '#C8960C', 'CELLULOSE': '#7B3F6E'}
TL = {'GLUCOSE': 'Glucose', 'XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}
TM = {'GLUCOSE': 'o', 'XYLOSE': '^', 'CELLULOSE': 's'}

AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
MARKER_SIZE = 5
LINE_WIDTH = 1.5

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}
ALL_TPS = [2, 8, 24, 72, 168, 336, 672]

METABOLITES = [
    ('PYR',   'Pyruvate',          '11\u00d7 [9, 18]', True),
    ('LAT',   'Lactate',           '', False),
    ('Cit',   'Citrate',           '', False),
    ('aKG',   '\u03b1-Ketoglutarate', '', False),
    ('Suc',   'Succinate',         '23\u00d7 [14, 47]', True),
    ('Fum',   'Fumarate',          '6\u00d7 [5, 8]', True),
    ('Mal',   'Malate',            '7\u00d7 [6, 9]', True),
    ('Asp',   'Aspartate',         '36\u00d7 [19, 60]', True),
    ('GLUM',  'Glutamate',         '4\u00d7 [3, 6]', True),
    ('NAA',   'N-Acetyl-alanine',  '', False),
]

TREATMENTS_INFO = {
    'GLUCOSE':  ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv',
                 'GlucoseMetadata.csv'),
    'XYLOSE':   ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv',
                 'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv',
                  'CelluloseMetadata.csv'),
}

N_CARBONS = {
    'PYR': 3, 'LAT': 3, 'Cit': 6, 'aKG': 5, 'Suc': 4,
    'Fum': 4, 'Mal': 4, 'Asp': 4, 'GLUM': 5, 'NAA': 3
}

print("Loading metabolite data...")

def load_per_replicate(treatment, metab_code):
    frac_file, meta_file = TREATMENTS_INFO[treatment]
    try:
        frac = pd.read_csv(os.path.join(METAB_DIR, frac_file))
        meta = pd.read_csv(os.path.join(METAB_DIR, meta_file))
    except FileNotFoundError:
        return {}
    sub = frac[frac.metabolite == metab_code]
    if len(sub) == 0:
        return {}
    sample_cols = [c for c in frac.columns if c not in ['metabolite', 'isotopologue']]
    nc = N_CARBONS.get(metab_code, 1)
    weighted = sub[sample_cols].multiply(sub['isotopologue'].values, axis=0)
    frac_label = weighted.sum(axis=0) / nc
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta.iterrows()}
    tp_vals = {}
    for sid in sample_cols:
        if str(sid) not in stp:
            continue
        td = stp[str(sid)]
        if td == 0 or td not in METAB_TO_METAT:
            continue
        mh = METAB_TO_METAT[td]
        tp_vals.setdefault(mh, []).append(float(frac_label[sid]))
    return tp_vals

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=3, width=0.8, labelsize=TICK_LABEL)

print("Generating Figure 4 (metabolites, log x-axis)...")

fig_w = 180 * MM
fig_h = 290 * MM
fig = plt.figure(figsize=(fig_w, fig_h))

gs = gridspec.GridSpec(5, 2, figure=fig,
                       height_ratios=[1, 1, 1, 1, 1],
                       hspace=0.55, wspace=0.35,
                       left=0.14, right=0.95, top=0.96, bottom=0.05)

LETTERS = 'CDEFGHIJKL'

for pi, (code, name, gx_str, gx_sig) in enumerate(METABOLITES):
    row = pi // 2
    col = pi % 2
    ax = fig.add_subplot(gs[row, col])
    letter = LETTERS[pi]

    all_means = []
    for treat in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
        tp_vals = load_per_replicate(treat, code)
        if not tp_vals:
            continue

        tps = sorted(tp_vals.keys())
        means = [np.mean(tp_vals[tp]) for tp in tps]
        sems = [np.std(tp_vals[tp], ddof=1) / np.sqrt(len(tp_vals[tp]))
                if len(tp_vals[tp]) > 1 else 0 for tp in tps]
        all_means.extend(means)
        # Use actual hour values as x coordinates
        xs = np.array(tps, dtype=float)

        ax.plot(xs, means, color=TC[treat], linewidth=LINE_WIDTH, zorder=3)
        ax.fill_between(xs, np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        color=TC[treat], alpha=0.20, zorder=2)
        ax.errorbar(xs, means, yerr=sems, fmt=TM[treat], color=TC[treat],
                    markersize=MARKER_SIZE, markeredgecolor='black',
                    markeredgewidth=0.3, elinewidth=1.0,
                    capsize=2, capthick=0.6, zorder=4)

    # Log-scale x-axis
    ax.set_xscale('log')
    ax.set_xticks(ALL_TPS)
    ax.set_xticklabels([TP_LABELS[t] for t in ALL_TPS], fontsize=TICK_LABEL, fontfamily=FF)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(1.5, 900)

    # Set y-axis with padding for annotation
    data_max = max(all_means) if all_means else 0.5
    y_top = max(data_max * 1.45, 0.15)  # extra headroom for annotation
    ax.set_ylim(-0.02, y_top)

    # G/X annotation — placed in upper-right, above the data
    if gx_str:
        gx_color = '#6B7C2E' if gx_sig else '#999999'
        # Place at top-right in axes coordinates, above data
        ax.text(0.97, 0.97, f'G/X {gx_str}', transform=ax.transAxes,
                fontsize=TICK_LABEL - 1, fontfamily=FF, ha='right', va='top',
                color=gx_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=gx_color, linewidth=0.5, alpha=0.9))

    ax.set_title(name, fontsize=11, fontfamily=FF, fontweight='bold', pad=4)
    if col == 0:
        ax.set_ylabel('Fractional \u00b9\u00b3C labeling', fontsize=AXIS_LABEL, fontfamily=FF)
    ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)
    style_ax(ax)
    ax.text(-0.22, 1.08, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight='bold', va='top', fontfamily=FF)

# Shared legend
handles = [
    Line2D([0], [0], marker=TM[t], color=TC[t], markerfacecolor=TC[t],
           markeredgecolor='black', markeredgewidth=0.3, markersize=MARKER_SIZE + 1,
           linewidth=LINE_WIDTH, label=TL[t])
    for t in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']
]
fig.legend(handles=handles, fontsize=LEGEND_SIZE, loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4)

# Save
for ext in ['pdf', 'png']:
    out = os.path.join(FIGURES_DIR, f'Figure4_carbon_fate_metabolites_logx.{ext}')
    fig.savefig(out, format=ext, dpi=600, bbox_inches='tight')
    print(f"  Saved {out}")
plt.close(fig)
print("Done.")
