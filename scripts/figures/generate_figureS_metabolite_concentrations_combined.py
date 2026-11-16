#!/usr/bin/env python3
"""
Supplementary Figure — Pore water metabolite concentrations (µM),
all three treatments shown together on each panel.

Outputs: FigureS_metabolite_concentrations_combined.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

TC = {'GLUCOSE': '#6B7C2E', 'XYLOSE': '#C8960C', 'CELLULOSE': '#7B3F6E'}
TL = {'GLUCOSE': 'Glucose', 'XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}
TM = {'GLUCOSE': 'o', 'XYLOSE': '^', 'CELLULOSE': 's'}

AXIS_LABEL = 10
TICK_LABEL = 8
PANEL_LABEL = 12
MARKER_SIZE = 5
LINE_WIDTH = 1.2

TP_DAY_TO_HOURS = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}
ALL_TPS = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

DILUTION_FACTOR = 2250 / 1250

TREATMENTS_INFO = {
    'GLUCOSE':  ('GlucoseResults.csv', 'GlucoseMetadata.csv'),
    'XYLOSE':   ('XyloseResults.csv', 'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults.csv', 'CelluloseMetadata.csv'),
}

METABOLITES = [
    ('PYR',   'Pyruvate'),
    ('LAT',   'Lactate'),
    ('Cit',   'Citrate'),
    ('aKG',   '\u03b1-Ketoglutarate'),
    ('Suc',   'Succinate'),
    ('Fum',   'Fumarate'),
    ('Mal',   'Malate'),
    ('Asp',   'Aspartate'),
    ('GLUM',  'Glutamate'),
    ('NAA',   'N-Acetyl-alanine'),
]

OUTPUT_STEM = 'FigureS_metabolite_concentrations_combined'


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(direction='out', length=2.5, width=0.6, labelsize=TICK_LABEL)


# ============================================================================
# Build standard curve converters
# ============================================================================
print("Building standard curves...")
std_df = pd.read_csv(os.path.join(METAB_DIR, 'StandardConcentration.csv'))
std_df['Metabolite'] = std_df['Unnamed: 0'].ffill()
sum_std = std_df.groupby('Metabolite').sum(numeric_only=True)
sum_std = sum_std.drop(columns=['Unnamed: 1'], errors='ignore')

concentrations = np.array([float(c.replace('uM', '')) for c in sum_std.columns])

converters = {}
for met in sum_std.index:
    ion_counts = sum_std.loc[met].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(ion_counts)
    model = LinearRegression().fit(X_poly, concentrations)
    converters[met] = (lambda m=model, p=poly: lambda x: m.predict(p.transform(np.array([[x]])))[0])()


# ============================================================================
# Load concentration data
# ============================================================================
def load_concentration_uM(treatment, metab_code):
    results_file, meta_file = TREATMENTS_INFO[treatment]
    results = pd.read_csv(os.path.join(METAB_DIR, results_file))
    meta = pd.read_csv(os.path.join(METAB_DIR, meta_file))

    results['Metabolite'] = results['Metabolite'].ffill()
    met_rows = results[results['Metabolite'] == metab_code]
    if len(met_rows) == 0:
        return {}

    sample_cols = [c for c in results.columns if c not in ['Metabolite', 'Isotopologue']]
    totals = met_rows[sample_cols].sum(axis=0)

    if metab_code not in converters:
        return {}
    conv = converters[metab_code]

    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta.iterrows()}

    tp_vals = {}
    for sid in sample_cols:
        if str(sid) not in stp:
            continue
        td = stp[str(sid)]
        if td == 0 or td not in TP_DAY_TO_HOURS:
            continue
        mh = TP_DAY_TO_HOURS[td]
        area = float(totals[sid])
        if pd.notna(area) and area > 0:
            conc = max(0, conv(area) * DILUTION_FACTOR)
            tp_vals.setdefault(mh, []).append(conc)

    return tp_vals


# ============================================================================
# Generate combined figure — all treatments on each panel
# ============================================================================
n_mets = len(METABOLITES)
n_cols = 2
n_rows = (n_mets + 1) // 2

print(f"Generating combined concentration figure ({n_mets} panels, 3 treatments each)...")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 * MM, n_rows * 55 * MM))
fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.08, hspace=0.50, wspace=0.35)

panel_labels = [chr(ord('A') + i) for i in range(n_mets)]

for idx, (code, name) in enumerate(METABOLITES):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col]

    for treat_key in ['GLUCOSE', 'XYLOSE', 'CELLULOSE']:
        color = TC[treat_key]
        marker = TM[treat_key]
        tp_vals = load_concentration_uM(treat_key, code)

        if not tp_vals:
            continue

        tps = sorted(tp_vals.keys())
        means = [np.mean(tp_vals[tp]) for tp in tps]
        sems = [np.std(tp_vals[tp], ddof=1) / np.sqrt(len(tp_vals[tp]))
                if len(tp_vals[tp]) > 1 else 0 for tp in tps]
        xs = [tp for tp in tps if tp in ALL_TPS]
        means_f = [means[i] for i, tp in enumerate(tps) if tp in ALL_TPS]
        sems_f = [sems[i] for i, tp in enumerate(tps) if tp in ALL_TPS]

        ax.plot(xs, means_f, color=color, linewidth=LINE_WIDTH, zorder=3)
        ax.fill_between(xs, np.array(means_f) - np.array(sems_f),
                        np.array(means_f) + np.array(sems_f),
                        color=color, alpha=0.15, zorder=2)
        ax.errorbar(xs, means_f, yerr=sems_f, fmt=marker, color=color,
                    markersize=MARKER_SIZE, markeredgecolor='black',
                    markeredgewidth=0.3, elinewidth=0.8,
                    capsize=1.5, capthick=0.6, zorder=4)

    ax.set_xscale('log')
    ax.set_xticks(ALL_TPS)
    ax.set_xlim(1.5, 900)
    ax.set_xticklabels([TP_LABELS[tp] for tp in ALL_TPS],
                       fontsize=TICK_LABEL, fontfamily=FF)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title(name, fontsize=AXIS_LABEL, fontfamily=FF, fontweight='bold', pad=4)
    if col == 0:
        ax.set_ylabel('Concentration (\u00b5M)', fontsize=AXIS_LABEL, fontfamily=FF)
    if row == n_rows - 1:
        ax.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)
    style_ax(ax)

    ax.text(-0.12, 1.08, panel_labels[idx], transform=ax.transAxes, fontsize=PANEL_LABEL,
            fontweight='bold', va='top', fontfamily=FF)

# Legend
legend_handles = [
    Line2D([0], [0], color=TC['GLUCOSE'], marker='o', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.2, label='Glucose'),
    Line2D([0], [0], color=TC['XYLOSE'], marker='^', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.2, label='Xylose'),
    Line2D([0], [0], color=TC['CELLULOSE'], marker='s', markersize=5,
           markeredgecolor='black', markeredgewidth=0.3, linewidth=1.2, label='Hemicellulose'),
]
fig.legend(handles=legend_handles, fontsize=8, loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.0))

fig.suptitle('Pore water metabolite concentrations',
             fontsize=11, fontfamily=FF, fontweight='bold', y=0.97)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'{OUTPUT_STEM}.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved {OUTPUT_STEM}.pdf/.png")
