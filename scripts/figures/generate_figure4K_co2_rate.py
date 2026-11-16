#!/usr/bin/env python3
"""
Figure 4 Panel K — Excess 13C-CO2 production rate over time (3 treatments).
Separate file from the metabolite panels (A-J).
Output: Figure4K_co2_rate.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
CO2_FILE = os.path.join(BASE, 'data', 'co2',
                        'sugar_experiment_CO2_mgC_withJarInfo.csv')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output', 'log_xaxis')
os.makedirs(FIGURES_DIR, exist_ok=True)

FF = 'Arial'
MM = 1 / 25.4

TC_CO2 = {'glucose': '#6B7C2E', 'xylose': '#C8960C', 'cellulose': '#7B3F6E'}
TL_CO2 = {'glucose': 'Glucose', 'xylose': 'Xylose', 'cellulose': 'Hemicellulose'}
TM_CO2 = {'glucose': 'o', 'xylose': '^', 'cellulose': 's'}

AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
MARKER_SIZE = 5
LINE_WIDTH = 1.5

R_VPDB = 0.0112372
C13_ADDED_MG = {'glucose': 5.155555556, 'xylose': 1.785119563, 'cellulose': 8.888888889}
RATE_CFG = {
    1: (1, 1), 2: (2, 2), 4: (4, 3), 8: (8, 4), 24: (24, 24),
    72: (72, 1), 168: (168, 1), 336: (336, 1), 672: (672, 1),
}

# ===========================================================================
# Load CO2 data and compute excess 13C production rates (per g DRY soil)
# ===========================================================================
print("Loading CO2 data and computing rates...")

co2 = pd.read_csv(CO2_FILE)
co2.columns = [c.strip() for c in co2.columns]
co2 = co2.loc[:, co2.columns != '']

co2['treatment'] = co2['treatment'].str.strip()
co2 = co2.rename(columns={
    '\u03b413CVPDB (\u2030)': 'delta13C', 'time point': 'time_h',
    'Converted CO2 (mg)': 'co2_mg', 'grams in jar': 'soil_wet_g',
    'jar number': 'jar_number',
})
co2['time_h'] = pd.to_numeric(co2['time_h'], errors='coerce')
co2['delta13C'] = pd.to_numeric(co2['delta13C'], errors='coerce')
co2['co2_mg'] = pd.to_numeric(co2['co2_mg'], errors='coerce')
co2['soil_wet_g'] = pd.to_numeric(co2['soil_wet_g'], errors='coerce')
co2['jar_number'] = pd.to_numeric(co2['jar_number'], errors='coerce')

# Load bench notes for per-jar gravimetric moisture (wet basis: (wet-dry)/wet)
BENCH_FILE = os.path.join(BASE, 'tables', 'output', 'TableS17_bench_notes.tsv')
bench = pd.read_csv(BENCH_FILE, sep='\t')
# sample_id in bench = jar_number in CO2; deduplicate by taking mean per sample_id
bench_dedup = bench.dropna(subset=['sample_id', 'gravimetric_moisture_fraction'])
bench_dedup = bench_dedup.groupby('sample_id')['gravimetric_moisture_fraction'].mean()
jar_moisture = bench_dedup.to_dict()

# Compute dry soil mass per jar: dry = wet * (1 - moisture_fraction_wet_basis)
# For jars without a bench note match (basal/control), use mean moisture
mean_moisture = bench['gravimetric_moisture_fraction'].mean()
print(f"  Mean gravimetric moisture (wet basis): {mean_moisture:.4f}")

co2['moisture'] = co2['jar_number'].map(jar_moisture).fillna(mean_moisture)
co2['soil_dry_g'] = co2['soil_wet_g'] * (1 - co2['moisture'])
print(f"  Dry soil mass: mean={co2['soil_dry_g'].mean():.1f}g, "
      f"range={co2['soil_dry_g'].min():.1f}-{co2['soil_dry_g'].max():.1f}g")
print(f"  (vs wet: mean={co2['soil_wet_g'].mean():.1f}g)")

co2['R_ratio'] = R_VPDB * (1 + co2['delta13C'] / 1000)
co2['F'] = co2['R_ratio'] / (1 + co2['R_ratio'])
F_baseline = co2[co2['treatment'].isin(['basal', 'control'])]['F'].mean()
print(f"  F_baseline: {F_baseline:.6f}")

co2_rates = {}
for treat in ['glucose', 'xylose', 'cellulose']:
    tdf = co2[co2['treatment'] == treat]
    c13_added = C13_ADDED_MG[treat]
    tp_data = {}
    for nom_tp, (use_t, dt_h) in RATE_CFG.items():
        sub = tdf[tdf['time_h'] == use_t]
        if len(sub) == 0:
            continue
        rates = []
        for _, row in sub.iterrows():
            excess_F = row['F'] - F_baseline
            # Normalize by DRY soil mass
            flux_per_g = (excess_F * row['co2_mg'] * 1000) / dt_h / row['soil_dry_g']
            added_per_g = c13_added * 1000 / row['soil_dry_g']
            rates.append(flux_per_g / added_per_g)
        tp_data[nom_tp] = rates
    co2_rates[treat] = tp_data
    print(f"  {treat}: {sorted(tp_data.keys())}")

# ===========================================================================
# Generate figure
# ===========================================================================
print("Generating Figure 4K...")

fig, ax = plt.subplots(figsize=(180 * MM, 100 * MM))
fig.subplots_adjust(left=0.14, right=0.95, top=0.94, bottom=0.18)

for treat in ['glucose', 'xylose', 'cellulose']:
    tp_data = co2_rates.get(treat, {})
    if not tp_data:
        continue
    tps = sorted(tp_data.keys())
    means = [np.mean(tp_data[tp]) for tp in tps]
    sems = [np.std(tp_data[tp], ddof=1) / np.sqrt(len(tp_data[tp]))
            if len(tp_data[tp]) > 1 else 0 for tp in tps]
    x = np.array(tps, dtype=float)

    ax.plot(x, means, color=TC_CO2[treat], linewidth=LINE_WIDTH, zorder=3)
    ax.fill_between(x, np.array(means) - np.array(sems),
                    np.array(means) + np.array(sems),
                    color=TC_CO2[treat], alpha=0.20, zorder=2)
    ax.errorbar(x, means, yerr=sems, fmt=TM_CO2[treat],
                color=TC_CO2[treat], markersize=MARKER_SIZE,
                markeredgecolor='black', markeredgewidth=0.3,
                elinewidth=1.0, capsize=2, capthick=0.6, zorder=4)

ax.set_xscale('log')
ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)
ax.set_ylabel('Excess $^{13}$C-CO$_2$ rate\n(h$^{-1}$ g$^{-1}$ dry soil)',
              fontsize=AXIS_LABEL, fontfamily=FF)
ax.set_xlim(0.8, 900)
tick_vals = [1, 2, 4, 8, 24, 72, 168, 336, 672]
tick_labels = ['1h', '2h', '4h', '8h', '1d', '3d', '7d', '14d', '28d']
ax.set_xticks(tick_vals)
ax.set_xticklabels(tick_labels, fontsize=TICK_LABEL, fontfamily=FF)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_linewidth(0.7)
ax.tick_params(axis='both', which='major', direction='out',
               length=3, width=0.8, labelsize=TICK_LABEL)
ax.text(-0.12, 1.08, 'B', transform=ax.transAxes,
        fontsize=PANEL_LABEL, fontweight='bold', va='top', fontfamily=FF)

# Legend
handles = [
    Line2D([0], [0], marker=TM_CO2[t], color=TC_CO2[t], markerfacecolor=TC_CO2[t],
           markeredgecolor='black', markeredgewidth=0.3, markersize=MARKER_SIZE + 1,
           linewidth=LINE_WIDTH, label=TL_CO2[t])
    for t in ['glucose', 'xylose', 'cellulose']
]
ax.legend(handles=handles, fontsize=LEGEND_SIZE, loc='upper right',
          frameon=False, handletextpad=0.4)

# ===========================================================================
# Save
# ===========================================================================
for ext in ['pdf', 'png']:
    out = os.path.join(FIGURES_DIR, f'Figure4B_co2_rate_logx.{ext}')
    fig.savefig(out, format=ext, dpi=600, bbox_inches='tight')
    print(f"  Saved {out}")
plt.close(fig)
print("Done.")
