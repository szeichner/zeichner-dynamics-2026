#!/usr/bin/env python3
"""
Figure S6 — Bacterial vs fungal/eukaryotic transcript trajectories within
each k=3 program. Uses saved domain classification from
data/derived/ko_domain_classification.tsv.
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']
PROG_DISPLAY = {
    'P_EARLY_MONO': 'P1: Monomer growth\nresponse',
    'P_LATE_MONO':  'P2: Late succession &\nnecromass recycling',
    'P_HEMI':       'P3: Hemicellulose-\nassociated community',
}
PROG_COLORS = {
    'P_EARLY_MONO': '#6B7C2E',
    'P_LATE_MONO':  '#A2C060',
    'P_HEMI':       '#7B3F6E',
}

TP_MONO = [2, 8, 24, 72, 168, 336]
TP_HEMI = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

# ============================================================================
# Load data — use saved domain classification
# ============================================================================
print("Loading program assignments...")
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')

domain_df = pd.read_csv(os.path.join(DERIVED_DIR, 'ko_domain_classification.tsv'), sep='\t')
domain_df['domain'] = domain_df['domain'].replace('Ambiguous', 'Unassigned')
assign = assign.merge(domain_df[['ko_id', 'domain']], on='ko_id', how='left')
assign['domain'] = assign['domain'].fillna('Unassigned')

print("\nDomain classification summary:")
for prog in PROG_ORDER:
    sub = assign[assign.mnemonic == prog]
    print(f"\n  {prog} (n={len(sub)}):")
    for dom in ['Bacterial', 'Fungal/eukaryotic', 'Unassigned']:
        n = (sub.domain == dom).sum()
        print(f"    {dom:20s}: {n:4d} ({n/len(sub)*100:5.1f}%)")

lfc_cols_mono = [f'MONOMER_t{t}' for t in TP_MONO]
lfc_cols_hemi = [f'HEMICELLULOSE_t{t}' for t in TP_HEMI]

# ============================================================================
# Figure: 3 programs × 2 treatments
# ============================================================================
print("\nGenerating figure...")

domain_colors = {
    'Bacterial': '#2A9D8F',
    'Fungal/eukaryotic': '#E76F51',
    'Unassigned': '#999999',
}
domain_ls = {
    'Bacterial': '-',
    'Fungal/eukaryotic': '-',
    'Unassigned': ':',
}

fig, axes = plt.subplots(len(PROG_ORDER), 2, figsize=(180 * MM, len(PROG_ORDER) * 45 * MM),
                          sharex='col')
fig.subplots_adjust(hspace=0.40, wspace=0.25, left=0.14, right=0.88,
                    top=0.92, bottom=0.08)

for pi, prog in enumerate(PROG_ORDER):
    sub = assign[assign.mnemonic == prog]

    for ti, (tp_list, lfc_cols, treat_label) in enumerate([
        (TP_MONO, lfc_cols_mono, 'Monomers (glucose + xylose)'),
        (TP_HEMI, lfc_cols_hemi, 'Hemicellulose'),
    ]):
        ax = axes[pi, ti]
        xs = range(len(tp_list))

        for domain in ['Bacterial', 'Fungal/eukaryotic', 'Unassigned']:
            dsub = sub[sub.domain == domain]
            if len(dsub) < 2:
                continue
            traj = dsub[lfc_cols].mean(axis=0)
            sem = dsub[lfc_cols].std(axis=0) / np.sqrt(len(dsub))

            ax.plot(xs, traj.values, color=domain_colors[domain],
                    linewidth=1.5, linestyle=domain_ls[domain], zorder=3,
                    label=f'{domain} (n={len(dsub)})')
            ax.fill_between(xs, traj.values - sem.values, traj.values + sem.values,
                            color=domain_colors[domain], alpha=0.1, zorder=2)

        ax.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='--', zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([TP_LABELS[t] for t in tp_list], fontsize=8, fontfamily=FF)
        ax.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for _sp in ['left', 'bottom']:
            ax.spines[_sp].set_linewidth(0.6)

        if pi == 0:
            ax.set_title(treat_label, fontsize=11, fontfamily=FF, fontweight='bold',
                         color='#6B7C2E' if ti == 0 else '#7B3F6E')
        if ti == 0:
            ax.set_ylabel(PROG_DISPLAY[prog] + '\n$\\log_2$FC', fontsize=8, fontfamily=FF,
                          color=PROG_COLORS[prog])
        if pi == len(PROG_ORDER) - 1:
            ax.set_xlabel('Time', fontsize=10, fontfamily=FF)
        if pi == 0 and ti == 1:
            ax.legend(fontsize=8, loc='upper right', frameon=False, handletextpad=0.3)

# Legend at bottom
handles = [
    Line2D([0], [0], color='#2A9D8F', linewidth=1.5, label='Bacterial'),
    Line2D([0], [0], color='#E76F51', linewidth=1.5, label='Fungal/eukaryotic'),
    Line2D([0], [0], color='#999999', linewidth=1.5, linestyle=':', label='Unassigned'),
]
fig.legend(handles=handles, fontsize=8, loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.0))

# suptitle removed per user request

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS6_domain_trajectories.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS6_domain_trajectories.pdf/.png")
