#!/usr/bin/env python3
"""
Figure S — Combined CAZyme investment (5 panels: Cellulases, Xylanases,
Amylases, LPMO/accessory, Chitinases). Pooled Monomers vs Hemicellulose.
Filled markers = >=20% significant; open = n.s.
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
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')

TC_POOL = {'Monomers': '#6B7C2E', 'Hemicellulose': '#7B3F6E'}
TM_POOL = {'Monomers': 'o', 'Hemicellulose': 's'}
POOL_ORDER = ['Monomers', 'Hemicellulose']
FF = 'Arial'
MM = 1 / 25.4

TP_HOURS = [2, 8, 24, 72, 168, 336]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d'}
# Use actual hours for log x-axis

# Load data
agg = pd.read_csv(os.path.join(TABLES_DIR, 'TableS6_CAZy_broad_group_DESeq2.tsv'), sep='\t')


def pool_monomers(grp):
    """Pool glucose + xylose into 'Monomers'; keep hemicellulose as-is."""
    rows = []
    for tp in TP_HOURS:
        g = grp[(grp.treatment == 'GLUCOSE') & (grp.timepoint_hours == tp)]
        x = grp[(grp.treatment == 'XYLOSE') & (grp.timepoint_hours == tp)]
        h = grp[(grp.treatment == 'CELLULOSE') & (grp.timepoint_hours == tp)]

        if not g.empty and not x.empty:
            g_mean, x_mean = g.mean_log2FC.values[0], x.mean_log2FC.values[0]
            g_sd, x_sd = g.SD_log2FC.values[0], x.SD_log2FC.values[0]
            g_n, x_n = g.n_families_detected.values[0], x.n_families_detected.values[0]
            pooled_mean = (g_mean + x_mean) / 2
            pooled_sd = np.sqrt((g_sd**2 + x_sd**2) / 2)
            pooled_n = (g_n + x_n) / 2
            pooled_sem = pooled_sd / np.sqrt(pooled_n) if pooled_n > 0 else 0
            g_sig = g.get('frac_significant', g.get('group_significant', pd.Series(['no']))).values[0]
            x_sig = x.get('frac_significant', x.get('group_significant', pd.Series(['no']))).values[0]
            # Handle both numeric and string significance
            if isinstance(g_sig, str):
                pooled_sig = 1.0 if (g_sig == 'yes' or x_sig == 'yes') else 0.0
            else:
                pooled_sig = max(float(g_sig), float(x_sig))
            rows.append({'treatment': 'Monomers', 'timepoint_hours': tp,
                         'mean_log2FC': pooled_mean, 'sem_log2FC': pooled_sem,
                         'frac_significant': pooled_sig, 'n_families': pooled_n})

        if not h.empty:
            h_sd = h.SD_log2FC.values[0]
            h_n = h.n_families_detected.values[0]
            h_sig = h.get('frac_significant', h.get('group_significant', pd.Series(['no']))).values[0]
            if isinstance(h_sig, str):
                h_sig = 1.0 if h_sig == 'yes' else 0.0
            rows.append({'treatment': 'Hemicellulose', 'timepoint_hours': tp,
                         'mean_log2FC': h.mean_log2FC.values[0],
                         'sem_log2FC': h_sd / np.sqrt(h_n) if h_n > 0 else 0,
                         'frac_significant': float(h_sig), 'n_families': h_n})
    return pd.DataFrame(rows)


# Define panels: (broad_group_key, display_name, n_families, title_color)
# LPMO/accessory = everything except Cellulases, Xylanases, Amylases, Chitinases
named_groups = ['Cellulases', 'Xylanases', 'Amylases', 'Chitinases']
PANELS = [
    ('Cellulases', 'Cellulases', '#7B3F6E'),
    ('Xylanases', 'Xylanases', '#7B3F6E'),
    ('Amylases', 'Amylases', '#6B7C2E'),
    ('LPMO_accessory', 'LPMO / accessory', '#6B7C2E'),
    ('Chitinases', 'Chitinases', '#7B3F6E'),
]

# Figure: 2 rows × 3 columns (5 panels, bottom-right empty or legend)
fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 110 * MM))
fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.97,
                    top=0.92, bottom=0.08)

panel_letters = ['A', 'B', 'C', 'D', 'E']
panel_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for pi, ((row_i, col_i), (group_key, display_name, title_color)) in enumerate(
        zip(panel_positions, PANELS)):
    ax = axes[row_i, col_i]

    if group_key == 'LPMO_accessory':
        grp = agg[~agg.broad_group.isin(named_groups)]
    else:
        grp = agg[agg.broad_group == group_key]

    pooled = pool_monomers(grp)
    n_fam = int(pooled.n_families.max()) if not pooled.empty else 0

    for treat in POOL_ORDER:
        td = pooled[pooled.treatment == treat].sort_values('timepoint_hours')
        td = td[td.timepoint_hours.isin(TP_HOURS)].copy()
        td['x'] = td['timepoint_hours']
        if td.empty:
            continue
        c = TC_POOL[treat]
        m = TM_POOL[treat]

        ax.plot(td.x, td.mean_log2FC, color=c, linewidth=1.5, zorder=2)
        ax.fill_between(td.x, td.mean_log2FC - td.sem_log2FC.fillna(0),
                        td.mean_log2FC + td.sem_log2FC.fillna(0),
                        color=c, alpha=0.12)

        sig = td[td.frac_significant >= 0.20]
        nsig = td[td.frac_significant < 0.20]

        if not sig.empty:
            ax.errorbar(sig.x, sig.mean_log2FC, yerr=sig.sem_log2FC,
                        fmt=m, color=c, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.3,
                        elinewidth=1.0, capsize=1.5, capthick=1.0, zorder=3)
        if not nsig.empty:
            ax.errorbar(nsig.x, nsig.mean_log2FC, yerr=nsig.sem_log2FC,
                        fmt=m, color=c, markersize=5,
                        markerfacecolor='white', markeredgecolor=c,
                        markeredgewidth=1.0, elinewidth=1.0, capsize=1.5,
                        capthick=1.0, zorder=3)

    ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xticks(TP_HOURS)
    ax.set_xlim(1.5, 900)
    ax.set_xticklabels([TP_LABELS[t] for t in TP_HOURS], fontsize=8, fontfamily=FF)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title(f'{display_name} ({n_fam} families)', fontsize=11, fontfamily=FF,
                 fontweight='bold', color=title_color, pad=4)
    if col_i == 0:
        ax.set_ylabel(r'Mean log$_2$FC vs control', fontsize=10, fontfamily=FF)
    if row_i == 1:
        ax.set_xlabel(r'Time (log$_{10}$ scale)', fontsize=10, fontfamily=FF)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for _sp in ['left', 'bottom']:
        ax.spines[_sp].set_linewidth(0.6)
    ax.tick_params(labelsize=8, direction='out', length=2.5, width=0.6)
    ax.text(-0.15, 1.05, panel_letters[pi], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', fontfamily=FF)

# Bottom-right panel: legend
ax_leg = axes[1, 2]
ax_leg.axis('off')
handles = [
    Line2D([0], [0], marker='o', color='#6B7C2E', markerfacecolor='#6B7C2E',
           markeredgecolor='black', markeredgewidth=0.3, markersize=5,
           linewidth=1.5, label='Monomers (pooled)'),
    Line2D([0], [0], marker='s', color='#7B3F6E', markerfacecolor='#7B3F6E',
           markeredgecolor='black', markeredgewidth=0.3, markersize=5,
           linewidth=1.5, label='Hemicellulose'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
           markersize=5, linewidth=0, label='Filled = sig. (≥20%)'),
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
           markeredgecolor='gray', markeredgewidth=1.0,
           markersize=5, linewidth=0, label='Open = n.s.'),
]
ax_leg.legend(handles=handles, fontsize=8, loc='center', frameon=False,
              handletextpad=0.5, labelspacing=1.2)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS13_CAZyme_combined.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS13_CAZyme_combined.pdf/.png")
