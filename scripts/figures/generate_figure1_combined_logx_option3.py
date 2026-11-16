#!/usr/bin/env python3
"""
Figure 1 — Option 3 variant: weighted mean EAF line + shaded >1% bin fill.

The black line shows total community weighted mean EAF. The shaded fill
underneath shows the contribution from the >1% abundance bin alone.
The gap between the fill and the line represents contributions from
moderate- and low-abundance taxa.

Output: figures/output/log_xaxis/Figure1_qsip_combined_logx_option3.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
QSIP_DIR = os.path.join(BASE, 'data', 'qSIP')
TABLE_DIR = os.path.join(BASE, 'tables', 'output')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output', 'log_xaxis')
os.makedirs(FIGURES_DIR, exist_ok=True)
FF = 'Arial'
MM = 1 / 25.4

TC = {'Glucose': '#6B7C2E', 'Xylose': '#C8960C', 'Hemicellulose': '#7B3F6E'}
TM = {'Glucose': 'o', 'Xylose': '^', 'Hemicellulose': 's'}

AXIS_LABEL = 12
TICK_LABEL = 11
PANEL_LABEL = 14
LEGEND_SIZE = 10
MARKER_SIZE = 6
LINE_WIDTH = 1.5

ALL_TPS = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.8, labelsize=TICK_LABEL)

# ============================================================================
# Load data
# ============================================================================
print("Loading qSIP data...")
qsip = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_community_incorporation_efficiency.tsv'), sep='\t')

print("Loading abundance-bin partition data...")
bin_df = pd.read_csv(os.path.join(TABLE_DIR, 'abundance_bin_EAF_partition.tsv'), sep='\t')

print("Loading qSIP phylum data...")
phylum_df = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_by_phylum.tsv'), sep='\t')

print("Loading per-ASV EAF data (for fungal class-level panel)...")
asv_df = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_EAF.tsv'), sep='\t')

# ============================================================================
# Figure layout (same as canonical Figure 1)
# ============================================================================
fig = plt.figure(figsize=(240 * MM, 420 * MM))
gs = fig.add_gridspec(5, 2, hspace=0.65, wspace=0.50,
                      left=0.13, right=0.96, top=0.95, bottom=0.05,
                      height_ratios=[1.0, 1.0, 1.0, 0.18, 0.60],
                      width_ratios=[1.0, 1.0])

treat_order = ['Glucose', 'Xylose', 'Hemicellulose']
marker_order = ['16S', 'ITS']
marker_titles = {'16S': 'Bacteria and Archaea', 'ITS': 'Fungi'}

# Compute y-axis limits
ylims = {}
for marker in marker_order:
    vals = []
    for treat in treat_order:
        sub = qsip[(qsip['marker'] == marker) & (qsip['treatment'] == treat)]
        vals.extend(sub['weighted_mean_EAF'].values)
    ylims[marker] = (min(min(vals), -0.02) - 0.01, max(vals) * 1.15 + 0.01)

panel_grid = [['A', 'B'], ['C', 'D'], ['E', 'F']]

# ============================================================================
# Panels A–F: Weighted EAF line + >1% bin shaded fill
# ============================================================================
FILL_ALPHA = 0.35

for row_idx, treat in enumerate(treat_order):
    for col_idx, marker in enumerate(marker_order):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Community weighted mean EAF (black line)
        sub_comm = qsip[(qsip['marker'] == marker) & (qsip['treatment'] == treat)].sort_values('timepoint')
        tps_comm = sub_comm['timepoint'].values
        hours_comm = np.array([float(t) for t in tps_comm if int(t) in ALL_TPS])
        w_vals = np.array([sub_comm[sub_comm['timepoint'] == t]['weighted_mean_EAF'].values[0]
                           for t in tps_comm if int(t) in ALL_TPS])

        # >1% bin contribution
        sub_bin = bin_df[(bin_df['marker'] == marker) & (bin_df['treatment'] == treat) &
                         (bin_df['bin_label'] == '>1%')].sort_values('timepoint')
        hours_bin = np.array([float(t) for t in sub_bin['timepoint'].values if int(t) in ALL_TPS])
        bin_vals = np.array([sub_bin[sub_bin['timepoint'] == t]['bin_weighted_EAF'].values[0]
                             for t in sub_bin['timepoint'].values if int(t) in ALL_TPS])

        # Clip bin vals to [0, weighted mean] for clean fill
        bin_vals_clipped = np.clip(bin_vals, 0, None)

        # Shaded fill: >1% bin contribution (from 0 up to bin value)
        ax.fill_between(hours_bin, 0, bin_vals_clipped, color='black', alpha=FILL_ALPHA,
                        zorder=1, label='>1% abundance bin' if (row_idx == 0 and col_idx == 0) else None)

        # Black line: total community weighted EAF
        ax.plot(hours_comm, w_vals, color='black', marker=TM[treat],
                markersize=MARKER_SIZE, markerfacecolor='black',
                markeredgecolor='black', markeredgewidth=0.3,
                linewidth=LINE_WIDTH, linestyle='-', zorder=3)

        # Zero line
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3, zorder=0)

        ax.set_xscale('log')
        ax.set_xticks(ALL_TPS)
        ax.set_xticklabels([TP_LABELS[t] for t in ALL_TPS], fontsize=TICK_LABEL, fontfamily=FF)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(1.5, 900)
        ax.set_ylim(*ylims[marker])

        if col_idx == 0:
            ax.set_ylabel('Weighted mean EAF', fontsize=AXIS_LABEL, fontfamily=FF)
        if row_idx == 0:
            ax.set_title(marker_titles[marker], fontsize=13, fontweight='bold',
                         fontfamily=FF, color='black')
        if col_idx == 1:
            ax.annotate(treat, xy=(1.03, 0.5), xycoords='axes fraction',
                        fontsize=AXIS_LABEL, fontfamily=FF, fontweight='bold',
                        color=TC[treat], va='center', rotation=-90)
        if row_idx == 2:
            ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)

        style_ax(ax)
        ax.text(-0.15, 1.08, panel_grid[row_idx][col_idx],
                transform=ax.transAxes, fontsize=PANEL_LABEL,
                fontweight='bold', va='top', fontfamily=FF)

# Shared legend
legend_handles = [
    Line2D([0], [0], color='black', linewidth=LINE_WIDTH, linestyle='-',
           marker='o', markersize=5, markerfacecolor='black',
           markeredgecolor='black', markeredgewidth=0.3,
           label='Community weighted EAF'),
    Patch(facecolor='black', alpha=FILL_ALPHA, edgecolor='none',
          label='>1% abundance bin contribution'),
]
fig.legend(handles=legend_handles, fontsize=LEGEND_SIZE, loc='upper center',
           ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.0),
           handletextpad=0.4, columnspacing=2.0)

# ============================================================================
# Heatmaps — identical to canonical Figure 1
# ============================================================================
def plot_phylum_heatmap(ax, marker, panel_label, min_asv_total=100, vmax=None,
                        highlight_phyla=None):
    phy = phylum_df[phylum_df['marker'] == marker].copy()
    if highlight_phyla is not None:
        phy = phy[phy['Phylum'].isin(highlight_phyla)]
        major = highlight_phyla
    else:
        phy_totals = phy.groupby('Phylum')['n_ASVs'].sum()
        threshold = 20 if marker == 'ITS' else min_asv_total
        major = phy_totals[phy_totals >= threshold].index.tolist()
    phy = phy[phy['Phylum'].isin(major)]
    phylum_mean = phy.groupby('Phylum')['mean_EAF'].mean().sort_values(ascending=True)
    phyla_ordered = phylum_mean.index.tolist()
    treats = ['Glucose', 'Xylose', 'Hemicellulose']
    treat_tps = {t: sorted(phy[phy['treatment_label'] == t]['timepoint'].unique()) for t in treats}
    all_cols = []
    for t in treats:
        for tp in treat_tps[t]:
            all_cols.append((tp, t))
    total_asvs = {}
    for tp, treat in all_cols:
        mask = (phy['treatment_label'] == treat) & (phy['timepoint'] == tp)
        total_asvs[(tp, treat)] = phy.loc[mask, 'n_ASVs'].sum()
    n_cols, n_rows = len(all_cols), len(phyla_ordered)
    mat = np.full((n_rows, n_cols), np.nan)
    for ci, (tp, treat) in enumerate(all_cols):
        tot = total_asvs[(tp, treat)]
        for ri, pname in enumerate(phyla_ordered):
            row = phy[(phy['Phylum'] == pname) & (phy['treatment_label'] == treat) &
                      (phy['timepoint'] == tp)]
            if len(row) > 0:
                eaf = row['mean_EAF'].iloc[0]
                n = row['n_ASVs'].iloc[0]
                mat[ri, ci] = eaf * n / tot if tot > 0 else 0.0
    if vmax is None:
        vmax = max(np.nanmax(mat) * 1.1, 0.005)
    norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax)
    im = ax.imshow(mat, aspect='auto', cmap='Greys', norm=norm, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    ylabels = [p.replace('Candidatus_', 'Ca. ').replace('p__', '') for p in phyla_ordered]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ylabels, fontsize=9.5, fontfamily=FF)
    xlabels = [str(int(tp)) for tp, _ in all_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, fontsize=9, fontfamily=FF, rotation=45, ha='right')
    col_offset = 0
    for ti, treat in enumerate(treats):
        n_tp = len(treat_tps[treat])
        center = col_offset + (n_tp - 1) / 2
        ax.text(center, -1.8, treat, ha='center', va='bottom', fontsize=AXIS_LABEL - 1,
                fontfamily=FF, fontweight='bold', color=TC[treat])
        if ti < len(treats) - 1:
            ax.axvline(col_offset + n_tp - 0.5, color='black', linewidth=1.0, zorder=5)
        col_offset += n_tp
    ax.set_xlabel('Time (h)', fontsize=AXIS_LABEL - 1, fontfamily=FF)
    ax.text(-0.08, 1.10, panel_label, transform=ax.transAxes, fontsize=PANEL_LABEL,
            fontweight='bold', va='top', fontfamily=FF)
    return im

ax_d = fig.add_subplot(gs[3:5, 0])
im_d = plot_phylum_heatmap(ax_d, '16S', 'G', min_asv_total=100)

ax_e = fig.add_subplot(gs[3, 1])
im_e = plot_phylum_heatmap(ax_e, 'ITS', 'H', min_asv_total=20,
                            highlight_phyla=['p__Ascomycota', 'p__Basidiomycota'])

# Panel I: Fungal class-level heatmap
print("Building class-level fungal heatmap...")
its_asv = asv_df[asv_df['marker'] == 'ITS'].copy()
its_asv['Class_clean'] = its_asv['Class'].fillna('unclassified').str.replace('c__', '', regex=False)
its_asv['Phylum_clean'] = its_asv['Phylum'].str.replace('p__', '', regex=False)
its_asv = its_asv[its_asv['Phylum_clean'].isin(['Ascomycota', 'Basidiomycota'])].copy()
its_asv.loc[its_asv['Class_clean'].str.contains('Incertae_sedis', na=False),
            'Class_clean'] = its_asv.loc[its_asv['Class_clean'].str.contains('Incertae_sedis', na=False),
                                         'Phylum_clean'] + ' (inc. sed.)'
its_asv['display_label'] = its_asv['Phylum_clean'] + ': ' + its_asv['Class_clean']
class_summary = its_asv.groupby(['treatment_label', 'timepoint', 'display_label']).agg(
    mean_EAF=('EAF', 'mean'), n_ASVs=('EAF', 'count')
).reset_index()
class_totals = class_summary.groupby('display_label')['n_ASVs'].sum()
major_classes = class_totals[class_totals >= 50].index.tolist()
class_summary = class_summary[class_summary['display_label'].isin(major_classes)]

ax_f = fig.add_subplot(gs[4, 1])
treats = ['Glucose', 'Xylose', 'Hemicellulose']
treat_tps_f = {t: sorted(class_summary[class_summary['treatment_label'] == t]['timepoint'].unique())
               for t in treats}
all_cols_f = []
for t in treats:
    for tp in treat_tps_f[t]:
        all_cols_f.append((tp, t))
class_mean = class_summary.groupby('display_label')['mean_EAF'].mean().sort_values(ascending=True)
classes_ordered = class_mean.index.tolist()
total_asvs_f = {}
for tp, treat in all_cols_f:
    mask = (class_summary['treatment_label'] == treat) & (class_summary['timepoint'] == tp)
    total_asvs_f[(tp, treat)] = class_summary.loc[mask, 'n_ASVs'].sum()
n_cols_f, n_rows_f = len(all_cols_f), len(classes_ordered)
mat_f = np.full((n_rows_f, n_cols_f), np.nan)
for ci, (tp, treat) in enumerate(all_cols_f):
    tot = total_asvs_f[(tp, treat)]
    for ri, cname in enumerate(classes_ordered):
        row = class_summary[(class_summary['display_label'] == cname) &
                            (class_summary['treatment_label'] == treat) &
                            (class_summary['timepoint'] == tp)]
        if len(row) > 0:
            eaf = row['mean_EAF'].iloc[0]
            n = row['n_ASVs'].iloc[0]
            mat_f[ri, ci] = eaf * n / tot if tot > 0 else 0.0
vmax_f = max(np.nanmax(mat_f) * 1.1, 0.005)
norm_f = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax_f)
im_f = ax_f.imshow(mat_f, aspect='auto', cmap='Greys', norm=norm_f, interpolation='nearest')
ax_f.set_xticks(np.arange(-0.5, n_cols_f, 1), minor=True)
ax_f.set_yticks(np.arange(-0.5, n_rows_f, 1), minor=True)
ax_f.grid(which='minor', color='black', linewidth=0.5)
ax_f.tick_params(which='minor', length=0)
ylabels_f = []
for c in classes_ordered:
    if ': ' in c:
        phylum, cls = c.split(': ', 1)
        ylabels_f.append(f"{phylum[:4]}. {cls}")
    else:
        ylabels_f.append(c)
ax_f.set_yticks(range(n_rows_f))
ax_f.set_yticklabels(ylabels_f, fontsize=9, fontfamily=FF)
xlabels_f = [str(int(tp)) for tp, _ in all_cols_f]
ax_f.set_xticks(range(n_cols_f))
ax_f.set_xticklabels(xlabels_f, fontsize=9, fontfamily=FF, rotation=45, ha='right')
col_offset = 0
for ti, treat in enumerate(treats):
    n_tp = len(treat_tps_f[treat])
    center = col_offset + (n_tp - 1) / 2
    ax_f.text(center, -1.8, treat, ha='center', va='bottom', fontsize=AXIS_LABEL - 1,
              fontfamily=FF, fontweight='bold', color=TC[treat])
    if ti < len(treats) - 1:
        ax_f.axvline(col_offset + n_tp - 0.5, color='black', linewidth=1.0, zorder=5)
    col_offset += n_tp
ax_f.set_xlabel('Time (h)', fontsize=AXIS_LABEL - 1, fontfamily=FF)
ax_f.text(-0.08, 1.10, 'I', transform=ax_f.transAxes, fontsize=PANEL_LABEL,
          fontweight='bold', va='top', fontfamily=FF)

# Colorbars
cax_d = fig.add_axes([0.12, 0.015, 0.35, 0.008])
cb_d = plt.colorbar(im_d, cax=cax_d, orientation='horizontal')
cb_d.set_label('Bacteria weighted EAF (phylum)', fontsize=9, fontfamily=FF)
cb_d.ax.tick_params(labelsize=8)

pos_e = ax_e.get_position()
cax_e = fig.add_axes([pos_e.x0, pos_e.y0 - 0.030, pos_e.width * 0.6, 0.008])
cb_e = plt.colorbar(im_e, cax=cax_e, orientation='horizontal')
cb_e.ax.tick_params(labelsize=8)
cb_e.set_label('Fungi weighted EAF (phylum)', fontsize=8, fontfamily=FF)

cax_f = fig.add_axes([0.58, 0.015, 0.35, 0.008])
cb_f = plt.colorbar(im_f, cax=cax_f, orientation='horizontal')
cb_f.set_label('Fungi weighted EAF (class)', fontsize=9, fontfamily=FF)
cb_f.ax.tick_params(labelsize=8)

# Save
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'Figure1_qsip_combined_logx_option3.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved Figure1_qsip_combined_logx_option3.pdf/.png")
