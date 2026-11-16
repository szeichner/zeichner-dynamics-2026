#!/usr/bin/env python3
"""
Figure 1 — qSIP growth physiology — LOG X-AXIS variant.

Same as generate_figure1_combined.py but line plots (Panels A–F) use
a true log-scale x-axis with hours as the coordinate.
Heatmaps remain categorical (grid layout doesn't support log axes).

Reviewer fixes applied:
- X-axis label: "Time (log10 scale)" instead of "Time (h)"
- Panel H: horizontal colorbar, x-tick labels visible
- Panels G, H, I: "Time (h)" x-axis label below tick numbers

Output: figures/output/log_xaxis/Figure1_qsip_combined_logx.pdf/.png/.tiff
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
QSIP_DIR = os.path.join(BASE, 'data', 'qSIP')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output', 'log_xaxis')
os.makedirs(FIGURES_DIR, exist_ok=True)
FF = 'Arial'
MM = 1 / 25.4

# ---------- Shared palette ----------
TC = {'Glucose': '#6B7C2E', 'Xylose': '#C8960C', 'Hemicellulose': '#7B3F6E'}
TM = {'Glucose': 'o', 'Xylose': '^', 'Hemicellulose': 's'}
DC = {'16S': '#2A9D8F', 'ITS': '#E76F51'}

AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
MARKER_SIZE = 5
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
QSIP_FILE = os.path.join(QSIP_DIR, 'output/tables/qsip_community_incorporation_efficiency.tsv')
qsip = pd.read_csv(QSIP_FILE, sep='\t')

# Load ASV-level data for computing SEM via bootstrap
print("Loading ASV-level EAF data for error bars...")
asv_eaf = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_EAF.tsv'), sep='\t')

# Compute bootstrap SEM for unweighted mean EAF per marker/treatment/timepoint
N_BOOT = 2000
np.random.seed(42)
eaf_sem = {}
for _, row in qsip.iterrows():
    key = (row['marker'], row['treatment'], int(row['timepoint']))
    sub = asv_eaf[(asv_eaf['marker'] == row['marker']) &
                  (asv_eaf['treatment_label'] == row['treatment']) &
                  (asv_eaf['timepoint'] == int(row['timepoint']))]
    if len(sub) < 3:
        eaf_sem[key] = (0, 0)
        continue
    vals = sub['EAF'].values
    abundances = sub['M'].values
    boot_uw = np.empty(N_BOOT)
    boot_w = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.randint(0, len(vals), size=len(vals))
        boot_uw[b] = vals[idx].mean()
        w = abundances[idx]
        if w.sum() > 0:
            boot_w[b] = np.average(vals[idx], weights=w)
        else:
            boot_w[b] = vals[idx].mean()
    eaf_sem[key] = (np.std(boot_w, ddof=1), np.std(boot_uw, ddof=1))
print(f"  Computed bootstrap SEM for {len(eaf_sem)} marker/treatment/timepoint combinations")

print("Loading qSIP phylum data...")
phylum_df = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_by_phylum.tsv'), sep='\t')

print("Loading per-ASV EAF data (for fungal class-level panel)...")
asv_df = pd.read_csv(os.path.join(QSIP_DIR, 'output/tables/qsip_per_treatment_timepoint_EAF.tsv'), sep='\t')

# ============================================================================
# Figure layout
# ============================================================================
n_16s = len(phylum_df[phylum_df['marker'] == '16S'].groupby('Phylum')['n_ASVs'].sum().pipe(
    lambda s: s[s >= 100]))
n_its = len(phylum_df[phylum_df['marker'] == 'ITS'].groupby('Phylum')['n_ASVs'].sum().pipe(
    lambda s: s[s >= 20]))
print(f"  16S: {n_16s} phyla, ITS: {n_its} phyla")

FIG_W = 175 * MM   # double-column width (under 180mm accounting for bbox_inches padding)
FIG_H = 235 * MM   # taller to add vertical gap between line plots and heatmaps
fig = plt.figure(figsize=(FIG_W, FIG_H))

# Layout: top section uses gridspec for A-F line plots
# Bottom section uses manual fig.add_axes() for precise heatmap placement
# G (bacteria phylum) under bacteria column (left)
# H (fungal phylum) + I (fungal class) under fungi column (right)

# Gridspec only for A-F line plots (top ~62% of figure)
gs_top = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.38,
                          left=0.13, right=0.93, top=0.96, bottom=0.45)

# ============================================================================
# Panels A–F: Per-treatment line plots — LOG X-AXIS (weighted + unweighted)
# ============================================================================
treat_order = ['Glucose', 'Xylose', 'Hemicellulose']
marker_order = ['16S', 'ITS']
marker_titles = {'16S': 'Bacteria and Archaea', 'ITS': 'Fungi'}

ylims = {}
for marker in marker_order:
    vals_lo, vals_hi = [], []
    for treat in treat_order:
        sub = qsip[(qsip['marker'] == marker) & (qsip['treatment'] == treat)]
        for _, r in sub.iterrows():
            key = (r['marker'], r['treatment'], int(r['timepoint']))
            w_sem, u_sem = eaf_sem.get(key, (0, 0))
            vals_lo.append(r['weighted_mean_EAF'] - 1.96 * w_sem)
            vals_lo.append(r['unweighted_mean_EAF'] - 1.96 * u_sem)
            vals_hi.append(r['weighted_mean_EAF'] + 1.96 * w_sem)
            vals_hi.append(r['unweighted_mean_EAF'] + 1.96 * u_sem)
    ylims[marker] = (min(min(vals_lo), -0.02) - 0.01, max(vals_hi) + 0.02)

panel_grid = [['A', 'B'], ['C', 'D'], ['E', 'F']]

for row_idx, treat in enumerate(treat_order):
    for col_idx, marker in enumerate(marker_order):
        ax = fig.add_subplot(gs_top[row_idx, col_idx])

        sub = qsip[(qsip['marker'] == marker) & (qsip['treatment'] == treat)].sort_values('timepoint')
        tps = sub['timepoint'].values
        weighted = sub['weighted_mean_EAF'].values
        unweighted = sub['unweighted_mean_EAF'].values

        hours = np.array([float(t) for t in tps if int(t) in ALL_TPS])
        w_vals = np.array([weighted[i] for i, t in enumerate(tps) if int(t) in ALL_TPS])
        u_vals = np.array([unweighted[i] for i, t in enumerate(tps) if int(t) in ALL_TPS])

        w_sem = np.array([eaf_sem.get((marker, treat, int(t)), (0, 0))[0]
                          for t in tps if int(t) in ALL_TPS])
        u_sem = np.array([eaf_sem.get((marker, treat, int(t)), (0, 0))[1]
                          for t in tps if int(t) in ALL_TPS])
        w_ci95 = 1.96 * w_sem
        u_ci95 = 1.96 * u_sem

        # Weighted: black solid, filled markers
        ax.errorbar(hours, w_vals, yerr=w_ci95, color='black', marker=TM[treat],
                    markersize=MARKER_SIZE, markerfacecolor='black',
                    markeredgecolor='black', markeredgewidth=0.3,
                    linewidth=LINE_WIDTH, linestyle='-',
                    elinewidth=0.8, capsize=2.5, capthick=0.7, zorder=3)
        # Unweighted: black dotted, open markers
        ax.errorbar(hours, u_vals, yerr=u_ci95, color='black', marker=TM[treat],
                    markersize=MARKER_SIZE, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=0.8,
                    linewidth=LINE_WIDTH * 0.8, linestyle=':',
                    elinewidth=0.8, capsize=2.5, capthick=0.7, zorder=2)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3, zorder=1)

        ax.set_xscale('log')
        ax.set_xticks(ALL_TPS)
        ax.set_xticklabels([TP_LABELS[t] for t in ALL_TPS], fontsize=TICK_LABEL, fontfamily=FF)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(1.5, 900)
        ax.set_ylim(*ylims[marker])

        if col_idx == 0:
            ax.set_ylabel('Mean EAF', fontsize=AXIS_LABEL, fontfamily=FF)
        if row_idx == 0:
            ax.set_title(marker_titles[marker], fontsize=AXIS_LABEL, fontweight='bold',
                         fontfamily=FF, color='black')
        if col_idx == 1:
            ax.annotate(treat, xy=(1.03, 0.5), xycoords='axes fraction',
                        fontsize=AXIS_LABEL, fontfamily=FF, fontweight='bold',
                        color=TC[treat], va='center', rotation=-90)
        # All panels get x-axis label for independent readability
        ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)

        style_ax(ax)
        ax.text(-0.18, 1.10, panel_grid[row_idx][col_idx],
                transform=ax.transAxes, fontsize=PANEL_LABEL,
                fontweight='bold', va='top', fontfamily=FF)

# Shared legend at top
legend_handles = [
    Line2D([0], [0], color='black', linewidth=LINE_WIDTH, linestyle='-',
           marker='o', markersize=5, markerfacecolor='black',
           markeredgecolor='black', markeredgewidth=0.3,
           label='Weighted EAF (\u00b195% CI)'),
    Line2D([0], [0], color='black', linewidth=LINE_WIDTH * 0.8, linestyle=':',
           marker='o', markersize=5, markerfacecolor='white',
           markeredgecolor='black', markeredgewidth=0.8,
           label='Unweighted EAF (\u00b195% CI)'),
]
fig.legend(handles=legend_handles, fontsize=LEGEND_SIZE, loc='upper center',
           ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.0),
           handletextpad=0.4, columnspacing=2.0)

# ============================================================================
# Heatmaps — unchanged (categorical grid)
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
    ax.set_yticklabels(ylabels, fontsize=9, fontfamily=FF)

    xlabels = [TP_LABELS.get(int(tp), str(int(tp))) for tp, _ in all_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, fontsize=7, fontfamily=FF, rotation=60, ha='right')

    col_offset = 0
    for ti, treat in enumerate(treats):
        n_tp = len(treat_tps[treat])
        center = col_offset + (n_tp - 1) / 2
        # Place treatment labels just above heatmap top edge (y=-1.0 in data coords)
        ax.text(center, -1.0, treat, ha='center', va='bottom', fontsize=9,
                fontfamily=FF, fontweight='bold', color=TC[treat],
                clip_on=False)
        if ti < len(treats) - 1:
            ax.axvline(col_offset + n_tp - 0.5, color='black', linewidth=1.0, zorder=5)
        col_offset += n_tp

    ax.set_xlabel('')  # x-axis label removed; hour numbers on ticks are self-explanatory
    ax.text(-0.10, 1.14, panel_label, transform=ax.transAxes, fontsize=PANEL_LABEL,
            fontweight='bold', va='top', fontfamily=FF)
    return im

# ---- Manual axes for heatmaps (bottom section) ----
# Coordinates: [left, bottom, width, height] in figure fraction
# Left column (bacteria): under panels A,C,E
# Right column (fungi): under panels B,D,F
# Bottom section spans y from ~0.04 to ~0.37

hm_left = 0.10       # left edge of G
hm_mid = 0.42        # right edge of G (narrower to prevent overlap with I labels)
hm_right_x = 0.58    # left edge of H,I (room for y-tick labels)
hm_right_end = 0.95  # right edge of H,I
hm_top = 0.33        # top of heatmap section
hm_bot = 0.07        # bottom of heatmap section (leave room for colorbars)

# Panel G: 14 bacterial phyla — spans full height of bottom section
ax_d = fig.add_axes([hm_left, hm_bot, hm_mid - hm_left, hm_top - hm_bot])
im_d = plot_phylum_heatmap(ax_d, '16S', 'G', min_asv_total=100)

# Panel H: 2 fungal phyla — short, top of right column
h_height = 0.06
h_bot = hm_top - h_height
ax_e = fig.add_axes([hm_right_x, h_bot, hm_right_end - hm_right_x, h_height])
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
min_asv_class = 50
major_classes = class_totals[class_totals >= min_asv_class].index.tolist()
class_summary = class_summary[class_summary['display_label'].isin(major_classes)]

# Suppress x-axis label on H since I is right below; keep tick labels
ax_e.set_xlabel('')
ax_e.set_xticklabels([])  # remove hour numbers from H bottom since I has them

# Panel I: fungal class heatmap — below H in right column
i_top = h_bot - 0.09  # gap between H and I for H colorbar + I treatment labels
i_height = i_top - hm_bot
ax_f = fig.add_axes([hm_right_x, hm_bot, hm_right_end - hm_right_x, i_height])
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
        # Abbreviate long class names
        cls = cls.replace('Ascomycota (inc. sed.)', 'Asco. (inc. sed.)')
        ylabels_f.append(f"{phylum[0]}. {cls}")
    else:
        ylabels_f.append(c if len(c) < 15 else c[:12] + '.')
ax_f.set_yticks(range(n_rows_f))
ax_f.set_yticklabels(ylabels_f, fontsize=8, fontfamily=FF)
xlabels_f = [TP_LABELS.get(int(tp), str(int(tp))) for tp, _ in all_cols_f]
ax_f.set_xticks(range(n_cols_f))
ax_f.set_xticklabels(xlabels_f, fontsize=7, fontfamily=FF, rotation=60, ha='right')
col_offset = 0
for ti, treat in enumerate(treats):
    n_tp = len(treat_tps_f[treat])
    center = col_offset + (n_tp - 1) / 2
    ax_f.text(center, -1.0, treat, ha='center', va='bottom', fontsize=9,
              fontfamily=FF, fontweight='bold', color=TC[treat], clip_on=False)
    if ti < len(treats) - 1:
        ax_f.axvline(col_offset + n_tp - 0.5, color='black', linewidth=1.0, zorder=5)
    col_offset += n_tp
ax_f.set_xlabel('')  # x-axis label removed; hour numbers on ticks are self-explanatory
ax_f.text(-0.10, 1.12, 'I', transform=ax_f.transAxes, fontsize=PANEL_LABEL,
          fontweight='bold', va='top', fontfamily=FF)

# Colorbars — each under its own panel
fig.canvas.draw()  # force layout calculation

cb_h = 0.012  # colorbar height

# Panel G colorbar: under G (left column, at very bottom)
cb_y_g = 0.015
cax_d = fig.add_axes([hm_left, cb_y_g, hm_mid - hm_left, cb_h])
cb_d = plt.colorbar(im_d, cax=cax_d, orientation='horizontal')
cb_d.set_label('Weighted EAF\n(bacteria, phylum)', fontsize=7, fontfamily=FF)
cb_d.ax.tick_params(labelsize=7, rotation=45)
# Reduce number of ticks to prevent overlap
cb_d.locator = plt.MaxNLocator(nbins=5)
cb_d.update_ticks()

# Panel H colorbar: directly under H (between H and I in right column)
cb_y_h = h_bot - 0.025  # just below Panel H
cax_e = fig.add_axes([hm_right_x, cb_y_h, hm_right_end - hm_right_x, cb_h])
cb_e = plt.colorbar(im_e, cax=cax_e, orientation='horizontal')
cb_e.set_label('Wt. EAF (phylum)', fontsize=7, fontfamily=FF)
cb_e.ax.tick_params(labelsize=7)

# Panel I colorbar: under I (bottom of right column)
cb_y_i = 0.015
cax_f = fig.add_axes([hm_right_x, cb_y_i, hm_right_end - hm_right_x, cb_h])
cb_f = plt.colorbar(im_f, cax=cax_f, orientation='horizontal')
cb_f.set_label('Wt. EAF (class)', fontsize=7, fontfamily=FF)
cb_f.ax.tick_params(labelsize=7)

# Save — print-quality output
for ext in ['pdf', 'png', 'tiff']:
    kwargs = dict(format=ext, dpi=600, bbox_inches='tight', pad_inches=0.05)
    if ext == 'tiff':
        kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(os.path.join(FIGURES_DIR, f'Figure1_qsip_combined_logx.{ext}'), **kwargs)
plt.close(fig)
print("Saved Figure1_qsip_combined_logx.pdf/.png/.tiff")
