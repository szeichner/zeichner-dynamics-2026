#!/usr/bin/env python3
"""
Figure 3 — Program trajectories + bubble heatmap — LOG X-AXIS variant.

Left column program trajectory line plots use true log-scale x-axis.
Right column bubble heatmaps remain categorical (grid-based).

Output: figures/output/log_xaxis/Figure3_programs_bubble_logx.pdf/.png
"""

import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
METAT_DIR = os.path.join(BASE, 'data', 'metatranscriptomics', 'img data')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output', 'log_xaxis')
os.makedirs(FIGURES_DIR, exist_ok=True)
FF = 'Arial'
MM = 1 / 25.4

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']
PROG_NAMES = {
    'P_EARLY_MONO': 'P1: Monomer growth\nresponse (n={n})',
    'P_LATE_MONO':  'P2: Late succession &\nnecromass recycling (n={n})',
    'P_HEMI':       'P3: Hemicellulose-\nassociated community (n={n})',
}
PROG_COLORS = {
    'P_EARLY_MONO': '#6B7C2E',
    'P_LATE_MONO':  '#A2C060',
    'P_HEMI':       '#7B3F6E',
}

TP_MONO = [2, 8, 24, 72, 168, 336]
TP_HEMI = [2, 8, 24, 72, 168, 336, 672]
ALL_TP = [2, 8, 24, 72, 168, 336, 672]
TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}

MONO_COLOR = '#6B7C2E'
HEMI_COLOR = '#7B3F6E'

AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
MARKER_SIZE = 5
LINE_WIDTH = 1.5
ROW_LABEL = 7
HEADER_SIZE = 9

# ============================================================================
# Load data for left column
# ============================================================================
print("Loading program assignments...")
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')

lfc_cols_mono = [f'MONOMER_t{t}' for t in TP_MONO]
lfc_cols_hemi = [f'HEMICELLULOSE_t{t}' for t in TP_HEMI]

print(f"  {len(assign)} KOs across {len(PROG_ORDER)} programs")

# ============================================================================
# Load data for right column (bubble heatmap)
# ============================================================================
print("Loading subcategory assignments...")
df_assign = pd.read_csv(os.path.join(TABLES_DIR, 'TableS9_program_subcategory_assignments.tsv'), sep='\t')
mono_cols_bubble = [c for c in df_assign.columns if c.startswith('MONOMER_')]
hemi_cols_bubble = [c for c in df_assign.columns if c.startswith('HEMICELLULOSE_')]
all_data_cols = mono_cols_bubble + hemi_cols_bubble
mono_tp_labels = [TP_LABELS[int(c.split('_t')[1])] for c in mono_cols_bubble]
hemi_tp_labels = [TP_LABELS[int(c.split('_t')[1])] for c in hemi_cols_bubble]
all_tp_labels = mono_tp_labels + hemi_tp_labels
n_mono = len(mono_cols_bubble)
n_hemi = len(hemi_cols_bubble)
n_tp = n_mono + n_hemi

print("Loading raw counts for bubble sizes...")
meta = pd.read_csv(os.path.join(METAT_DIR, 'IMG_metadata.tsv'), sep='\t')
meta['taxon_id'] = meta['taxon_id'].astype(str)
excluded = ['sample20_GLUCOSE_xylose_168hrs_rep4', 'sample25_glucose_XYLOSE_2hrs_rep1',
            'sample71_CELLULOSE_336hrs_rep3']
meta = meta[~meta.sample_name.isin(excluded)].reset_index(drop=True)
meta['substrate_group'] = np.where(meta.treatment.isin(['GLUCOSE', 'XYLOSE']), 'MONOMER',
                          np.where(meta.treatment == 'CELLULOSE', 'HEMICELLULOSE', 'control'))

with open(os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')) as f:
    ko_json = json.load(f)
keep = set(meta.taxon_id)
rows_data = {}
for ko_id, tc in ko_json.items():
    for t, c in tc.items():
        if t in keep:
            rows_data.setdefault(t, {})[ko_id] = c
raw_counts = pd.DataFrame(rows_data).T.fillna(0).astype(int)
raw_counts.index = raw_counts.index.astype(str)
raw_counts = raw_counts.reindex(meta.taxon_id.values)

prog_subcat = {}
for prog in PROG_ORDER:
    sub = df_assign[df_assign.mnemonic == prog]
    cats_seen = []
    for cat in sub.subcategory.unique():
        if cat != 'Other/diverse' and cat not in cats_seen:
            cats_seen.append(cat)
    rows = []
    for cat in cats_seen:
        n = (sub.subcategory == cat).sum()
        if n > 0:
            rows.append((cat, n))
    n_other = (sub.subcategory == 'Other/diverse').sum()
    if n_other > 0:
        rows.append(('Other/diverse', n_other))
    prog_subcat[prog] = rows

prog_lfc = {}
prog_counts = {}
for prog in PROG_ORDER:
    rows = prog_subcat[prog]
    n_rows = len(rows)
    lfc_mat = np.zeros((n_rows, n_tp))
    cnt_mat = np.zeros((n_rows, n_tp))
    for ri, (cat, n) in enumerate(rows):
        mask = (df_assign.mnemonic == prog) & (df_assign.subcategory == cat)
        kos_in_cat = df_assign.loc[mask, 'ko_id'].tolist()
        lfc_mat[ri, :] = df_assign.loc[mask, all_data_cols].mean(axis=0).values
        for ci, cond in enumerate(all_data_cols):
            parts = cond.split('_t')
            sg, tp_val = parts[0], int(parts[1])
            sids = meta[(meta.substrate_group == sg) & (meta.time_hours == tp_val)].taxon_id.values
            if len(sids) == 0:
                continue
            kos_present = [k for k in kos_in_cat if k in raw_counts.columns]
            if kos_present:
                cnt_mat[ri, ci] = raw_counts.loc[sids, kos_present].values.mean()
    prog_lfc[prog] = lfc_mat
    prog_counts[prog] = cnt_mat

all_counts = np.concatenate([prog_counts[p].flatten() for p in PROG_ORDER])
max_count = np.max(all_counts[all_counts > 0]) if (all_counts > 0).any() else 1

all_lfc = np.concatenate([prog_lfc[p].flatten() for p in PROG_ORDER])
vmax_e = min(max(np.nanmax(np.abs(all_lfc)), 1.0), 1.5)
norm_e = mcolors.TwoSlopeNorm(vmin=-vmax_e, vcenter=0, vmax=vmax_e)
cmap_e = plt.cm.RdBu_r
min_radius = 0.05
max_radius = 0.42

# ============================================================================
# FIGURE LAYOUT
# ============================================================================
print("Generating Figure 3 (log x-axis variant)...")

n_rows_per = [len(prog_subcat[p]) for p in PROG_ORDER]
height_ratios = [max(nr, 4) for nr in n_rows_per]

fig = plt.figure(figsize=(200 * MM, 260 * MM))
gs_outer = fig.add_gridspec(1, 2, width_ratios=[0.32, 0.68],
    wspace=0.55, left=0.02, right=0.84, top=0.92, bottom=0.06)
gs_left = gs_outer[0, 0].subgridspec(3, 1, hspace=0.50, height_ratios=height_ratios)
gs_right = gs_outer[0, 1].subgridspec(3, 1, hspace=0.50, height_ratios=height_ratios)

# ============================================================================
# Compute per-replicate program LFC for biological replicate error bars
# ============================================================================
print("Loading VST matrix for replicate-level error bars...")
import gzip
vst = pd.read_csv(os.path.join(BASE, 'data', 'derived', 'vst_counts_R.tsv.gz'), sep='\t', index_col=0)
vst.index = vst.index.astype(str)

# Load metadata
meta_full = pd.read_csv(os.path.join(METAT_DIR, 'IMG_metadata.tsv'), sep='\t')
meta_full['taxon_id'] = meta_full['taxon_id'].astype(str)
meta_full = meta_full[~meta_full.sample_name.isin(excluded)].reset_index(drop=True)
meta_full['substrate_group'] = np.where(meta_full.treatment.isin(['GLUCOSE', 'XYLOSE']), 'MONOMER',
                               np.where(meta_full.treatment == 'CELLULOSE', 'HEMICELLULOSE', 'control'))

# Control mean VST per KO (baseline for LFC)
ctrl_ids = meta_full[meta_full.substrate_group == 'control'].taxon_id.values
ctrl_ids_in_vst = [s for s in ctrl_ids if s in vst.index]
ctrl_mean = vst.loc[ctrl_ids_in_vst].mean(axis=0)

# For each program, compute per-replicate mean LFC across program KOs
rep_lfc = {}  # (prog, substrate_group, timepoint) -> list of per-replicate mean LFCs
for prog in PROG_ORDER:
    prog_kos = assign[assign.mnemonic == prog]['ko_id'].tolist()
    prog_kos_in_vst = [k for k in prog_kos if k in vst.columns]
    if not prog_kos_in_vst:
        continue
    ctrl_prog = ctrl_mean[prog_kos_in_vst].values  # baseline for these KOs

    for sg in ['MONOMER', 'HEMICELLULOSE']:
        tps = TP_MONO if sg == 'MONOMER' else TP_HEMI
        for tp in tps:
            rep_ids = meta_full[(meta_full.substrate_group == sg) &
                               (meta_full.time_hours == tp)].taxon_id.values
            rep_ids_in_vst = [s for s in rep_ids if s in vst.index]
            rep_means = []
            for rid in rep_ids_in_vst:
                sample_vst = vst.loc[rid, prog_kos_in_vst].values
                lfc = sample_vst - ctrl_prog  # VST-space difference ≈ log2FC
                rep_means.append(np.mean(lfc))
            rep_lfc[(prog, sg, tp)] = rep_means

# Print replicate counts
print("  Replicate counts per condition:")
for sg in ['MONOMER', 'HEMICELLULOSE']:
    tps = TP_MONO if sg == 'MONOMER' else TP_HEMI
    counts = [len(rep_lfc.get(('P_EARLY_MONO', sg, tp), [])) for tp in tps]
    print(f"    {sg}: {dict(zip(tps, counts))}")

# ============================================================================
# LEFT COLUMN: Program trajectories — LOG X-AXIS with replicate error bars
# ============================================================================
left_axes = []
for pi, prog in enumerate(PROG_ORDER):
    ax = fig.add_subplot(gs_left[pi])
    left_axes.append(ax)
    sub = assign[assign.mnemonic == prog]
    n_prog = len(sub)

    # Monomer trajectory with replicate-level SEM
    mono_hours = np.array(TP_MONO, dtype=float)
    mono_traj = sub[lfc_cols_mono].mean(axis=0).values
    mono_rep_sem = np.array([
        np.std(rep_lfc.get((prog, 'MONOMER', tp), [0]), ddof=1) /
        np.sqrt(max(len(rep_lfc.get((prog, 'MONOMER', tp), [0])), 1))
        for tp in TP_MONO])

    ax.errorbar(mono_hours, mono_traj, yerr=mono_rep_sem, color=MONO_COLOR,
                linewidth=LINE_WIDTH, linestyle='-', zorder=3,
                marker='o', markersize=MARKER_SIZE, markeredgecolor='black', markeredgewidth=0.3,
                elinewidth=0.8, capsize=2.5, capthick=0.7)

    # Hemicellulose trajectory with replicate-level SEM
    hemi_hours = np.array(TP_HEMI, dtype=float)
    hemi_traj = sub[lfc_cols_hemi].mean(axis=0).values
    hemi_rep_sem = np.array([
        np.std(rep_lfc.get((prog, 'HEMICELLULOSE', tp), [0]), ddof=1) /
        np.sqrt(max(len(rep_lfc.get((prog, 'HEMICELLULOSE', tp), [0])), 1))
        for tp in TP_HEMI])

    ax.errorbar(hemi_hours, hemi_traj, yerr=hemi_rep_sem, color=HEMI_COLOR,
                linewidth=LINE_WIDTH, linestyle='-', zorder=3,
                marker='s', markersize=MARKER_SIZE, markeredgecolor='black', markeredgewidth=0.3,
                elinewidth=0.8, capsize=2.5, capthick=0.7)

    # n= labels: place mono above its error bar, hemi below its error bar
    # to avoid overlap between the two series
    mono_n = [len(rep_lfc.get((prog, 'MONOMER', tp), [])) for tp in TP_MONO]
    hemi_n = [len(rep_lfc.get((prog, 'HEMICELLULOSE', tp), [])) for tp in TP_HEMI]

    for k, tp in enumerate(TP_MONO):
        y_top = mono_traj[k] + mono_rep_sem[k]
        ax.text(mono_hours[k], y_top + 0.025,
                f'n={mono_n[k]}', fontsize=5, fontfamily=FF, ha='center', va='bottom',
                color=MONO_COLOR, fontstyle='italic')

    for k, tp in enumerate(TP_HEMI):
        y_bot = hemi_traj[k] - hemi_rep_sem[k]
        ax.text(hemi_hours[k], y_bot - 0.025,
                f'n={hemi_n[k]}', fontsize=5, fontfamily=FF, ha='center', va='top',
                color=HEMI_COLOR, fontstyle='italic')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3, zorder=1)
    if pi == 0:
        ax.text(500, 0.02, 'control', fontsize=8, fontfamily=FF,
                color='#AAAAAA', ha='right', va='bottom')

    # Log-scale x-axis
    ax.set_xscale('log')
    ax.set_xticks(ALL_TP)
    ax.set_xticklabels([TP_LABELS[t] for t in ALL_TP], fontsize=TICK_LABEL, fontfamily=FF)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(1.5, 900)
    ax.tick_params(labelsize=TICK_LABEL, direction='out', length=3, width=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
    ax.set_xlabel('Time (log$_{10}$ scale)', fontsize=AXIS_LABEL, fontfamily=FF)
    ax.set_ylabel('Mean $\\log_2$FC\nvs control', fontsize=AXIS_LABEL, fontfamily=FF,
                  linespacing=1.1)

# ============================================================================
# RIGHT COLUMN: Bubble heatmaps — unchanged (categorical)
# ============================================================================
right_axes = []
for pi, prog in enumerate(PROG_ORDER):
    ax = fig.add_subplot(gs_right[pi])
    right_axes.append(ax)

    rows = prog_subcat[prog]
    n_rows = len(rows)
    lfc_mat = prog_lfc[prog]
    cnt_mat = prog_counts[prog]

    for i in range(n_rows):
        for j in range(n_tp):
            lfc_val = lfc_mat[i, j]
            count_val = cnt_mat[i, j]
            if np.isnan(lfc_val) or count_val <= 0:
                continue
            frac = np.sqrt(count_val / max_count)
            radius = min_radius + frac * (max_radius - min_radius)
            color = cmap_e(norm_e(lfc_val))
            ax.add_patch(Circle((j, i), radius, facecolor=color, edgecolor='#888888',
                                linewidth=0.2, zorder=3))

    ax.axvline(n_mono - 0.5, color='black', linewidth=1.2, zorder=5)
    ax.text(n_mono / 2 - 0.5, -0.9, 'Monomers',
            fontsize=HEADER_SIZE, fontfamily=FF, ha='center', va='bottom',
            fontweight='bold', color=MONO_COLOR, clip_on=False)
    ax.text(n_mono + n_hemi / 2 - 0.5, -0.9, 'Hemicellulose',
            fontsize=HEADER_SIZE, fontfamily=FF, ha='center', va='bottom',
            fontweight='bold', color=HEMI_COLOR, clip_on=False)

    ax.set_xticks(range(n_tp))
    ax.set_xticklabels(all_tp_labels, fontsize=7, fontfamily=FF, rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=AXIS_LABEL, fontfamily=FF)
    ax.tick_params(axis='x', length=3, width=0.8, pad=2, top=False, bottom=True, direction='out')

    row_labels = [f"{cat.replace(chr(10), ' ')} (n={n})" for cat, n in rows]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=ROW_LABEL, fontfamily=FF, ha='right', color='black')
    ax.tick_params(axis='y', length=0, pad=3)

    ax.set_xlim(-0.5, n_tp - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect('equal')

    for i_line in range(n_rows + 1):
        ax.axhline(i_line - 0.5, color='#EEEEEE', linewidth=0.3, zorder=0)
    for j_line in range(n_tp + 1):
        ax.axvline(j_line - 0.5, color='#EEEEEE', linewidth=0.3, zorder=0)

    for sp in ax.spines.values():
        sp.set_visible(False)

# Panel labels + program names
PROG_LABELS_HORIZ = {
    'P_EARLY_MONO': 'P1: Monomer growth response',
    'P_LATE_MONO':  'P2: Late succession & necromass recycling',
    'P_HEMI':       'P3: Hemicellulose-associated community',
}

fig.canvas.draw()
for pi, prog in enumerate(PROG_ORDER):
    n_prog = len(assign[assign.mnemonic == prog])
    letter = chr(65 + pi)
    left_axes[pi].text(-0.18, 1.12, letter, transform=left_axes[pi].transAxes,
                       fontsize=PANEL_LABEL, fontweight='bold', va='bottom',
                       ha='left', fontfamily=FF)
    left_axes[pi].text(-0.02, 1.12,
                       f'{PROG_LABELS_HORIZ[prog]} (n={n_prog})',
                       transform=left_axes[pi].transAxes,
                       fontsize=11, fontfamily=FF, fontweight='bold',
                       color=PROG_COLORS[prog], va='bottom', ha='left')

# Colorbar
pos_r_top = right_axes[0].get_position()
pos_r_bot = right_axes[2].get_position()
leg_x = pos_r_top.x1 + 0.02
cbar_y0 = pos_r_bot.y0 + 0.05
cbar_h = (pos_r_top.y1 - pos_r_bot.y0) * 0.5
cax_e = fig.add_axes([leg_x, cbar_y0 + 0.15, 0.012, cbar_h])
sm = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
sm.set_array([])
cb_e = plt.colorbar(sm, cax=cax_e, orientation='vertical')
cb_e.set_label('Mean $\\log_2$FC vs control', fontsize=AXIS_LABEL, fontfamily=FF)
cb_e.ax.tick_params(labelsize=TICK_LABEL)

# Size legend
pos_r_c = right_axes[2].get_position()
leg_sz_x = leg_x - 0.005
leg_sz_y = pos_r_c.y0
leg_sz_h = pos_r_c.y1 - pos_r_c.y0
ax_leg = fig.add_axes([leg_sz_x, leg_sz_y, 0.065, leg_sz_h])
ax_leg.set_facecolor('#FAFAFA')
for sp in ax_leg.spines.values():
    sp.set_edgecolor('#CCCCCC'); sp.set_linewidth(0.5)
all_count_flat = all_counts[all_counts > 0]
if len(all_count_flat) > 0:
    legend_counts = sorted(set([max(1, int(np.percentile(all_count_flat, p)))
                                for p in [5, 25, 50, 75, 95, 100]]))
    actual_max = int(np.max(all_count_flat))
    if actual_max not in legend_counts:
        legend_counts.append(actual_max)
        legend_counts = sorted(set(legend_counts))
    n_leg = min(len(legend_counts), 6)
    ax_leg.set_xlim(-0.7, 2.2); ax_leg.set_ylim(-0.5, n_leg + 0.8)
    ax_leg.set_aspect('equal'); ax_leg.set_xticks([]); ax_leg.set_yticks([])
    ax_leg.text(0.75, n_leg + 0.5, 'Mean count\nper gene', fontsize=8, fontfamily=FF,
                fontweight='bold', color='black', va='top', ha='center', linespacing=1.1)
    for k, cnt in enumerate(legend_counts[:n_leg]):
        y_pos = n_leg - 1 - k
        frac_v = np.sqrt(cnt / max_count)
        radius = min_radius + frac_v * (max_radius - min_radius)
        ax_leg.add_patch(Circle((0, y_pos), radius, facecolor='#AAAAAA',
                                edgecolor='#666666', linewidth=0.3, zorder=3))
        ax_leg.text(0.7, y_pos, str(cnt), fontsize=8, fontfamily=FF,
                    ha='left', va='center', color='black')
else:
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.set_xticks([]); ax_leg.set_yticks([])

# Treatment legend
legend_handles = [
    Line2D([0], [0], color=MONO_COLOR, linewidth=LINE_WIDTH, linestyle='-',
           marker='o', markersize=MARKER_SIZE + 1, markeredgecolor='black', markeredgewidth=0.3,
           label='Monomers (glucose + xylose)'),
    Line2D([0], [0], color=HEMI_COLOR, linewidth=LINE_WIDTH, linestyle='-',
           marker='s', markersize=MARKER_SIZE + 1, markeredgecolor='black', markeredgewidth=0.3,
           label='Hemicellulose'),
]
pos_l_bot = left_axes[2].get_position()
fig.legend(handles=legend_handles, fontsize=LEGEND_SIZE, loc='upper center', ncol=2,
           frameon=False, bbox_to_anchor=((pos_l_bot.x0 + pos_l_bot.x1) / 2, pos_l_bot.y0 - 0.02),
           handletextpad=0.4, columnspacing=1.5)

# Save
print("Saving...")
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'Figure3_programs_bubble_logx.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved Figure3_programs_bubble_logx.pdf/.png")
