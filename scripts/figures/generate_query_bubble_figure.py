#!/usr/bin/env python3
"""
Bubble heatmap for specific genes mentioned in query.md,
styled like Figure 3 bubble panels.

Rows = individual genes (grouped by program/category).
Columns = timepoints (Monomers | Hemicellulose).
Bubble color = mean log₂FC vs control (RdBu_r).
Bubble size = mean count per gene (sqrt-scaled).
"""

import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
METAT_DIR = os.path.join(BASE, 'data', 'metatranscriptomics', 'img data')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

# ============================================================================
# Define genes from query.md, organized by program and category
# Each entry: (display_label, ko_id)
# ============================================================================
P1_GENES = [
    # Proteostasis & stress response
    ("Chaperone DnaK", "KO:K04043"),
    ("Chaperone DnaJ", "KO:K03686"),
    ("Chaperone GrpE", "KO:K03687"),
    ("Clp protease (ClpB)", "KO:K03695"),
    ("Lon protease", "KO:K01338"),
    ("Cell division protease FtsH", "KO:K03798"),
    ("Protease HtpX", "KO:K03799"),
    # Thioredoxin / redox
    ("Thioredoxin reductase (NADPH)", "KO:K00384"),
    ("Thioredoxin", "KO:K03671"),
    # Fe-S cluster biogenesis
    ("Fe-S cluster assembly (ATP-binding)", "KO:K09013"),
    ("Fe-S cluster assembly SufB", "KO:K09014"),
    # TCA cycle & central carbon
    ("2-Oxoglutarate metabolism enzyme", "KO:K01616"),
    ("Fumarase class II", "KO:K01679"),
    ("Malate dehydrogenase (decarboxylating)", "KO:K00027"),
    # DNA replication & repair
    ("DNA gyrase subunit A", "KO:K02469"),
    ("DNA gyrase subunit B", "KO:K02470"),
    ("Excinuclease ABC subunit B", "KO:K03702"),
    # Sigma-32 (proteotoxic stress)
    ("Sigma-32 (proteotoxic stress)", "KO:K03089"),
    # Energy conservation & respiration
    ("NADH:quinone reductase (non-electrogenic)", "KO:K03885"),
    ("Quinol-cytochrome c reductase (Fe-S)", "KO:K03890"),
    # Overflow & fermentation
    ("L-Lactate dehydrogenase", "KO:K00016"),
    ("Citrate lyase (beta subunit)", "KO:K01644"),
    # Transport
    ("Sulfonate transport (ATP-binding)", "KO:K15555"),
]

P2_GENES = [
    # Bacterial stress & stationary phase
    ("HSP20 family protein", "KO:K13993"),
    ("Chaperone IbpA", "KO:K04080"),
    ("Glutamate decarboxylase", "KO:K01580"),
    ("Superoxide reductase", "KO:K05919"),
    ("N-Acetylmuramoyl-L-alanine amidase", "KO:K01448"),
    # Protein recycling & autophagy
    ("E3 ubiquitin ligase HUWE1", "KO:K10592"),
    ("Autophagy protein Atg2", "KO:K17906"),
    # Fungal cell wall & growth
    ("Alpha-1,3-glucan synthase", "KO:K00749"),
    ("Delta14-sterol reductase", "KO:K00222"),
    ("Sterol 22-desaturase", "KO:K09831"),
    ("Ergosteryl-aspartate synthase", "KO:K24278"),
    # Fungal energy & iron storage
    ("Ferricrocin synthase", "KO:K22148"),
    ("F-type ATPase subunit O", "KO:K02137"),
    # Necromass N scavenging
    ("Allantoin permease DAL5", "KO:K08192"),
    ("Yeast amino acid transporter", "KO:K16261"),
    # Chromatin remodeling
    ("Chromatin-remodeling ATPase INO80", "KO:K11665"),
    ("SWI/SNF subunit SWI1", "KO:K11771"),
]

P3_GENES = [
    # Surface attachment & secretion
    ("Type IV pilus PilA", "KO:K02650"),
    ("Type IV pilus PilB", "KO:K02652"),
    ("General secretion pathway GspD", "KO:K02453"),
    ("General secretion pathway GspG", "KO:K02456"),
    # TonB-dependent transport
    ("TonB", "KO:K03832"),
    ("Biopolymer transport TolR", "KO:K03560"),
    ("Biopolymer transport ExbB", "KO:K03561"),
    # Polysaccharide utilization
    ("SusC (TonB-dependent)", "KO:K21573"),
    ("SusD/RagB family", "KO:K21572"),
    ("Endoglucanase", "KO:K01179"),
    ("Endo-1,4-beta-xylanase", "KO:K01181"),
    ("Glucuronate-lignin esterase (CE15)", "KO:K25010"),
    # Aromatic catabolism
    ("Protocatechuate 3,4-dioxygenase", "KO:K00449"),
    # AA catabolism & proteolysis
    ("BCAA transport (substrate-binding)", "KO:K01999"),
    ("Tricorn protease", "KO:K08676"),
    ("Dipeptidyl-peptidase 4", "KO:K01278"),
    ("Zinc metalloprotease", "KO:K16922"),
    # Iron acquisition
    ("Iron(III) transport (substrate-binding)", "KO:K02012"),
    # Sugar oxidation
    ("Quinoprotein glucose dehydrogenase", "KO:K00117"),
    ("Aldose sugar dehydrogenase", "KO:K21430"),
]

# Category labels with their positions (row index where category starts)
P1_CATEGORIES = [
    (0, "Proteostasis &\nstress response"),
    (7, "Thioredoxin &\nredox maintenance"),
    (9, "Fe-S cluster\nbiogenesis"),
    (11, "TCA cycle &\ncentral carbon"),
    (14, "DNA replication\n& cell division"),
    (17, "Sigma factors"),
    (18, "Aerobic respiration\n& ETC"),
    (20, "Fermentation &\noverflow metabolism"),
    (22, "ABC/PTS transport"),
]

P2_CATEGORIES = [
    (0, "Bacterial stress &\nstationary phase"),
    (5, "Protein recycling\n& autophagy"),
    (7, "Fungal cell wall\n& growth"),
    (11, "Fungal energy &\niron storage"),
    (13, "Necromass N\nscavenging"),
    (15, "Chromatin\nremodeling"),
]

P3_CATEGORIES = [
    (0, "Surface attachment\n& secretion"),
    (4, "TonB-dependent\ntransport"),
    (7, "CAZymes &\npolymer hydrolysis"),
    (12, "Aromatic & lignin\ncatabolism"),
    (13, "AA catabolism\n& proteolysis"),
    (17, "Iron & metal\nacquisition"),
    (18, "Carbohydrate transport\n& phosphorylation"),
]

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']
PROG_LABELS = {
    'P_EARLY_MONO': 'P1: Monomer growth response',
    'P_LATE_MONO':  'P2: Late succession & necromass recycling',
    'P_HEMI':       'P3: Hemicellulose-associated community',
}
PROG_COLORS = {
    'P_EARLY_MONO': '#6B7C2E',
    'P_LATE_MONO':  '#A2C060',
    'P_HEMI':       '#7B3F6E',
}

GENE_LISTS = {
    'P_EARLY_MONO': P1_GENES,
    'P_LATE_MONO': P2_GENES,
    'P_HEMI': P3_GENES,
}
CAT_LISTS = {
    'P_EARLY_MONO': P1_CATEGORIES,
    'P_LATE_MONO': P2_CATEGORIES,
    'P_HEMI': P3_CATEGORIES,
}

TP_LABELS = {2: '2h', 8: '8h', 24: '1d', 72: '3d', 168: '7d', 336: '14d', 672: '28d'}
MONO_COLOR = '#6B7C2E'
HEMI_COLOR = '#7B3F6E'

AXIS_LABEL = 10
TICK_LABEL = 9
PANEL_LABEL = 12
LEGEND_SIZE = 8
ROW_LABEL = 7
HEADER_SIZE = 8

# ============================================================================
# Load data
# ============================================================================
print("Loading program assignments...")
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')
assign = assign.set_index('ko_id')

mono_cols = [f'MONOMER_t{t}' for t in [2, 8, 24, 72, 168, 336]]
hemi_cols = [f'HEMICELLULOSE_t{t}' for t in [2, 8, 24, 72, 168, 336, 672]]
all_data_cols = mono_cols + hemi_cols
n_mono = len(mono_cols)
n_hemi = len(hemi_cols)
n_tp = n_mono + n_hemi
mono_tp_labels = [TP_LABELS[int(c.split('_t')[1])] for c in mono_cols]
hemi_tp_labels = [TP_LABELS[int(c.split('_t')[1])] for c in hemi_cols]
all_tp_labels = mono_tp_labels + hemi_tp_labels

print("Loading raw counts...")
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

# ============================================================================
# Build LFC and count matrices per program
# ============================================================================
prog_lfc = {}
prog_counts = {}
prog_labels_list = {}
prog_found = {}

for prog in PROG_ORDER:
    genes = GENE_LISTS[prog]
    n_rows = len(genes)
    lfc_mat = np.full((n_rows, n_tp), np.nan)
    cnt_mat = np.zeros((n_rows, n_tp))
    labels = []
    found_flags = []

    for ri, (label, ko_id) in enumerate(genes):
        if ko_id in assign.index:
            row = assign.loc[ko_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            lfc_mat[ri, :] = row[all_data_cols].values.astype(float)

            # Get raw counts per condition
            for ci, cond in enumerate(all_data_cols):
                parts = cond.split('_t')
                sg, tp_val = parts[0], int(parts[1])
                sids = meta[(meta.substrate_group == sg) & (meta.time_hours == tp_val)].taxon_id.values
                if len(sids) == 0:
                    continue
                if ko_id in raw_counts.columns:
                    cnt_mat[ri, ci] = raw_counts.loc[sids, ko_id].values.mean()

            bm = row['baseMean'] if 'baseMean' in row.index else 0
            labels.append(f"{label} (baseMean={bm:.0f})")
            found_flags.append(True)
        else:
            labels.append(f"{label} (not found)")
            found_flags.append(False)

    prog_lfc[prog] = lfc_mat
    prog_counts[prog] = cnt_mat
    prog_labels_list[prog] = labels
    prog_found[prog] = found_flags

# Report missing genes
for prog in PROG_ORDER:
    genes = GENE_LISTS[prog]
    for ri, (label, ko_id) in enumerate(genes):
        if not prog_found[prog][ri]:
            print(f"  WARNING: {ko_id} ({label}) not found in dataset")

# Global scaling
all_counts = np.concatenate([prog_counts[p].flatten() for p in PROG_ORDER])
max_count = np.max(all_counts[all_counts > 0]) if (all_counts > 0).any() else 1
all_lfc = np.concatenate([prog_lfc[p].flatten() for p in PROG_ORDER])
all_lfc_valid = all_lfc[~np.isnan(all_lfc)]
vmax_e = min(max(np.nanmax(np.abs(all_lfc_valid)), 1.0), 2.0)
norm_e = mcolors.TwoSlopeNorm(vmin=-vmax_e, vcenter=0, vmax=vmax_e)
cmap_e = plt.cm.RdBu_r
min_radius = 0.08
max_radius = 0.42

# ============================================================================
# GENERATE ONE FIGURE PER PROGRAM (separate supplemental figures)
# ============================================================================
PROG_SUFFIXES = {
    'P_EARLY_MONO': 'P1',
    'P_LATE_MONO': 'P2',
    'P_HEMI': 'P3',
}

# Shared size legend counts (computed once from all programs)
all_count_flat = all_counts[all_counts > 0]
if len(all_count_flat) > 0:
    legend_counts = sorted(set([max(1, int(np.percentile(all_count_flat, p)))
                                for p in [5, 25, 50, 75, 95, 100]]))
    actual_max = int(np.max(all_count_flat))
    if actual_max not in legend_counts:
        legend_counts.append(actual_max)
        legend_counts = sorted(set(legend_counts))
    legend_counts = legend_counts[:6]
else:
    legend_counts = []

for pi, prog in enumerate(PROG_ORDER):
    genes = GENE_LISTS[prog]
    n_rows = len(genes)
    lfc_mat = prog_lfc[prog]
    cnt_mat = prog_counts[prog]
    labels = prog_labels_list[prog]
    found = prog_found[prog]
    categories = CAT_LISTS[prog]

    # Single-program figure: 180mm wide, height scales with rows
    fig_h = max(n_rows * 9 * MM + 25 * MM, 70 * MM)
    fig = plt.figure(figsize=(180 * MM, fig_h))

    # Single axes for bubbles; tighter margins to reduce whitespace
    fig.subplots_adjust(left=0.42, right=0.78, top=0.96, bottom=0.04)
    ax = fig.add_subplot(111)

    # Draw bubbles
    for i in range(n_rows):
        if not found[i]:
            continue
        for j in range(n_tp):
            lfc_val = lfc_mat[i, j]
            count_val = cnt_mat[i, j]
            if np.isnan(lfc_val) or count_val <= 0:
                continue
            frac = np.sqrt(count_val / max_count)
            radius = min_radius + frac * (max_radius - min_radius)
            color = cmap_e(norm_e(lfc_val))
            ax.add_patch(Circle((j, i), radius, facecolor=color, edgecolor='#888888',
                                linewidth=0.3, zorder=3))

    # Vertical divider
    ax.axvline(n_mono - 0.5, color='black', linewidth=1.2, zorder=5)

    # Column headers — positioned above via transAxes so they don't overlap data
    ax.text(0.21, 1.04, 'Monomers',
            fontsize=HEADER_SIZE + 1, fontfamily=FF, ha='center', va='bottom',
            fontweight='bold', color=MONO_COLOR, transform=ax.transAxes)
    ax.text(0.73, 1.04, 'Hemicellulose',
            fontsize=HEADER_SIZE + 1, fontfamily=FF, ha='center', va='bottom',
            fontweight='bold', color=HEMI_COLOR, transform=ax.transAxes)

    # X-axis
    ax.set_xticks(range(n_tp))
    ax.set_xticklabels(all_tp_labels, fontsize=8, fontfamily=FF)
    ax.set_xlabel('Time', fontsize=AXIS_LABEL, fontfamily=FF)
    ax.tick_params(axis='x', length=3, pad=2, top=False, bottom=True)

    # Y-axis: gene name + KO as two-line tick labels (properly padded by matplotlib)
    gene_tick_labels = []
    for ri, (label, ko_id) in enumerate(genes):
        ko_short = ko_id.replace('KO:', '')
        display_name = labels[ri].split(" (baseMean")[0]
        gene_tick_labels.append(f"{display_name}\n{ko_short}")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(gene_tick_labels, fontsize=6.5, fontfamily=FF,
                       ha='right', linespacing=1.2)
    ax.tick_params(axis='y', length=0, pad=4)

    # Axis limits
    ax.set_xlim(-0.5, n_tp - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect('equal')

    # Grid
    for i_line in range(n_rows + 1):
        ax.axhline(i_line - 0.5, color='#EEEEEE', linewidth=0.3, zorder=0)
    for j_line in range(n_tp + 1):
        ax.axvline(j_line - 0.5, color='#EEEEEE', linewidth=0.3, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ---- Category brackets drawn in axes data coords, far left of gene labels ----
    fig.canvas.draw()  # force layout to get accurate transforms
    for ci, (start_row, cat_name) in enumerate(categories):
        if ci + 1 < len(categories):
            end_row = categories[ci + 1][0] - 1
        else:
            end_row = n_rows - 1
        mid = (start_row + end_row) / 2.0

        # Convert data y coords to axes fraction, then to figure coords
        # Draw bracket line and text using ax.annotate in data coords, offset to the left
        bx = -n_tp * 0.95  # far left of the data area
        ax.plot([bx, bx], [start_row - 0.3, end_row + 0.3],
                color=PROG_COLORS[prog], linewidth=1.5, clip_on=False, solid_capstyle='round')
        ax.plot([bx, bx + 0.15], [start_row - 0.3, start_row - 0.3],
                color=PROG_COLORS[prog], linewidth=1.0, clip_on=False)
        ax.plot([bx, bx + 0.15], [end_row + 0.3, end_row + 0.3],
                color=PROG_COLORS[prog], linewidth=1.0, clip_on=False)
        ax.text(bx - 0.2, mid, cat_name, fontsize=6.5, fontfamily=FF,
                ha='right', va='center', color=PROG_COLORS[prog], fontweight='bold',
                clip_on=False)

    # Title above entire figure
    fig.suptitle(f'{PROG_LABELS[prog]}',
                 fontsize=PANEL_LABEL, fontfamily=FF, fontweight='bold',
                 color=PROG_COLORS[prog], y=0.98)

    # ---- Colorbar ----
    fig.canvas.draw()
    pos_ax = ax.get_position()
    cbar_x = pos_ax.x1 + 0.03
    cbar_h = pos_ax.height * 0.35
    cbar_y = pos_ax.y0 + pos_ax.height * 0.60
    cax = fig.add_axes([cbar_x, cbar_y, 0.015, cbar_h])
    sm = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_label('Mean $\\log_2$FC\nvs control', fontsize=AXIS_LABEL - 1, fontfamily=FF)
    cb.ax.tick_params(labelsize=TICK_LABEL - 1)

    # ---- Size legend ----
    n_leg = len(legend_counts)
    if n_leg > 0:
        leg_h = min(pos_ax.height * 0.45, 0.35)
        ax_leg = fig.add_axes([cbar_x - 0.01, pos_ax.y0, 0.10, leg_h])
        ax_leg.set_facecolor('#FAFAFA')
        for sp2 in ax_leg.spines.values():
            sp2.set_edgecolor('#CCCCCC')
            sp2.set_linewidth(0.5)
        ax_leg.set_xlim(-0.5, 1.5)
        ax_leg.set_ylim(-0.5, n_leg + 0.8)
        ax_leg.set_aspect('equal')
        ax_leg.set_xticks([])
        ax_leg.set_yticks([])
        ax_leg.text(0.5, n_leg + 0.5, 'Mean count\nper gene', fontsize=7, fontfamily=FF,
                    fontweight='bold', color='black', va='top', ha='center', linespacing=1.1)
        for k, cnt in enumerate(legend_counts):
            y_pos = n_leg - 1 - k
            frac_v = np.sqrt(cnt / max_count)
            radius = min_radius + frac_v * (max_radius - min_radius)
            ax_leg.add_patch(Circle((0, y_pos), radius, facecolor='#AAAAAA',
                                    edgecolor='#666666', linewidth=0.3, zorder=3))
            ax_leg.text(0.6, y_pos, str(cnt), fontsize=6.5, fontfamily=FF,
                        ha='left', va='center', color='black')

    # Save
    suffix = PROG_SUFFIXES[prog]
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'query_genes_bubble_{suffix}.{ext}'),
                    format=ext, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved query_genes_bubble_{suffix}.pdf/.png")

print(f"All 3 figures saved to {FIGURES_DIR}")
