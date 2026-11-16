#!/usr/bin/env python3
"""
Generate supplementary figure — Metabolite correlation heatmaps.

Panel A: Spearman correlation of each metabolite labeled fraction with PCo1
         score, per treatment, at timepoint-mean level.
Panel B: Spearman correlation of each metabolite labeled fraction with each
         of the 5 transcriptional programs, per treatment, at timepoint-mean
         level. (Reformatted from program_metabolite_spearman.tsv)

Outputs:
  - FigureS7_program_metabolite_heatmap.pdf/.png
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
FF = 'Arial'
MM = 1 / 25.4

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}
METABOLITES = ['PYR', 'LAT', 'Cit', 'aKG', 'Suc', 'Fum', 'Mal', 'Asp', 'GLUM', 'AGLUM', 'ASR', 'NAA']
METAB_NAMES = {
    'PYR': 'Pyruvate', 'LAT': 'Lactate',
    'Cit': 'Citrate', 'aKG': 'α-KG', 'Suc': 'Succinate',
    'Fum': 'Fumarate', 'Mal': 'Malate', 'Asp': 'Aspartate',
    'GLUM': 'Glutamate', 'AGLUM': 'N-AcGlut',
    'ASR': 'O-Ac-Serine', 'NAA': 'N-Ac-Alanine',
}
TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']
TREAT_LABELS = {'GLUCOSE': 'Glucose', 'XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}
TREAT_COLORS = {'GLUCOSE': '#6B7C2E', 'XYLOSE': '#C8960C', 'CELLULOSE': '#7B3F6E'}

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']
PROG_SHORT = {
    'P_EARLY_MONO': 'P1: Monomer growth',
    'P_LATE_MONO': 'P2: Late succession',
    'P_HEMI': 'P3: Hemicellulose',
}

# ============================================================================
# Load metabolite labeled fractions
# ============================================================================
print("Loading metabolite data...")

TREAT_FILES = {
    'GLUCOSE': ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv', 'GlucoseMetadata.csv'),
    'XYLOSE': ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'CelluloseMetadata.csv'),
}

def load_metab_means(treatment):
    frac_file, meta_file = TREAT_FILES[treatment]
    frac = pd.read_csv(os.path.join(METAB_DIR, frac_file))
    meta = pd.read_csv(os.path.join(METAB_DIR, meta_file))
    sample_cols = [c for c in frac.columns if c not in ['metabolite', 'isotopologue']]
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta.iterrows()}
    results = {}
    for metab in METABOLITES:
        m0 = frac[(frac.metabolite == metab) & (frac.isotopologue == 0)]
        if len(m0) == 0:
            continue
        m0_vals = m0[sample_cols].iloc[0]
        tp_vals = {}
        for sid in sample_cols:
            if str(sid) not in stp: continue
            td = stp[str(sid)]
            if td == 0 or td not in METAB_TO_METAT: continue
            mh = METAB_TO_METAT[td]
            tp_vals.setdefault(mh, []).append(1.0 - float(m0_vals[sid]))
        results[metab] = {tp: np.mean(v) for tp, v in tp_vals.items()}
    return results

metab_data = {}
for treat in TREATMENTS:
    metab_data[treat] = load_metab_means(treat)

# ============================================================================
# Panel A: Metabolite vs PCo1 Spearman correlations
# ============================================================================
print("Computing metabolite vs PCo1 correlations...")

coords = pd.read_csv(os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'), sep='\t')

pco1_corr = np.full((len(METABOLITES), len(TREATMENTS)), np.nan)
pco1_pval = np.full((len(METABOLITES), len(TREATMENTS)), np.nan)

for ti, treat in enumerate(TREATMENTS):
    sub = coords[(coords.treatment == treat) & (coords.time_hours > 0)]
    pco1_means = sub.groupby('time_hours').PCo1.mean()

    for mi, metab in enumerate(METABOLITES):
        if metab not in metab_data[treat]:
            continue
        md = metab_data[treat][metab]
        matched = sorted(set(pco1_means.index) & set(md.keys()))
        if len(matched) < 4:
            continue
        pco1_v = [pco1_means[tp] for tp in matched]
        metab_v = [md[tp] for tp in matched]
        rho, p = spearmanr(pco1_v, metab_v)
        pco1_corr[mi, ti] = rho
        pco1_pval[mi, ti] = p

# ============================================================================
# Panel B: Metabolite vs program Spearman correlations
# Recompute for the expanded metabolite list
# ============================================================================
print("Computing program-metabolite correlations...")

assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')
lfc_cols = [c for c in assign.columns if c.startswith('MONOMER_') or c.startswith('HEMICELLULOSE_')]
prog_lfc = {}
for prog in PROG_ORDER:
    prog_lfc[prog] = assign[assign.mnemonic == prog][lfc_cols].mean(axis=0)

TREAT_TO_SG = {'GLUCOSE': 'MONOMER', 'XYLOSE': 'MONOMER', 'CELLULOSE': 'HEMICELLULOSE'}
from statsmodels.stats.multitest import multipletests

prog_corr_rows = []
for prog in PROG_ORDER:
    for metab in METABOLITES:
        for treat in TREATMENTS:
            sg = TREAT_TO_SG[treat]
            if metab not in metab_data[treat]:
                continue
            md = metab_data[treat][metab]
            matched = sorted(md.keys())
            prog_vals = []
            metab_vals = []
            for tp in matched:
                col = f'{sg}_t{tp}'
                if col not in prog_lfc[prog].index:
                    continue
                prog_vals.append(prog_lfc[prog][col])
                metab_vals.append(md[tp])
            if len(prog_vals) < 4:
                continue
            rho, p = spearmanr(prog_vals, metab_vals)
            prog_corr_rows.append({
                'program': prog, 'metabolite': metab, 'treatment': treat,
                'spearman_rho': rho, 'p_value': p, 'n': len(prog_vals),
            })

prog_corr_df = pd.DataFrame(prog_corr_rows)
if len(prog_corr_df) > 0:
    reject, padj, _, _ = multipletests(prog_corr_df.p_value.values, method='fdr_bh')
    prog_corr_df['padj'] = padj
    prog_corr_df['significant'] = reject
print(f"  {len(prog_corr_df)} correlations computed")

# ============================================================================
# Figure: Panel A (left) + Panel B (3 treatment sub-panels, right)
# ============================================================================
print("Generating figure...")
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(280 * MM, 190 * MM))

# Layout: A on left, gap, then B as 3 sub-heatmaps — generous spacing
gs = fig.add_gridspec(1, 5, width_ratios=[0.6, 0.12, 1, 1, 1], wspace=0.20,
                      left=0.08, right=0.94, top=0.88, bottom=0.12)

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.cm.RdBu_r

def draw_sig_box(ax, row, col, color='black', lw=1.5):
    """Draw a rectangle around a significant cell."""
    ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                            linewidth=lw, edgecolor=color, facecolor='none',
                            zorder=5, clip_on=False))

# --- Panel A: Metabolite vs PCo1 ---
ax_a = fig.add_subplot(gs[0, 0])

masked_pco1 = np.ma.masked_invalid(pco1_corr)
im_a = ax_a.imshow(masked_pco1, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

for mi in range(len(METABOLITES)):
    for ti in range(len(TREATMENTS)):
        val = pco1_corr[mi, ti]
        pval = pco1_pval[mi, ti]
        if np.isnan(val):
            ax_a.add_patch(Rectangle((ti - 0.5, mi - 0.5), 1, 1,
                           facecolor='#F5F5F5', edgecolor='#DDDDDD',
                           linewidth=0.3, zorder=1))
            continue
        txt_color = 'white' if abs(val) > 0.6 else 'black'
        ax_a.text(ti, mi, f'{val:.2f}', ha='center', va='center',
                  fontsize=8, fontfamily=FF, color=txt_color, fontweight='bold')
        if pval < 0.05:
            draw_sig_box(ax_a, mi, ti, color='black', lw=1.8)

ax_a.set_xticks(range(len(TREATMENTS)))
ax_a.set_xticklabels([TREAT_LABELS[t] for t in TREATMENTS],
                      fontsize=8, fontfamily=FF, rotation=45, ha='right')
ax_a.set_yticks(range(len(METABOLITES)))
ax_a.set_yticklabels([METAB_NAMES[m] for m in METABOLITES], fontsize=8, fontfamily=FF)
ax_a.set_title('Metabolite vs PCo1', fontsize=11, fontfamily=FF, fontweight='bold', pad=8)
for sp in ax_a.spines.values():
    sp.set_visible(False)
ax_a.text(-0.35, 1.04, 'A', transform=ax_a.transAxes, fontsize=12,
          fontweight='bold', va='top', fontfamily=FF)

for ti, treat in enumerate(TREATMENTS):
    ax_a.get_xticklabels()[ti].set_color(TREAT_COLORS[treat])

# --- Panel B: Per-treatment program × metabolite heatmaps ---
for ti, treat in enumerate(TREATMENTS):
    ax_b = fig.add_subplot(gs[0, ti + 2])

    sub = prog_corr_df[prog_corr_df.treatment == treat]
    mat = np.full((len(METABOLITES), len(PROG_ORDER)), np.nan)
    pmat = np.full((len(METABOLITES), len(PROG_ORDER)), np.nan)

    for _, row in sub.iterrows():
        if row.metabolite in METABOLITES and row.program in PROG_ORDER:
            mi = METABOLITES.index(row.metabolite)
            pi = PROG_ORDER.index(row.program)
            mat[mi, pi] = row.spearman_rho
            pmat[mi, pi] = row.padj if 'padj' in row.index else row.p_value

    masked_mat = np.ma.masked_invalid(mat)
    im_b = ax_b.imshow(masked_mat, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    for mi in range(len(METABOLITES)):
        for pi in range(len(PROG_ORDER)):
            val = mat[mi, pi]
            pval = pmat[mi, pi]
            if np.isnan(val):
                # Draw light gray hatching for missing data
                ax_b.add_patch(Rectangle((pi - 0.5, mi - 0.5), 1, 1,
                               facecolor='#F5F5F5', edgecolor='#DDDDDD',
                               linewidth=0.3, zorder=1))
                continue
            txt_color = 'white' if abs(val) > 0.6 else 'black'
            ax_b.text(pi, mi, f'{val:.2f}', ha='center', va='center',
                      fontsize=5.5, fontfamily=FF, color=txt_color)
            if not np.isnan(pval) and pval < 0.05:
                draw_sig_box(ax_b, mi, pi, color='black', lw=1.8)

    ax_b.set_xticks(range(len(PROG_ORDER)))
    ax_b.set_xticklabels([PROG_SHORT[p] for p in PROG_ORDER],
                          fontsize=8, fontfamily=FF, rotation=45, ha='right')
    if ti == 0:
        ax_b.set_yticks(range(len(METABOLITES)))
        ax_b.set_yticklabels([METAB_NAMES[m] for m in METABOLITES],
                              fontsize=8, fontfamily=FF)
    else:
        ax_b.set_yticks([])

    ax_b.set_title(TREAT_LABELS[treat], fontsize=11, fontfamily=FF,
                   fontweight='bold', color=TREAT_COLORS[treat], pad=8)
    for sp in ax_b.spines.values():
        sp.set_visible(False)

    if ti == 0:
        ax_b.text(-0.40, 1.04, 'B', transform=ax_b.transAxes, fontsize=12,
                  fontweight='bold', va='top', fontfamily=FF)

# Colorbar
cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.018])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('Spearman ρ', fontsize=8, fontfamily=FF)
cb.ax.tick_params(labelsize=8)

fig.suptitle('Metabolite labeling correlations with community trajectory and transcriptional programs',
             fontsize=11, fontfamily=FF, fontweight='bold', y=0.95)
fig.text(0.5, 0.015, 'Black border = significant (Panel A: raw p < 0.05; Panel B: BH-adjusted p < 0.05). '
         'Timepoint-mean level, n = 6\u20137.',
         fontsize=8, fontfamily=FF, ha='center', color='#666666')

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'FigureS7_program_metabolite_heatmap.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved FigureS7_program_metabolite_heatmap.pdf/.png")
