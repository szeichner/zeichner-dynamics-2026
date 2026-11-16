#!/usr/bin/env python3
"""
Generate Figure S12 — Temporal similarity of core functional categories
across substrate treatments.

Five-panel line plot, one per functional category:
  Energy metabolism, Amino acid metabolism, Transport,
  Transcription/Translation, Carbohydrate metabolism

X-axis: log-transformed time (2-336h).
Y-axis: proportion of total expression (%).
All three treatments per panel with SEM error bars.
Significance asterisks where Kruskal-Wallis p < 0.05.

Style: CLAUDE.md (PNAS format, double-column 180 mm wide)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import kruskal

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAT_DIR = os.path.join(BASE_DIR, 'data', 'metatranscriptomics', 'img data')
IMG_METADATA_TSV = os.path.join(METAT_DIR, 'IMG_metadata.tsv')
KO_OUTPUT_JSON = os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')
KO_MAPPING_TXT = os.path.join(METAT_DIR, 'Ko_clustering', 'ko_functionIdMapping.txt')
QC_OUTPUT_STATS_CSV = os.path.join(BASE_DIR, 'data', 'metatranscriptomics',
                                    'qc', 'output_stat_summary.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'output')
TABLES_DIR = os.path.join(BASE_DIR, 'tables', 'output')
os.makedirs(TABLES_DIR, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================
TREATMENT_COLORS = {
    'GLUCOSE':   '#6B7C2E', 'XYLOSE':    '#C8960C',
    'CELLULOSE': '#7B3F6E',
}
TREATMENT_LABELS = {
    'GLUCOSE':   'Glucose', 'XYLOSE':    'Xylose',
    'CELLULOSE': 'Hemicellulose',
}
TREATMENT_MARKERS = {'GLUCOSE': 'o', 'XYLOSE': '^', 'CELLULOSE': 's'}
MAIN_TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']

FONT_FAMILY = 'Arial'
FONT_AXIS_LABEL = 10
FONT_TICK_LABEL = 8
FONT_PANEL_LABEL = 12
FONT_LEGEND = 8
FONT_ANNOTATION = 8
LINEWIDTH_DATA = 1.5
LINEWIDTH_ERROR = 1.0
MARKER_SIZE = 5
CI_ALPHA = 0.15
DPI_PNG = 600
MM_TO_INCH = 1 / 25.4
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 130 * MM_TO_INCH


def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis='both', which='major', direction='out',
                   length=2.5, width=0.6, labelsize=FONT_TICK_LABEL)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def add_panel_label(ax, label):
    ax.text(-0.18, 1.10, label, transform=ax.transAxes,
            fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top',
            fontfamily=FONT_FAMILY)


# ============================================================================
# Functional category keyword classification (from notebook 11)
# ============================================================================
FUNC_CATEGORIES = {
    'Energy metabolism': [
        'atp synthase', 'cytochrome', 'nadh dehydrogenase', 'oxidoreductase',
        'reductase', 'oxidase', 'electron transfer',
    ],
    'Amino acid metabolism': [
        'amino acid', 'glutamate', 'lysine', 'leucine', 'arginine',
        'tryptophan', 'phenylalanine', 'tyrosine', 'histidine', 'cysteine',
        'methionine', 'serine', 'threonine', 'asparagine', 'glutamine',
        'proline', 'valine', 'isoleucine', 'alanine', 'glycine',
        'aminotransferase',
    ],
    'Transport': [
        'transporter', 'transport', 'permease', 'channel', 'pump',
    ],
    'Transcription/Translation': [
        'ribosom', 'translation', 'transcription', 'rna polymerase',
        'aminoacyl-trna', 'elongation factor', 'initiation factor',
    ],
    'Carbohydrate metabolism': [
        'carbohydrate', 'glucose', 'glycos', 'cellulose', 'xylose',
        'polysaccharide', 'starch', 'cellulase', 'xylanase',
        'glycoside hydrolase', 'amylase',
    ],
}


def classify_ko(description):
    if not isinstance(description, str):
        return 'Other'
    desc = description.lower()
    for cat, kws in FUNC_CATEGORIES.items():
        if any(kw in desc for kw in kws):
            return cat
    return 'Other'


# ============================================================================
# QC + DATA LOAD
# ============================================================================
print("Running QC and loading data...")
qc = pd.read_csv(QC_OUTPUT_STATS_CSV)
qc['qc_pass_rate'] = (qc['reads_passed_qc'] / qc['raw_read_count']) * 100
qc['total_flags'] = (
    (qc['qc_pass_rate'] < 40).astype(int) +
    (qc['pct_reads_assembled'] < 70).astype(int) +
    (qc['contig_n50'] < 200).astype(int) +
    (qc['pct_fully_stranded'] < 85).astype(int) +
    (qc['raw_read_count'] < 50e6).astype(int)
)
qc_passed = set(qc[qc['total_flags'] == 0]['sample_name'].values)

img_metadata = pd.read_csv(IMG_METADATA_TSV, sep='\t')
img_metadata = img_metadata[img_metadata['sample_name'].isin(qc_passed)].copy()
img_metadata = img_metadata[img_metadata['treatment'].isin(MAIN_TREATMENTS)].copy()

with open(KO_OUTPUT_JSON, 'r') as f:
    ko_data = json.load(f)

# Build sample × KO matrix (raw counts)
taxon_ids = img_metadata['taxon_id'].astype(str).tolist()
ko_ids = list(ko_data.keys())
ko_matrix = []
for tid in taxon_ids:
    ko_matrix.append([ko_data[ko].get(tid, 0) for ko in ko_ids])
counts_df = pd.DataFrame(ko_matrix, columns=ko_ids, index=taxon_ids)

# Annotate KOs with functional categories
ko_mapping = pd.read_csv(KO_MAPPING_TXT, sep='\t', header=None,
                          names=['ko_id', 'description'])
ko_desc = dict(zip(ko_mapping['ko_id'], ko_mapping['description']))
ko_to_cat = {ko: classify_ko(ko_desc.get(ko, '')) for ko in ko_ids}

# For each sample, sum counts by functional category
sample_cat_data = []
for tid in taxon_ids:
    row = counts_df.loc[tid]
    cat_sums = {cat: 0 for cat in FUNC_CATEGORIES}
    cat_sums['Other'] = 0
    for ko, count in row.items():
        cat = ko_to_cat.get(ko, 'Other')
        cat_sums[cat] += count
    total = sum(cat_sums.values())
    rec = {'taxon_id': tid, 'total': total}
    for cat in cat_sums:
        rec[cat] = cat_sums[cat]
        rec[f'{cat}_prop'] = (cat_sums[cat] / total * 100) if total > 0 else 0
    sample_cat_data.append(rec)

cat_df = pd.DataFrame(sample_cat_data)
cat_df = cat_df.merge(
    img_metadata[['taxon_id', 'treatment', 'time_hours']].astype({'taxon_id': str}),
    left_on='taxon_id', right_on='taxon_id'
)


# ============================================================================
# Aggregate per treatment × timepoint
# ============================================================================
print("Aggregating by treatment and timepoint...")
KEY_CATEGORIES = list(FUNC_CATEGORIES.keys())

summary = []
for cat in KEY_CATEGORIES:
    prop_col = f'{cat}_prop'
    for treatment in MAIN_TREATMENTS:
        for tp in sorted(cat_df['time_hours'].unique()):
            sub = cat_df[(cat_df['treatment'] == treatment) &
                         (cat_df['time_hours'] == tp)]
            if len(sub) == 0:
                continue
            summary.append({
                'category': cat,
                'treatment': treatment,
                'time_hours': tp,
                'mean': sub[prop_col].mean(),
                'sem': sub[prop_col].sem() if len(sub) > 1 else 0,
                'n': len(sub),
            })
summary_df = pd.DataFrame(summary)

# Kruskal-Wallis tests at each timepoint
print("Running Kruskal-Wallis tests...")
sig_tests = []
for cat in KEY_CATEGORIES:
    prop_col = f'{cat}_prop'
    for tp in sorted(cat_df['time_hours'].unique()):
        groups = []
        for treatment in MAIN_TREATMENTS:
            vals = cat_df[(cat_df['treatment'] == treatment) &
                          (cat_df['time_hours'] == tp)][prop_col].values
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) == 3:
            try:
                h, p = kruskal(*groups)
                sig_tests.append({'category': cat, 'time_hours': tp,
                                  'H': h, 'p': p, 'sig': p < 0.05})
            except Exception:
                pass
sig_df = pd.DataFrame(sig_tests)


# ============================================================================
# COMPUTE SIMILARITY METRICS PER CATEGORY (for header annotation)
# ============================================================================
similarity_metrics = []
for cat in KEY_CATEGORIES:
    cat_data = summary_df[summary_df['category'] == cat]
    if len(cat_data) == 0:
        similarity_metrics.append({'category': cat, 'AUC_CV_pct': np.nan})
        continue
    # Compute AUC per treatment over only the timepoints that treatment has
    aucs = []
    for treatment in MAIN_TREATMENTS:
        td = cat_data[cat_data['treatment'] == treatment].sort_values('time_hours')
        td = td[td['time_hours'] <= 336]  # restrict to common range
        if len(td) < 2:
            continue
        log_t = np.log10(td['time_hours'].values)
        aucs.append(np.trapezoid(td['mean'].values, log_t))
    if len(aucs) >= 2:
        auc_arr = np.array(aucs)
        auc_cv = (auc_arr.std() / auc_arr.mean() * 100) if auc_arr.mean() > 0 else 0
    else:
        auc_cv = np.nan
    similarity_metrics.append({'category': cat, 'AUC_CV_pct': auc_cv})

sim_df = pd.DataFrame(similarity_metrics)
print("\n  Similarity metrics:")
print(sim_df.to_string(index=False))


# ============================================================================
# FIGURE S12 — 5 panels
# ============================================================================
print("Generating Figure S12...")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=False)
gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.40,
                      left=0.07, right=0.97, top=0.90, bottom=0.10)

panel_labels = ['A', 'B', 'C', 'D', 'E']
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for idx, (cat, pos, plabel) in enumerate(zip(KEY_CATEGORIES, positions, panel_labels)):
    ax = fig.add_subplot(gs[pos[0], pos[1]])

    for treatment in MAIN_TREATMENTS:
        td = summary_df[(summary_df['category'] == cat) &
                        (summary_df['treatment'] == treatment)].sort_values('time_hours')
        if td.empty:
            continue
        color = TREATMENT_COLORS[treatment]
        # Limit to common timepoints (≤336h)
        td = td[td['time_hours'] <= 336]
        ax.errorbar(td['time_hours'], td['mean'],
                    yerr=td['sem'],
                    fmt=TREATMENT_MARKERS[treatment] + '-',
                    color=color, markersize=MARKER_SIZE,
                    linewidth=LINEWIDTH_DATA,
                    elinewidth=LINEWIDTH_ERROR, capsize=2,
                    capthick=LINEWIDTH_ERROR,
                    markeredgecolor='black', markeredgewidth=0.3,
                    label=TREATMENT_LABELS[treatment] if idx == 0 else '',
                    zorder=3)
        ax.fill_between(td['time_hours'],
                        td['mean'] - td['sem'],
                        td['mean'] + td['sem'],
                        color=color, alpha=CI_ALPHA, zorder=2)

    # Significance asterisks where Kruskal-Wallis p < 0.05
    sig_cat = sig_df[(sig_df['category'] == cat) & (sig_df['sig'])]
    if len(sig_cat) > 0:
        # Place asterisks just above the data
        y_max = ax.get_ylim()[1]
        for _, srow in sig_cat.iterrows():
            ax.text(srow['time_hours'], y_max * 0.96, '*',
                    ha='center', va='top',
                    fontsize=FONT_AXIS_LABEL, fontweight='bold',
                    color='#444444')

    ax.set_xscale('log')
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(1.5, 900)
    # Title with similarity score
    sim_row = sim_df[sim_df['category'] == cat]
    auc_cv = sim_row['AUC_CV_pct'].values[0] if len(sim_row) > 0 else None
    title = cat
    if auc_cv is not None:
        title += f'\n(AUC CV = {auc_cv:.1f}%)'
    ax.set_title(title, fontsize=FONT_AXIS_LABEL,
                 fontfamily=FONT_FAMILY, fontweight='bold', pad=3)

    if pos[0] == 1:
        ax.set_xlabel(r'Time (log$_{10}$ scale)',
                      fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)
    if pos[1] == 0:
        ax.set_ylabel('Proportion of\ntotal expression (%)',
                      fontsize=FONT_AXIS_LABEL, fontfamily=FONT_FAMILY)

    apply_style(ax)
    add_panel_label(ax, plabel)

# Hide the unused 6th subplot (bottom-right)
ax_unused = fig.add_subplot(gs[1, 2])
ax_unused.axis('off')

# Shared legend (use the empty 6th cell)
legend_handles = []
for treatment in MAIN_TREATMENTS:
    legend_handles.append(
        Line2D([0], [0], marker=TREATMENT_MARKERS[treatment],
               color=TREATMENT_COLORS[treatment],
               markerfacecolor=TREATMENT_COLORS[treatment],
               markeredgecolor='black', markeredgewidth=0.3,
               markersize=5, linewidth=LINEWIDTH_DATA,
               label=TREATMENT_LABELS[treatment]))
legend_handles.append(
    Line2D([0], [0], linestyle='none', marker='*', markersize=8,
           color='#444444', label='Kruskal-Wallis $p$ < 0.05'))

ax_unused.legend(handles=legend_handles, fontsize=FONT_LEGEND,
                 loc='center', frameon=False,
                 handletextpad=0.4, labelspacing=0.6)

out_pdf = os.path.join(FIGURES_DIR, 'FigureS12_functional_category_temporal.pdf')
out_png = os.path.join(FIGURES_DIR, 'FigureS12_functional_category_temporal.png')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=DPI_PNG, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")


# ============================================================================
# Save summary as Table S17 (per CLAUDE.md spec)
# ============================================================================
print("\nSaving Table S17 (functional category temporal similarity)...")
out_tsv = os.path.join(TABLES_DIR, 'TableS17_functional_category_similarity.tsv')

# Build the per-category similarity table
table_rows = []
for cat in KEY_CATEGORIES:
    sim_row = sim_df[sim_df['category'] == cat]
    n_sig = sig_df[(sig_df['category'] == cat) & (sig_df['sig'])].shape[0]
    n_total = sig_df[sig_df['category'] == cat].shape[0]
    table_rows.append({
        'functional_category': cat,
        'AUC_CV_pct': round(sim_row['AUC_CV_pct'].values[0], 2)
            if len(sim_row) > 0 else np.nan,
        'sig_timepoints': f'{n_sig}/{n_total}',
    })
pd.DataFrame(table_rows).to_csv(out_tsv, sep='\t', index=False)
print(f"Saved: {out_tsv}")

print("\nDone.")
