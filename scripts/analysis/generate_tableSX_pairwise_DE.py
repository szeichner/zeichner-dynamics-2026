#!/usr/bin/env python3
"""
Generate TableSX — Pairwise treatment-normalized DE summary.

Uses treatment-wise z-score normalized VST values. For each KO term, compare
treatments at matched timepoints using Welch's t-test, combine p-values across
timepoints with Fisher's method, apply BH FDR correction.

Results (Welch's t-test + Fisher's method + BH FDR on raw VST):
- Glucose vs Xylose: 0 DE KO terms (0.0%)
- Glucose vs Hemicellulose: 2,307 DE KO terms (20.6%)
- Xylose vs Hemicellulose: 2,416 DE KO terms (21.5%)
"""

import os, json, warnings
import numpy as np, pandas as pd
from pydeseq2.dds import DeseqDataSet
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAT_DIR = os.path.join(BASE, 'data', 'metatranscriptomics', 'img data')
QC_CSV = os.path.join(BASE, 'data', 'metatranscriptomics', 'qc', 'output_stat_summary.csv')
OUT_DIR = os.path.join(BASE, 'tables', 'output')

MAIN = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']
TL = {'GLUCOSE': 'Glucose', 'XYLOSE': 'Xylose', 'CELLULOSE': 'Hemicellulose'}

# ============================================================================
# DATA LOADING
# ============================================================================
print("Loading data...")
img = pd.read_csv(os.path.join(METAT_DIR, 'IMG_metadata.tsv'), sep='\t')
with open(os.path.join(METAT_DIR, 'Ko_clustering', 'ko_output.json')) as f:
    ko_data = json.load(f)

# QC filtering
qc = pd.read_csv(QC_CSV)
qc['qc_pass_rate'] = (qc['reads_passed_qc'] / qc['raw_read_count']) * 100
qc['flags'] = ((qc['qc_pass_rate'] < 40).astype(int) +
               (qc['pct_reads_assembled'] < 70).astype(int) +
               (qc['contig_n50'] < 200).astype(int) +
               (qc['pct_fully_stranded'] < 85).astype(int) +
               (qc['raw_read_count'] < 50e6).astype(int))
passed_samples = set(qc[qc['flags'] == 0]['sample_name'])
img_qc = img[img['sample_name'].isin(passed_samples)].copy()

# Only treatment samples (not control)
img_treat = img_qc[img_qc['treatment'].isin(MAIN)].copy()
tids = img_treat['taxon_id'].astype(str).tolist()

# Build count matrix
mat = pd.DataFrame([{ko: ko_data[ko].get(t, 0) for ko in ko_data} for t in tids], index=tids)
keep = ((mat > 1).sum(0) >= 5) & (mat.var(0) > 0)
mat = mat.loc[:, keep].astype(int)
meta = img_treat.set_index(img_treat['taxon_id'].astype(str)).loc[mat.index]
ko_columns = mat.columns.tolist()

print(f"  {len(mat)} samples, {mat.shape[1]} KO features")

# ============================================================================
# VST + Treatment-wise z-score normalization
# ============================================================================
print("Running DESeq2 VST...")
clinical = meta[['treatment', 'time_hours']].copy()
clinical['time_hours'] = clinical['time_hours'].astype(str)
dds = DeseqDataSet(counts=mat, metadata=clinical,
                   design_factors=['treatment', 'time_hours'], n_cpus=4)
dds.fit_size_factors()
dds.vst(use_design=False)
vst = pd.DataFrame(dds.layers['vst_counts'], index=mat.index, columns=mat.columns)

# Use raw VST values for pairwise DE (no treatment-wise z-score normalization,
# which would remove the cross-treatment signal we're testing for)
vst_norm = vst

# ============================================================================
# Pairwise DE: Welch's t-test per timepoint + Fisher's method + BH FDR
# ============================================================================
comparisons = [
    ('GLUCOSE', 'XYLOSE'),
    ('GLUCOSE', 'CELLULOSE'),
    ('XYLOSE', 'CELLULOSE'),
]

treat_timepoints = {}
for treat in MAIN:
    treat_timepoints[treat] = sorted(meta[meta['treatment'] == treat]['time_hours'].unique())

results = []

for t1, t2 in comparisons:
    shared_tp = sorted(set(treat_timepoints[t1]) & set(treat_timepoints[t2]))
    print(f"\n{TL[t1]} vs {TL[t2]}: {len(shared_tp)} shared timepoints: {shared_tp}")

    ko_pvals = {}
    ko_mean_diff = {}

    for ko in ko_columns:
        tp_pvals = []
        diffs = []

        for tp in shared_tp:
            idxA = meta[(meta['treatment'] == t1) & (meta['time_hours'] == tp)].index
            idxB = meta[(meta['treatment'] == t2) & (meta['time_hours'] == tp)].index

            valsA = vst_norm.loc[idxA, ko].values.astype(float)
            valsB = vst_norm.loc[idxB, ko].values.astype(float)

            if len(valsA) >= 2 and len(valsB) >= 2:
                if np.std(valsA) == 0 and np.std(valsB) == 0:
                    tp_pvals.append(1.0)
                else:
                    _, p = stats.ttest_ind(valsA, valsB, equal_var=False)
                    if np.isnan(p):
                        p = 1.0
                    tp_pvals.append(p)
                diffs.append(np.mean(valsA) - np.mean(valsB))

        if len(tp_pvals) >= 2:
            tp_pvals_c = [max(p, 1e-300) for p in tp_pvals]
            _, combined_p = stats.combine_pvalues(tp_pvals_c, method='fisher')
            ko_pvals[ko] = combined_p
            ko_mean_diff[ko] = np.mean(diffs)

    ko_list = list(ko_pvals.keys())
    pvals_arr = np.array([ko_pvals[ko] for ko in ko_list])
    reject, fdr_vals, _, _ = multipletests(pvals_arr, alpha=0.05, method='fdr_bh')

    n_sig = int(np.sum(reject))
    n_higher_A = int(sum(1 for i, ko in enumerate(ko_list) if reject[i] and ko_mean_diff[ko] > 0))
    n_higher_B = int(sum(1 for i, ko in enumerate(ko_list) if reject[i] and ko_mean_diff[ko] <= 0))

    comp_label = f"{TL[t1]} vs {TL[t2]}"
    print(f"  Total KO tested: {len(ko_list)}")
    print(f"  Significant (FDR<0.05): {n_sig} ({100*n_sig/len(ko_list):.1f}%)")
    print(f"  Higher in {TL[t1]}: {n_higher_A}, Higher in {TL[t2]}: {n_higher_B}")

    results.append({
        'comparison': comp_label,
        'total_KO_tested': len(ko_list),
        'n_significant_FDR05': n_sig,
        'pct_significant': round(100 * n_sig / len(ko_list), 1),
        'n_higher_treatA': n_higher_A,
        'n_higher_treatB': n_higher_B,
    })

# Save
df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, 'TableSX_pairwise_DE_summary.tsv')
df.to_csv(out_path, sep='\t', index=False)
print(f"\nSaved: {out_path}")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))
