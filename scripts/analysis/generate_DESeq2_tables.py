#!/usr/bin/env python3
"""
Generate Tables S7, S10, S11 from the DESeq2 contrast table.

  S7  — Full DESeq2 differential abundance results (already produced as
        DESeq2_treatment_timepoint_vs_control_t0.tsv.gz; rename + symlink)
  S10 — Pathway-level / CAZy family-level aggregated DESeq2 results
  S11 — Full DESeq2 results for PC2 driver genes (motility/adhesion,
        secretion/export, substrate acquisition machinery)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
TABLES_DIR = os.path.join(BASE_DIR, 'tables', 'output')
DESEQ_TSV = os.path.join(TABLES_DIR, 'DESeq2_treatment_timepoint_vs_control_t0.tsv.gz')

print("Loading DESeq2 results...")
deseq = pd.read_csv(DESEQ_TSV, sep='\t', compression='gzip')
print(f"  {len(deseq)} rows")


# ============================================================================
# TABLE S7 — Full DESeq2 differential abundance (rename existing file)
# ============================================================================
print("\n[Table S7] Full DESeq2 differential abundance...")
# Remove placeholder if exists
ph = os.path.join(TABLES_DIR, 'TableS7_DESeq2_full_differential_abundance_PLACEHOLDER.txt')
if os.path.exists(ph):
    os.remove(ph)

# Save under proper name
table_s7_path = os.path.join(TABLES_DIR, 'TableS7_DESeq2_full_differential_abundance.tsv.gz')
deseq.to_csv(table_s7_path, sep='\t', index=False, compression='gzip')
print(f"  Saved {table_s7_path} ({len(deseq):,} rows)")


# ============================================================================
# TABLE S10 — Pathway-level and CAZy family-level aggregated DESeq2
# ============================================================================
print("\n[Table S10] Pathway / CAZy family aggregates...")

# Pathway KO lists (matches Figure S15)
PATHWAY_KOS = {
    'Glycolysis': ['KO:K00844', 'KO:K00845', 'KO:K01810', 'KO:K00850',
                   'KO:K01623', 'KO:K01624', 'KO:K00134', 'KO:K00150',
                   'KO:K00927', 'KO:K01834', 'KO:K15633', 'KO:K01689',
                   'KO:K00873'],
    'TCA cycle': ['KO:K01647', 'KO:K01681', 'KO:K01682', 'KO:K00031',
                  'KO:K00030', 'KO:K01902', 'KO:K01903', 'KO:K00239',
                  'KO:K00240', 'KO:K00241', 'KO:K00242', 'KO:K01676',
                  'KO:K00024', 'KO:K00026'],
    'Entner-Doudoroff': ['KO:K01625', 'KO:K00036', 'KO:K01690'],
    'Phosphoketolase': ['KO:K01621', 'KO:K00616', 'KO:K00625', 'KO:K00925'],
    'Fermentation': ['KO:K00016', 'KO:K00018', 'KO:K03778', 'KO:K00656',
                     'KO:K00138', 'KO:K04072', 'KO:K13953', 'KO:K00001',
                     'KO:K04021', 'KO:K01652'],
    'Substrate acquisition': ['KO:K21572', 'KO:K21573', 'KO:K02014',
                               'KO:K02666', 'KO:K02651', 'KO:K02043',
                               'KO:K02556'],
    'Acetoclastic methanogenesis': ['KO:K00925', 'KO:K01895', 'KO:K00193',
                                     'KO:K00194', 'KO:K00197', 'KO:K00625',
                                     'KO:K14128'],
}

# CAZyme family keyword definitions (matches Figure S11)
CAZ_FAMILIES = {
    'Cellulases': ['cellulase', 'endoglucanase', 'cellobiohydrolase',
                   'beta-glucosidase'],
    'Xylanases': ['xylanase', 'xylosidase', 'arabinofuranosidase'],
    'Mannan_pectin': ['mannosidase', 'mannanase', 'pectinase'],
    'Amylases': ['amylase', 'glucoamylase', 'pullulanase'],
    'Chitinases': ['chitinase', 'chitobiase', 'chitin'],
    'LPMO_accessory': ['lytic polysaccharide monooxygenase', 'lpmo',
                       'glucuronidase'],
}


def classify_caz(desc):
    if not isinstance(desc, str):
        return None
    d = desc.lower()
    for fam, kws in CAZ_FAMILIES.items():
        if any(kw in d for kw in kws):
            return fam
    return None


# Build aggregated rows
rows = []
for pathway, kos in PATHWAY_KOS.items():
    sub = deseq[deseq['ko_id'].isin(kos)]
    n_defined = len(kos)
    n_detected = sub['ko_id'].nunique()
    if n_detected == 0:
        continue
    for treatment in sub['treatment'].unique():
        for tp in sorted(sub['timepoint_hours'].dropna().unique()):
            t_sub = sub[(sub['treatment'] == treatment) &
                        (sub['timepoint_hours'] == tp)]
            if t_sub.empty:
                continue
            lfc_arr = t_sub['log2FoldChange'].dropna().values
            sig = (t_sub['padj'] < 0.05).sum()
            rows.append({
                'group_name': pathway,
                'annotation_type': 'KO',
                'n_KO_defined': n_defined,
                'n_detected': n_detected,
                'treatment': treatment,
                'timepoint_hours': tp,
                'mean_log2FC': round(np.mean(lfc_arr), 4) if len(lfc_arr) > 0 else np.nan,
                'median_log2FC': round(np.median(lfc_arr), 4) if len(lfc_arr) > 0 else np.nan,
                'SD_log2FC': round(np.std(lfc_arr, ddof=1), 4) if len(lfc_arr) > 1 else np.nan,
                'n_significant': sig,
                'frac_significant': round(sig / len(t_sub), 3) if len(t_sub) > 0 else 0,
                'pathway_significant': 'yes' if (sig >= max(2, 0.2 * n_detected)) else 'no',
            })

# Add CAZyme families
deseq['caz_family'] = deseq['ko_description'].apply(classify_caz)
for fam in CAZ_FAMILIES:
    sub = deseq[deseq['caz_family'] == fam]
    n_detected = sub['ko_id'].nunique()
    if n_detected == 0:
        continue
    for treatment in sub['treatment'].unique():
        for tp in sorted(sub['timepoint_hours'].dropna().unique()):
            t_sub = sub[(sub['treatment'] == treatment) &
                        (sub['timepoint_hours'] == tp)]
            if t_sub.empty:
                continue
            lfc_arr = t_sub['log2FoldChange'].dropna().values
            sig = (t_sub['padj'] < 0.05).sum()
            rows.append({
                'group_name': fam,
                'annotation_type': 'KO_keyword',
                'n_KO_defined': n_detected,
                'n_detected': n_detected,
                'treatment': treatment,
                'timepoint_hours': tp,
                'mean_log2FC': round(np.mean(lfc_arr), 4) if len(lfc_arr) > 0 else np.nan,
                'median_log2FC': round(np.median(lfc_arr), 4) if len(lfc_arr) > 0 else np.nan,
                'SD_log2FC': round(np.std(lfc_arr, ddof=1), 4) if len(lfc_arr) > 1 else np.nan,
                'n_significant': sig,
                'frac_significant': round(sig / len(t_sub), 3) if len(t_sub) > 0 else 0,
                'pathway_significant': 'yes' if (sig >= max(2, 0.2 * n_detected)) else 'no',
            })

table_s10 = pd.DataFrame(rows)
ph = os.path.join(TABLES_DIR, 'TableS10_pathway_CAZy_aggregated_DESeq2_PLACEHOLDER.txt')
if os.path.exists(ph):
    os.remove(ph)
table_s10.to_csv(
    os.path.join(TABLES_DIR, 'TableS10_pathway_CAZy_aggregated_DESeq2.tsv'),
    sep='\t', index=False)
print(f"  Saved {len(table_s10)} aggregated rows across "
      f"{table_s10['group_name'].nunique()} pathway/family groups")


# ============================================================================
# TABLE S11 — Full DESeq2 results for PC2 driver categories
# ============================================================================
print("\n[Table S11] PC2 driver gene full DESeq2 results...")

PC2_CATEGORIES = {
    'motility_adhesion': ['flagell', 'pilus', 'pilin', 'fimbria', 'chemotaxis',
                          'motility', 'adhesin', 'adhesion'],
    'secretion_export': ['secretion', 'sec-translocase', 'signal peptidase',
                         'twin-arginine', 'general secretion'],
    'substrate_acquisition': ['transporter', 'permease', 'tonb', 'susc',
                               'susd', 'outer membrane receptor', 'porin'],
}


def classify_pc2(desc):
    if not isinstance(desc, str):
        return None
    d = desc.lower()
    for cat, kws in PC2_CATEGORIES.items():
        if any(kw in d for kw in kws):
            return cat
    return None


deseq_pc2 = deseq.copy()
deseq_pc2['pc2_category'] = deseq_pc2['ko_description'].apply(classify_pc2)
deseq_pc2 = deseq_pc2[deseq_pc2['pc2_category'].notna()]

ph = os.path.join(TABLES_DIR, 'TableS11_PC2_drivers_full_DESeq2_PLACEHOLDER.txt')
if os.path.exists(ph):
    os.remove(ph)

# Reorder columns
cols = ['ko_id', 'ko_description', 'pc2_category', 'treatment',
        'timepoint_hours', 'log2FoldChange', 'lfcSE', 'pvalue', 'padj',
        'significant']
deseq_pc2_out = deseq_pc2[[c for c in cols if c in deseq_pc2.columns]]
deseq_pc2_out.to_csv(
    os.path.join(TABLES_DIR, 'TableS11_PC2_drivers_full_DESeq2.tsv.gz'),
    sep='\t', index=False, compression='gzip')
print(f"  Saved {len(deseq_pc2_out):,} rows for "
      f"{deseq_pc2_out['ko_id'].nunique()} unique KOs across "
      f"{len(PC2_CATEGORIES)} categories")

print("\nDone.")
