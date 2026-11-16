#!/usr/bin/env python3
"""
Generate supplementary tables S1–S18 from existing analysis pipeline.

Re-uses the same QC + DESeq2 VST + PCA + metabolomics + EA pipeline as the
figure scripts. Tables that depend on outputs we don't currently produce
(full DESeq2 differential abundance, qSIP) are skipped or stubbed with a
clearly-marked placeholder note.

Outputs all .tsv files to tables/output/.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pydeseq2.dds import DeseqDataSet

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
JGI_METADATA_XLSX = os.path.join(BASE_DIR, 'data', 'metatranscriptomics', 'qc',
                                  'Zeichner_JGI_MetaT.xlsx')
JGI_REEXTRACT_XLSX = os.path.join(BASE_DIR, 'data', 'metatranscriptomics', 'qc',
                                   'Zeichner_JGI_MetaT_Fall2025_reextract.xlsx')

METABOLOMICS_DIR = os.path.join(BASE_DIR, 'data', 'metabolomics')
GLUCOSE_CORR_AREA_CSV = (
    '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity '
    '/data/Metabolomics/Data/GlucoseResults_ForIsoCor_res_corrected_area.csv')
XYLOSE_ISO_FRAC_CSV = os.path.join(METABOLOMICS_DIR,
    'XyloseResults_ForIsoCor_res_isotopologue_fraction.csv')
CELLULOSE_ISO_FRAC_CSV = os.path.join(METABOLOMICS_DIR,
    'CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv')
GLUCOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'GlucoseMetadata.csv')
XYLOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'XyloseMetadata.csv')
CELLULOSE_META_CSV = os.path.join(METABOLOMICS_DIR, 'CelluloseMetadata.csv')

EA_DIR = os.path.join(BASE_DIR, 'data', 'ea')
EA_GLUCOSE_CSV = os.path.join(EA_DIR, 'glucose_all.csv')
EA_XYLOSE_CSV = os.path.join(EA_DIR, 'xylose_all.csv')
EA_CELLULOSE_CSV = os.path.join(EA_DIR, 'cellulose_all.csv')

CARBON_ADDITION_CSV = os.path.join(BASE_DIR, 'data', 'carbon_addition.csv')

TABLES_DIR = os.path.join(BASE_DIR, 'tables', 'output')
os.makedirs(TABLES_DIR, exist_ok=True)

MAIN_TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']
TIME_MAPPING = {
    1.92: 2.0, 7.92: 8.0, 24.0: 24.0, 48.0: 72.0,
    168.0: 168.0, 336.0: 336.0, 672.0: 672.0,
}
TCA_METS = ['Cit', 'aKG', 'Suc', 'Fum', 'Mal']
ALL_METS = ['Cit', 'aKG', 'Suc', 'Fum', 'Mal', 'Asp', 'GLUM', 'AGLUM']

# ============================================================================
# Functional category classifier (matches Figure S12 / S2)
# ============================================================================
FUNC_CATEGORIES = {
    'Ribosomal_translation': [
        'ribosom', 'translation', 'elongation factor', 'initiation factor',
        'aminoacyl-trna', 'trna synthetase',
    ],
    'Central_carbon_metabolism': [
        'glycolysis', 'glucokinase', 'hexokinase', 'phosphofructokinase',
        'pyruvate kinase', 'pyruvate carboxylase', 'enolase',
        'phosphoglycerate', 'citrate synthase', 'aconit',
        'isocitrate dehydrogenase', 'succinate dehydrogenase',
        'fumarate hydratase', 'malate dehydrogenase', 'succinyl-coa',
    ],
    'Energy_electron_transport': [
        'atp synthase', 'cytochrome', 'nadh dehydrogenase',
        'electron transfer', 'ferredoxin', 'oxidoreductase',
    ],
    'Lipid_biosynthesis': [
        'fatty acid synthase', 'fatty acid', 'phospholipid', 'phosphatidyl',
        'glycerolipid', 'choline kinase',
    ],
    'AA_nt_biosynthesis': [
        'amino acid biosynth', 'purine biosynth', 'pyrimidine biosynth',
        'histidine biosynth', 'tryptophan synth', 'chorismate', 'shikimate',
        'nucleotide biosynth',
    ],
    'Substrate_acquisition': [
        'transporter', 'transport system', 'permease', 'abc transport',
        'tonb', 'susc', 'susd', 'outer membrane receptor', 'porin',
    ],
    'CAZymes': [
        'cellulase', 'xylanase', 'glucosidase', 'galactosidase', 'amylase',
        'glycoside hydrolase', 'polysaccharide', 'cellobiohydrolase',
        'endoglucanase', 'mannosidase', 'pectinase', 'chitinase',
    ],
    'Fermentation': [
        'alcohol dehydrogenase', 'lactate dehydrogenase', 'acetate kinase',
        'phosphate acetyltransferase',
    ],
    'Motility_adhesion': [
        'flagell', 'pilus', 'pilin', 'fimbria', 'chemotaxis', 'motility',
        'adhesin', 'adhesion',
    ],
    'Secretion_export': [
        'secretion', 'sec-translocase', 'signal peptidase',
        'twin-arginine', 'general secretion',
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
# QC FILTERING
# ============================================================================
print("Running QC filtering...")
qc = pd.read_csv(QC_OUTPUT_STATS_CSV)
qc['qc_pass_rate'] = (qc['reads_passed_qc'] / qc['raw_read_count']) * 100
qc['flag_qc_pass'] = qc['qc_pass_rate'] < 40
qc['flag_assembly'] = qc['pct_reads_assembled'] < 70
qc['flag_n50'] = qc['contig_n50'] < 200
qc['flag_stranded'] = qc['pct_fully_stranded'] < 85
qc['flag_reads'] = qc['raw_read_count'] < 50e6
qc['total_flags'] = (qc[['flag_qc_pass', 'flag_assembly', 'flag_n50',
                          'flag_stranded', 'flag_reads']].sum(axis=1))
qc['flagged'] = qc['total_flags'] > 0
qc_passed_set = set(qc[~qc['flagged']]['sample_name'].values)


# ============================================================================
# TABLE S1 — Sample metadata and QC
# ============================================================================
print("\n[Table S1] Sample metadata and sequencing quality metrics...")
img_metadata = pd.read_csv(IMG_METADATA_TSV, sep='\t')

def parse_treatment(name):
    if name.lower().startswith('basal'):
        return 'basal'
    if 'control' in name.lower():
        return 'control'
    if 'CELLULOSE' in name:
        return 'CELLULOSE'
    if 'GLUCOSE_xylose' in name:
        return 'GLUCOSE'
    if 'glucose_XYLOSE' in name:
        return 'XYLOSE'
    return 'other'

qc['treatment'] = qc['sample_name'].apply(parse_treatment)
qc['timepoint_hours'] = qc['sample_name'].apply(
    lambda s: int(re.search(r'_(\d+)hrs', s).group(1))
              if re.search(r'_(\d+)hrs', s) else 0)
qc['replicate'] = qc['sample_name'].apply(
    lambda s: int(re.search(r'rep(\d+)', s).group(1))
              if re.search(r'rep(\d+)', s) else 1)

# Merge with IMG metadata to get taxon_id
img_lookup = dict(zip(img_metadata['sample_name'], img_metadata['taxon_id']))
qc['IMG_taxon_id'] = qc['sample_name'].map(img_lookup)

# Reason for flag
def flag_reason(row):
    reasons = []
    if row['flag_qc_pass']:
        reasons.append(f"low_QC_pass ({row['qc_pass_rate']:.1f}%)")
    if row['flag_assembly']:
        reasons.append(f"low_assembly ({row['pct_reads_assembled']:.1f}%)")
    if row['flag_n50']:
        reasons.append(f"low_N50 ({row['contig_n50']:.0f})")
    if row['flag_stranded']:
        reasons.append(f"low_stranded ({row['pct_fully_stranded']:.1f}%)")
    if row['flag_reads']:
        reasons.append(f"low_reads ({row['raw_read_count']/1e6:.0f}M)")
    return '; '.join(reasons) if reasons else 'none'

qc['flag_reason'] = qc.apply(flag_reason, axis=1)
qc['retained_in_analysis'] = (~qc['flagged']).map({True: 'yes', False: 'no'})

# --- Merge JGI library prep metadata (RNA concentration, A260/280, plate, batch) ---
def _load_jgi_sheet(path, source_label):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['DATA SHEET']
    rows = []
    # Row 1 = header, row 2 = descriptions, data starts row 3
    for r in range(3, ws.max_row + 1):
        sn = ws.cell(r, 5).value
        if not sn:
            continue
        rows.append({
            'jgi_sample_name': sn,
            'RNA_concentration_ng_per_ul': ws.cell(r, 8).value,
            'RNA_volume_ul': ws.cell(r, 9).value,
            'A260_280': ws.cell(r, 10).value,
            'tube_or_plate_label': ws.cell(r, 12).value,
            'plate_well': ws.cell(r, 14).value,
            'sample_isolation_method': (ws.cell(r, 19).value or '').strip()
                                        if ws.cell(r, 19).value else None,
            'JGI_metadata_source': source_label,
        })
    return pd.DataFrame(rows)

jgi_main = _load_jgi_sheet(JGI_METADATA_XLSX, 'original')
jgi_re = _load_jgi_sheet(JGI_REEXTRACT_XLSX, 'fall2025_reextract')
# Re-extract values take precedence: drop main rows whose sample id was re-extracted
def _sample_key(name):
    if not isinstance(name, str):
        return None
    m = re.match(r'(sample\d+|basal\d+)', name, re.IGNORECASE)
    return m.group(1).lower() if m else name.lower()

jgi_main['sample_key'] = jgi_main['jgi_sample_name'].apply(_sample_key)
jgi_re['sample_key'] = jgi_re['jgi_sample_name'].apply(_sample_key)
jgi_main = jgi_main[~jgi_main['sample_key'].isin(set(jgi_re['sample_key']))]
jgi_all = pd.concat([jgi_main, jgi_re], ignore_index=True)

qc['sample_key'] = qc['sample_name'].apply(_sample_key)
qc = qc.merge(
    jgi_all.drop(columns=['jgi_sample_name']),
    on='sample_key', how='left',
).drop(columns=['sample_key'])

table_s1 = qc[[
    'IMG_taxon_id', 'sample_name', 'treatment', 'timepoint_hours', 'replicate',
    'raw_read_count', 'reads_passed_qc', 'qc_pass_rate',
    'avg_read_length', 'contig_count', 'contig_n50', 'total_bases_assembled',
    'pct_reads_assembled', 'pct_fully_stranded',
    'RNA_concentration_ng_per_ul', 'RNA_volume_ul', 'A260_280',
    'tube_or_plate_label', 'plate_well', 'sample_isolation_method',
    'JGI_metadata_source',
    'flag_reason', 'retained_in_analysis',
]].sort_values(['treatment', 'timepoint_hours', 'replicate'])
table_s1.to_csv(os.path.join(TABLES_DIR, 'TableS1_sample_metadata_QC.tsv'),
                sep='\t', index=False)
print(f"  {len(table_s1)} samples; {table_s1['retained_in_analysis'].value_counts().to_dict()}")


# ============================================================================
# TABLE S2 — Soil and amendment summary
# ============================================================================
print("\n[Table S2] Soil and amendment summary...")
table_s2 = pd.DataFrame([
    {'treatment': 'Glucose+xylose (13C-glucose)',
     'soil_mass_g': 20.0,
     'total_C_added_mg': 6.940675,
     'C_per_g_soil_mg': 6.940675 / 20.0,
     '13C_fraction_of_amendment': 5.155556 / 6.940675,
     '13C_added_mg': 5.155556,
     'amendment_pct_of_soil_C': 6.940675 / 200.0 * 100,
     'note': '13C-glucose + unlabeled xylose; soil C ~200 mg/microcosm (1% TOC)'},
    {'treatment': 'Glucose+xylose (13C-xylose)',
     'soil_mass_g': 20.0,
     'total_C_added_mg': 6.940675,
     'C_per_g_soil_mg': 6.940675 / 20.0,
     '13C_fraction_of_amendment': 1.785120 / 6.940675,
     '13C_added_mg': 1.785120,
     'amendment_pct_of_soil_C': 6.940675 / 200.0 * 100,
     'note': '13C-xylose + unlabeled glucose'},
    {'treatment': 'Hemicellulose',
     'soil_mass_g': 20.0,
     'total_C_added_mg': 8.888889,
     'C_per_g_soil_mg': 8.888889 / 20.0,
     '13C_fraction_of_amendment': 1.0,
     '13C_added_mg': 8.888889,
     'amendment_pct_of_soil_C': 8.888889 / 200.0 * 100,
     'note': '100% 13C-hemicellulose polymer; substrate purity 97%'},
])
table_s2.to_csv(os.path.join(TABLES_DIR, 'TableS2_soil_amendment_summary.tsv'),
                sep='\t', index=False)
print(f"  Saved {len(table_s2)} treatment rows")


# ============================================================================
# Run VST + PCA pipeline (shared by Tables S3, S4, S5, S6)
# ============================================================================
print("\nLoading metatranscriptomics + running VST + PCA...")
img_meta_qc = img_metadata[img_metadata['sample_name'].isin(qc_passed_set)].copy()

with open(KO_OUTPUT_JSON, 'r') as f:
    ko_data = json.load(f)

taxon_ids = img_meta_qc['taxon_id'].astype(str).tolist()
ko_matrix = []
for tid in taxon_ids:
    ko_matrix.append([ko_data[ko].get(tid, 0) for ko in ko_data.keys()])
ko_df = pd.DataFrame(ko_matrix, columns=list(ko_data.keys()), index=taxon_ids)

ko_presence = (ko_df > 1).sum(axis=0)
ko_filt = ko_df.loc[:, ko_presence >= 5]
ko_filt = ko_filt.loc[:, ko_filt.var(axis=0) > 0]

counts = ko_filt.astype(int)
meta = img_meta_qc.set_index(img_meta_qc['taxon_id'].astype(str))
counts.index = counts.index.astype(str)
common = counts.index.intersection(meta.index)
counts = counts.loc[common]
meta = meta.loc[common]

dds = DeseqDataSet(counts=counts, metadata=meta,
                   design_factors=['treatment', 'time_hours'], n_cpus=4)
dds.deseq2()
dds.vst_fit()
vst_counts = dds.vst_transform()
vst_df = pd.DataFrame(vst_counts, index=counts.index, columns=counts.columns)

ctrl_mask = meta['treatment'] == 'control'
ctrl_mean = vst_df.loc[ctrl_mask].mean(axis=0)
ctrl_std = vst_df.loc[ctrl_mask].std(axis=0).replace(0, 1)
vst_cs = (vst_df - ctrl_mean) / ctrl_std

pca = PCA(n_components=10)
pca_coords = pca.fit_transform(vst_cs.values)
pca_score_df = pd.DataFrame(pca_coords[:, :4], index=vst_cs.index,
                             columns=['PC1', 'PC2', 'PC3', 'PC4'])
pca_score_df = pca_score_df.join(meta[['sample_name', 'treatment',
                                        'time_hours', 'replicate']])

# Load KO descriptions
ko_mapping = pd.read_csv(KO_MAPPING_TXT, sep='\t', header=None,
                          names=['ko_id', 'description'])
ko_desc = dict(zip(ko_mapping['ko_id'], ko_mapping['description']))


# ============================================================================
# TABLE S3 — Full PC1 and PC2 loading values
# ============================================================================
print("\n[Table S3] Full PC1 and PC2 loadings...")
table_s3 = pd.DataFrame({
    'KO_id': vst_cs.columns,
    'KO_description': [ko_desc.get(k, '') for k in vst_cs.columns],
    'PC1_loading': pca.components_[0],
    'PC2_loading': pca.components_[1],
})
table_s3['abs_PC1'] = table_s3['PC1_loading'].abs()
table_s3['abs_PC2'] = table_s3['PC2_loading'].abs()
table_s3['rank_abs_PC1'] = table_s3['abs_PC1'].rank(ascending=False).astype(int)
table_s3['rank_abs_PC2'] = table_s3['abs_PC2'].rank(ascending=False).astype(int)
table_s3['top1000_PC1'] = (table_s3['rank_abs_PC1'] <= 1000).map({True: 'yes', False: 'no'})
table_s3['top1000_PC2'] = (table_s3['rank_abs_PC2'] <= 1000).map({True: 'yes', False: 'no'})
table_s3['functional_category'] = table_s3['KO_description'].apply(classify_ko)
table_s3 = table_s3.sort_values('abs_PC1', ascending=False)
table_s3.to_csv(os.path.join(TABLES_DIR, 'TableS3_PCA_loadings.tsv'),
                sep='\t', index=False)
print(f"  {len(table_s3)} KO features")


# ============================================================================
# TABLE S4 — PC1 trajectory regression
# ============================================================================
print("\n[Table S4] PC1 trajectory regressions...")
pca_main = pca_score_df[pca_score_df['treatment'].isin(MAIN_TREATMENTS)].copy()
pca_main['time_hours'] = pca_main['time_hours'].astype(float)
pca_main = pca_main[pca_main['time_hours'] > 0]
pca_main['log10_time'] = np.log10(pca_main['time_hours'])

table_s4_rows = []
for treatment in MAIN_TREATMENTS:
    td = pca_main[pca_main['treatment'] == treatment]
    slope, intercept, r, p, se = stats.linregress(td['log10_time'], td['PC1'])
    table_s4_rows.append({
        'treatment': treatment,
        'n_samples': len(td),
        'slope_PC1_per_log10h': round(slope, 3),
        'slope_SE': round(se, 3),
        'intercept': round(intercept, 3),
        'pearson_r': round(r, 4),
        'r_squared': round(r ** 2, 4),
        'p_value': p,
        'PC1_change_2h_to_336h': round(slope * (np.log10(336) - np.log10(2)), 2),
    })

# Add ANCOVA comparisons
import statsmodels.formula.api as smf
pca_gx = pca_main[pca_main['treatment'].isin(['GLUCOSE', 'XYLOSE'])].copy()
m_gx = smf.ols('PC1 ~ log10_time * C(treatment)', data=pca_gx).fit()
gx_term = [k for k in m_gx.params.index if 'log10_time:' in k][0]
gx_p = m_gx.pvalues[gx_term]

pca_main['is_monomer'] = pca_main['treatment'].isin(['GLUCOSE', 'XYLOSE']).astype(int)
pca_main['group'] = pca_main['is_monomer'].map({1: 'monomer', 0: 'hemicellulose'})
m_mh = smf.ols('PC1 ~ log10_time * C(group)', data=pca_main).fit()
mh_term = [k for k in m_mh.params.index if 'log10_time:' in k][0]
mh_p = m_mh.pvalues[mh_term]

table_s4_rows.append({
    'treatment': 'ANCOVA Glucose vs Xylose (slope diff)',
    'n_samples': int(m_gx.nobs),
    'slope_PC1_per_log10h': round(m_gx.params[gx_term], 3),
    'slope_SE': round(m_gx.bse[gx_term], 3),
    'intercept': '',
    'pearson_r': '',
    'r_squared': round(m_gx.rsquared, 4),
    'p_value': gx_p,
    'PC1_change_2h_to_336h': '',
})
table_s4_rows.append({
    'treatment': 'ANCOVA Monomer vs Hemicellulose (slope diff)',
    'n_samples': int(m_mh.nobs),
    'slope_PC1_per_log10h': round(m_mh.params[mh_term], 3),
    'slope_SE': round(m_mh.bse[mh_term], 3),
    'intercept': '',
    'pearson_r': '',
    'r_squared': round(m_mh.rsquared, 4),
    'p_value': mh_p,
    'PC1_change_2h_to_336h': '',
})

pd.DataFrame(table_s4_rows).to_csv(
    os.path.join(TABLES_DIR, 'TableS4_PC1_trajectory_regression.tsv'),
    sep='\t', index=False)
print(f"  3 per-treatment + 2 ANCOVA rows")


# ============================================================================
# TABLE S5 — Per-sample PC scores
# ============================================================================
print("\n[Table S5] Per-sample PC scores...")
table_s5 = pca_score_df[['sample_name', 'treatment', 'time_hours',
                          'replicate', 'PC1', 'PC2', 'PC3', 'PC4']].copy()
table_s5 = table_s5.sort_values(['treatment', 'time_hours', 'replicate'])
table_s5.to_csv(os.path.join(TABLES_DIR, 'TableS5_per_sample_PC_scores.tsv'),
                sep='\t', index=False)
print(f"  {len(table_s5)} samples")


# ============================================================================
# TABLE S6 — NMDS coordinates
# ============================================================================
print("\n[Table S6] NMDS coordinates...")
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
dist = squareform(pdist(vst_cs.values, metric='euclidean'))
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
          max_iter=300, n_init=4)
mds_coords = mds.fit_transform(dist)
table_s6 = pd.DataFrame({
    'sample_name': meta['sample_name'].values,
    'treatment': meta['treatment'].values,
    'time_hours': meta['time_hours'].values,
    'NMDS1': mds_coords[:, 0],
    'NMDS2': mds_coords[:, 1],
})
table_s6 = table_s6.sort_values(['treatment', 'time_hours'])
table_s6.to_csv(os.path.join(TABLES_DIR, 'TableS6_NMDS_coordinates.tsv'),
                sep='\t', index=False)
# Footer info
with open(os.path.join(TABLES_DIR, 'TableS6_NMDS_coordinates.tsv'), 'a') as f:
    f.write(f"\n# NMDS stress = {mds.stress_:.4f}\n")
    f.write(f"# n_iterations (max) = 300, n_init = 4\n")
print(f"  {len(table_s6)} samples; stress={mds.stress_:.4f}")


# ============================================================================
# TABLES S8 & S9 — Top 1000 PC1 / PC2 driver genes
# ============================================================================
print("\n[Tables S8 & S9] Top 1000 PC1 / PC2 driver genes (loadings only)...")
# Per-treatment mean expression for each KO (in control-standardized VST)
treatment_means = pd.DataFrame(index=vst_cs.columns)
for treatment in MAIN_TREATMENTS:
    tmask = meta['treatment'] == treatment
    treatment_means[treatment] = vst_cs.loc[meta[tmask].index].mean(axis=0)

# Top 1000 by |PC1|
top_pc1 = table_s3.nlargest(1000, 'abs_PC1')[
    ['KO_id', 'KO_description', 'functional_category', 'PC1_loading', 'rank_abs_PC1']
].copy()
for treatment in MAIN_TREATMENTS:
    top_pc1[f'mean_VST_{treatment}'] = top_pc1['KO_id'].map(treatment_means[treatment])
top_pc1.to_csv(os.path.join(TABLES_DIR, 'TableS8_top1000_PC1_drivers.tsv'),
               sep='\t', index=False)

# Top 1000 by |PC2|, with category flags
top_pc2 = table_s3.nlargest(1000, 'abs_PC2')[
    ['KO_id', 'KO_description', 'functional_category', 'PC2_loading', 'rank_abs_PC2']
].copy()
for treatment in MAIN_TREATMENTS:
    top_pc2[f'mean_VST_{treatment}'] = top_pc2['KO_id'].map(treatment_means[treatment])
# Specific KO flags
SUS_KOS = {'KO:K21572', 'KO:K21573'}
T4P_KOS = {'KO:K02666', 'KO:K02651'}
top_pc2['motility_adhesion_flag'] = top_pc2['functional_category'].apply(
    lambda c: 'yes' if c == 'Motility_adhesion' else 'no')
top_pc2['secretion_export_flag'] = top_pc2['functional_category'].apply(
    lambda c: 'yes' if c == 'Secretion_export' else 'no')
top_pc2['substrate_acquisition_flag'] = top_pc2['KO_id'].apply(
    lambda k: 'yes' if k in (SUS_KOS | T4P_KOS) else 'no')
top_pc2.to_csv(os.path.join(TABLES_DIR, 'TableS9_top1000_PC2_drivers.tsv'),
               sep='\t', index=False)
print(f"  Saved top-1000 PC1 and PC2 driver tables")


# ============================================================================
# Tables S7, S10, S11 — DESeq2 differential abundance (NOT GENERATED)
# ============================================================================
print("\n[Tables S7, S10, S11] SKIPPED — require full DESeq2 differential")
print("  abundance results (treatment-timepoint vs control comparisons).")
print("  Generate from notebook 02 / 06 outputs once available.")
# Write a placeholder note
for tnum, name in [
    ('S7', 'DESeq2_full_differential_abundance'),
    ('S10', 'pathway_CAZy_aggregated_DESeq2'),
    ('S11', 'PC2_drivers_full_DESeq2'),
]:
    with open(os.path.join(TABLES_DIR, f'Table{tnum}_{name}_PLACEHOLDER.txt'),
              'w') as f:
        f.write(f"Table {tnum} placeholder.\n")
        f.write("This table requires full DESeq2 differential abundance results\n")
        f.write("(treatment-timepoint comparisons vs unamended control at t=0).\n")
        f.write("To generate: run DESeq2 contrasts in notebook 02 or 06,\n")
        f.write("export results, and merge with KO functional annotations.\n")


# ============================================================================
# LOAD METABOLOMICS (Tables S12, S13, S14)
# ============================================================================
print("\nLoading metabolomics for Tables S12-S14...")
gluc_meta = pd.read_csv(GLUCOSE_META_CSV)
gluc_meta['sample'] = gluc_meta['sample'].astype(str)
xyl_meta = pd.read_csv(XYLOSE_META_CSV)
cell_meta = pd.read_csv(CELLULOSE_META_CSV)


def load_iso_data(filepath, metadata, substrate, from_areas=False):
    df = pd.read_csv(filepath)
    sample_cols = [c for c in df.columns if c not in ['metabolite', 'isotopologue']]
    rows = []
    for met in df['metabolite'].unique():
        if met not in ALL_METS:
            continue
        met_data = df[df['metabolite'] == met]
        n_iso = len(met_data)
        for sample in sample_cols:
            mr = metadata[metadata['sample'] == sample]
            if len(mr) == 0:
                continue
            tp_d = mr['timepoint'].values[0]
            tp_h = tp_d * 24
            rep = mr['replicate'].values[0]
            if from_areas:
                areas = met_data[sample].values.astype(float)
                total = np.nansum(areas)
                if total == 0 or np.isnan(total):
                    continue
                fractions = areas / total
                isos = met_data['isotopologue'].values
                for i, iso in enumerate(isos):
                    rows.append({
                        'sample_id': sample, 'substrate': substrate,
                        'timepoint_hours': tp_h, 'replicate': rep,
                        'metabolite': met, 'isotopologue': f'M+{int(iso)}',
                        'corrected_area': float(areas[i]),
                        'isotopologue_fraction': float(fractions[i]),
                    })
            else:
                fractions = met_data[sample].values.astype(float)
                isos = met_data['isotopologue'].values
                for i, iso in enumerate(isos):
                    if np.isnan(fractions[i]):
                        continue
                    rows.append({
                        'sample_id': sample, 'substrate': substrate,
                        'timepoint_hours': tp_h, 'replicate': rep,
                        'metabolite': met, 'isotopologue': f'M+{int(iso)}',
                        'corrected_area': np.nan,
                        'isotopologue_fraction': float(fractions[i]),
                    })
    return pd.DataFrame(rows)


iso_g = load_iso_data(GLUCOSE_CORR_AREA_CSV, gluc_meta, 'GLUCOSE', True)
iso_x = load_iso_data(XYLOSE_ISO_FRAC_CSV, xyl_meta, 'XYLOSE', False)
iso_c = load_iso_data(CELLULOSE_ISO_FRAC_CSV, cell_meta, 'CELLULOSE', False)
iso_all = pd.concat([iso_g, iso_x, iso_c], ignore_index=True)

# Compute labeled fraction (1 - M+0) per sample × metabolite
m0 = iso_all[iso_all['isotopologue'] == 'M+0'].copy()
m0['labeled_fraction'] = 1.0 - m0['isotopologue_fraction']
iso_all = iso_all.merge(
    m0[['sample_id', 'metabolite', 'labeled_fraction']],
    on=['sample_id', 'metabolite'], how='left'
)


# ============================================================================
# TABLE S12 — Full isotopologue fraction data
# ============================================================================
print("\n[Table S12] Full isotopologue fractions...")
table_s12 = iso_all.copy()
table_s12.to_csv(os.path.join(TABLES_DIR, 'TableS12_isotopologue_fractions.tsv'),
                 sep='\t', index=False)
print(f"  {len(table_s12)} rows")


# ============================================================================
# TABLE S13 — Per-metabolite G/X bootstrap rate ratios
# ============================================================================
print("\n[Table S13] G/X bootstrap rate ratios per metabolite...")
np.random.seed(42)
N_BOOT = 10000

lab_long = iso_all[
    (iso_all['isotopologue'] == 'M+0') & (iso_all['timepoint_hours'] > 0)
].copy()
lab_long = lab_long.drop_duplicates(['sample_id', 'metabolite']).copy()

table_s13_rows = []
for met in TCA_METS + ['Asp']:
    g_data = lab_long[(lab_long['substrate'] == 'GLUCOSE') &
                      (lab_long['metabolite'] == met)]
    x_data = lab_long[(lab_long['substrate'] == 'XYLOSE') &
                      (lab_long['metabolite'] == met)]

    # Point estimate slopes
    g_avg = g_data.groupby('timepoint_hours')['labeled_fraction'].mean().reset_index()
    x_avg = x_data.groupby('timepoint_hours')['labeled_fraction'].mean().reset_index()
    if len(g_avg) >= 3 and len(x_avg) >= 3:
        g_slope = LinearRegression().fit(
            np.log10(g_avg['timepoint_hours'].values).reshape(-1, 1),
            g_avg['labeled_fraction'].values).coef_[0]
        x_slope = LinearRegression().fit(
            np.log10(x_avg['timepoint_hours'].values).reshape(-1, 1),
            x_avg['labeled_fraction'].values).coef_[0]
    else:
        g_slope = x_slope = np.nan

    # Bootstrap
    g_boot, x_boot = [], []
    for _ in range(N_BOOT):
        gv, xv = [], []
        for t in g_data['timepoint_hours'].unique():
            gt = g_data[g_data['timepoint_hours'] == t]['labeled_fraction'].values
            xt = x_data[x_data['timepoint_hours'] == t]['labeled_fraction'].values
            if len(gt) > 0:
                gv.append((t, np.mean(np.random.choice(gt, len(gt), replace=True))))
            if len(xt) > 0:
                xv.append((t, np.mean(np.random.choice(xt, len(xt), replace=True))))
        if len(gv) >= 3:
            gt2, gv2 = zip(*gv)
            g_boot.append(LinearRegression().fit(
                np.log10(np.array(gt2)).reshape(-1, 1),
                np.array(gv2)).coef_[0])
        if len(xv) >= 3:
            xt2, xv2 = zip(*xv)
            x_boot.append(LinearRegression().fit(
                np.log10(np.array(xt2)).reshape(-1, 1),
                np.array(xv2)).coef_[0])

    g_arr = np.array(g_boot)
    x_arr = np.array(x_boot)
    n_min = min(len(g_arr), len(x_arr))
    ratio_arr = g_arr[:n_min] / x_arr[:n_min]
    ratio_valid = ratio_arr[np.isfinite(ratio_arr) & (ratio_arr > 0) & (ratio_arr < 100)]

    table_s13_rows.append({
        'metabolite': met,
        'glucose_slope': round(g_slope, 4),
        'xylose_slope': round(x_slope, 4),
        'GX_ratio_median': round(np.median(ratio_valid), 3),
        'CI_lo_2.5pct': round(np.percentile(ratio_valid, 2.5), 3),
        'CI_hi_97.5pct': round(np.percentile(ratio_valid, 97.5), 3),
        'p_G_greater_X': round(np.mean(g_arr[:n_min] > x_arr[:n_min]), 4),
        'significant': 'yes' if np.percentile(ratio_valid, 2.5) > 1.0 else 'no',
    })
pd.DataFrame(table_s13_rows).to_csv(
    os.path.join(TABLES_DIR, 'TableS13_GX_bootstrap_rate_ratios.tsv'),
    sep='\t', index=False)
print(f"  {len(table_s13_rows)} metabolites")


# ============================================================================
# TABLE S14 — Per-treatment Pearson correlations: PC1 vs metabolite labeling
# ============================================================================
print("\n[Table S14] PC1 vs metabolite labeling correlations...")
pca_main_means = pca_main.groupby(['treatment', 'time_hours'])['PC1'].mean()

table_s14_rows = []
for met in ALL_METS:
    for treatment in MAIN_TREATMENTS:
        lab_avg = lab_long[
            (lab_long['metabolite'] == met) &
            (lab_long['substrate'] == treatment)
        ].groupby('timepoint_hours')['labeled_fraction'].mean()

        x_vals, y_vals = [], []
        for tp_h, lf in lab_avg.items():
            metat_t = TIME_MAPPING.get(tp_h)
            if metat_t is None:
                continue
            try:
                pc1_v = pca_main_means.loc[(treatment, float(metat_t))]
            except KeyError:
                continue
            x_vals.append(lf)
            y_vals.append(pc1_v)

        if len(x_vals) >= 4:
            r, p = pearsonr(x_vals, y_vals)
            rho, p_sp = spearmanr(x_vals, y_vals)
        else:
            r, p, rho, p_sp = np.nan, np.nan, np.nan, np.nan

        table_s14_rows.append({
            'metabolite': met,
            'treatment': treatment,
            'n_matched_timepoints': len(x_vals),
            'pearson_r': round(r, 4) if not np.isnan(r) else np.nan,
            'pearson_p': round(p, 4) if not np.isnan(p) else np.nan,
            'spearman_rho': round(rho, 4) if not np.isnan(rho) else np.nan,
            'spearman_p': round(p_sp, 4) if not np.isnan(p_sp) else np.nan,
            'significant': 'yes' if (not np.isnan(p) and p < 0.05) else 'no',
        })
pd.DataFrame(table_s14_rows).to_csv(
    os.path.join(TABLES_DIR, 'TableS14_PC1_metabolite_correlations.tsv'),
    sep='\t', index=False)
print(f"  {len(table_s14_rows)} correlations")


# ============================================================================
# TABLE S15 — CCF results
# ============================================================================
print("\n[Table S15] CCF results...")
# Simple raw CCF on TCA aggregate vs PC1 (matches Figure 3 dual-axis data)
tca_lab = lab_long[lab_long['metabolite'].isin(TCA_METS)].copy()

table_s15_rows = []
for treatment in MAIN_TREATMENTS:
    tca_avg = tca_lab[tca_lab['substrate'] == treatment].groupby(
        'timepoint_hours')['labeled_fraction'].mean()
    pc1_avg_t = pca_main[pca_main['treatment'] == treatment].groupby(
        'time_hours')['PC1'].mean()

    pairs_t, pairs_lab, pairs_pc1 = [], [], []
    for tp_h, lf in tca_avg.items():
        metat_t = TIME_MAPPING.get(tp_h)
        if metat_t is None or metat_t not in pc1_avg_t.index:
            continue
        pairs_t.append(tp_h)
        pairs_lab.append(lf)
        pairs_pc1.append(pc1_avg_t[metat_t])

    n = len(pairs_lab)
    lab_arr = np.array(pairs_lab)
    pc1_arr = np.array(pairs_pc1)

    for shift in range(-2, 3):
        if shift > 0:
            x, y = lab_arr[:n - shift], pc1_arr[shift:]
        elif shift < 0:
            x, y = pc1_arr[:n + shift], lab_arr[-shift:]
        else:
            x, y = lab_arr, pc1_arr
        if len(x) >= 3:
            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan
        table_s15_rows.append({
            'treatment': treatment,
            'shift': shift,
            'n_overlap': len(x),
            'pearson_r': round(r, 4) if not np.isnan(r) else np.nan,
            'pearson_p': round(p, 4) if not np.isnan(p) else np.nan,
            'r_squared': round(r ** 2, 4) if not np.isnan(r) else np.nan,
        })
pd.DataFrame(table_s15_rows).to_csv(
    os.path.join(TABLES_DIR, 'TableS15_CCF_results.tsv'),
    sep='\t', index=False)
print(f"  {len(table_s15_rows)} CCF rows")


# ============================================================================
# TABLE S16 — Bulk 13C recovery in soil
# ============================================================================
print("\n[Table S16] Bulk 13C recovery...")

def parse_ea(filepath, treatment_key):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    rows = []
    for _, row in df.iterrows():
        sid = row['Sample ID']
        m = re.search(r'_(\d+)hrs', sid)
        if not m:
            continue
        tp_h = int(m.group(1))
        rep_m = re.search(r'rep(\d+)', sid)
        rep = int(rep_m.group(1)) if rep_m else 1
        treatment = 'control' if 'control' in sid.lower() else treatment_key
        rows.append({
            'treatment': treatment, 'replicate': rep,
            'timepoint_hours': tp_h,
            'AT_pct_13C': float(row['AT%13']),
            'pct_C': float(row['%C']),
        })
    return pd.DataFrame(rows)

ea_all = pd.concat([
    parse_ea(EA_GLUCOSE_CSV, 'GLUCOSE'),
    parse_ea(EA_XYLOSE_CSV, 'XYLOSE'),
    parse_ea(EA_CELLULOSE_CSV, 'CELLULOSE'),
], ignore_index=True)
ea_all = ea_all.drop_duplicates()

NATURAL_AB = 1.09
SOIL_C = 200.0
C13_ADDED = {'GLUCOSE': 5.156, 'XYLOSE': 1.785, 'CELLULOSE': 8.889, 'control': 0}

ea_all['excess_AT13'] = ea_all['AT_pct_13C'] - NATURAL_AB
ea_all['excess_13C_mg'] = (ea_all['excess_AT13'] / 100.0) * SOIL_C
ea_all['mg_13C_added'] = ea_all['treatment'].map(C13_ADDED)
ea_all['pct_13C_recovered'] = ea_all.apply(
    lambda r: (r['excess_13C_mg'] / r['mg_13C_added'] * 100)
              if r['mg_13C_added'] > 0 else np.nan, axis=1)
ea_all['soil_C_pool_mg'] = SOIL_C

ea_all[['treatment', 'replicate', 'timepoint_hours',
        'AT_pct_13C', 'excess_AT13',
        'soil_C_pool_mg', 'mg_13C_added',
        'pct_13C_recovered']].to_csv(
    os.path.join(TABLES_DIR, 'TableS16_bulk_13C_recovery.tsv'),
    sep='\t', index=False)
print(f"  {len(ea_all)} EA measurements")


# ============================================================================
# Tables S17 (already generated by figure S12 script)
# Tables S18, S19, S20 — placeholders
# ============================================================================
print("\n[Tables S18-S20] Placeholders for pathway VST and qSIP...")
for tnum, name in [
    ('S18', 'pathway_VST_temporal_correlations'),
    ('S19', 'qSIP_per_taxon_13C_incorporation'),
    ('S20', 'qSIP_density_fractionation_summary'),
]:
    with open(os.path.join(TABLES_DIR, f'Table{tnum}_{name}_PLACEHOLDER.txt'),
              'w') as f:
        f.write(f"Table {tnum} placeholder.\n")
        if tnum == 'S18':
            f.write("Pathway-level VST expression and Spearman temporal correlations\n")
            f.write("for the 7 pathway groups in Figure S15. Generate from\n")
            f.write("notebook 06/07 functional analysis outputs.\n")
        else:
            f.write("Awaiting qSIP-DNA results.\n")

print("\nDone.")
