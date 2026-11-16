#!/usr/bin/env python3
"""
Bundle every supplementary table into a single multi-sheet Excel workbook
for journal submission (v6 numbering).

Output: tables/output/compiled_for_publication/
  - Zeichner_et_al_supplementary_tables.xlsx  (one tab per table)
  - TableS4_DESeq2_full_results.tsv.gz  (separate, too large for Excel)

Tab naming: "S<num>_<short_descriptor>", capped at Excel's 31-char
sheet-name limit.
"""

import os
import re
import shutil
import gzip
import pandas as pd

BASE_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
SRC_DIR = os.path.join(BASE_DIR, 'tables', 'output')
DERIVED_DIR = os.path.join(BASE_DIR, 'data', 'derived')
OUT_DIR = os.path.join(SRC_DIR, 'compiled_for_publication')
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Tab order, source filenames, and short descriptors (sheet names ≤31 chars)
# ----------------------------------------------------------------------------
# (sheet_name, source_path, format)  format = 'tsv' | 'tsv.gz' | 'multisection'
TABLE_SPECS = [
    ('S1_sample_metadata',          os.path.join(SRC_DIR, 'TableS1_sample_metadata_QC.tsv'),         'tsv'),
    ('S2_soil_amendment',           os.path.join(SRC_DIR, 'TableS2_soil_amendment_summary.tsv'),     'tsv'),
    ('S3_NMDS_coordinates',         os.path.join(SRC_DIR, 'TableS3_NMDS_coordinates.tsv'),           'tsv'),
    # S4 excluded — full DESeq2 results (227K rows), exported as separate .tsv.gz
    ('S5_pathway_CAZy_DESeq2',      os.path.join(SRC_DIR, 'TableS5_pathway_CAZy_aggregated_DESeq2.tsv'), 'tsv'),
    ('S6_CAZy_broad_group',         os.path.join(SRC_DIR, 'TableS6_CAZy_broad_group_DESeq2.tsv'),   'tsv'),
    ('S7_CAZy_family_DESeq2',       os.path.join(SRC_DIR, 'TableS7_CAZy_family_DESeq2.tsv'),        'tsv'),
    ('S8_program_assignments_k3',   os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'),     'tsv'),
    ('S8b_program_assignments_k5',  os.path.join(DERIVED_DIR, 'kmeans_program_assignments_k5.tsv'),  'tsv'),
    ('S9_subcategory_assignments',  os.path.join(SRC_DIR, 'TableS9_program_subcategory_assignments.tsv'), 'tsv'),
    ('S10_isotopologue_fractions',  os.path.join(SRC_DIR, 'TableS10_isotopologue_fractions.tsv'),    'tsv'),
    ('S11_GX_bootstrap_ratios',     os.path.join(SRC_DIR, 'TableS11_GX_bootstrap_rate_ratios.tsv'), 'tsv'),
    ('S12_CCF_results',             os.path.join(SRC_DIR, 'TableS12_CCF_results.tsv'),               'tsv'),
    ('S13_bulk_13C_recovery',       os.path.join(SRC_DIR, 'TableS13_bulk_13C_recovery.tsv'),         'tsv'),
    ('S14_func_cat_similarity',     os.path.join(SRC_DIR, 'TableS14_functional_category_similarity.tsv'), 'tsv'),
    ('S15_pentose_hexose',          os.path.join(SRC_DIR, 'TableS15_pentose_hexose_catabolic_investment.tsv'), 'multisection'),
    ('S16_pairwise_DE_summary',     os.path.join(SRC_DIR, 'TableS16_pairwise_DE_summary.tsv'),       'tsv'),
    ('S17_bench_notes',             os.path.join(SRC_DIR, 'TableS17_bench_notes.tsv'),               'tsv'),
    # Derived data tables
    ('SD1_BCPCoA_coords',           os.path.join(DERIVED_DIR, 'bc_pcoa_coords.tsv'),                 'tsv'),
    ('SD2_BCPCoA_eigenvalues',      os.path.join(DERIVED_DIR, 'bc_pcoa_eigenvalues.tsv'),            'tsv'),
    ('SD3_BC_dist_from_baseline',   os.path.join(DERIVED_DIR, 'bc_distance_from_baseline.tsv'),      'tsv'),
    ('SD4_spline_LRT_results',      os.path.join(DERIVED_DIR, 'spline_lrt_results_R.tsv'),           'tsv'),
    ('SD5_kmeans_k_selection',      os.path.join(DERIVED_DIR, 'kmeans_k_selection.tsv'),              'tsv'),
    ('SD6_domain_classification',   os.path.join(DERIVED_DIR, 'ko_domain_classification.tsv'),        'tsv'),
    ('SD7_prog_metab_spearman',     os.path.join(DERIVED_DIR, 'program_metabolite_spearman.tsv'),     'tsv'),
    ('SD8_PERMANOVA_per_timepoint', os.path.join(DERIVED_DIR, 'permanova_per_timepoint.tsv'),         'tsv'),
    ('SD9_PERMANOVA_all_pairwise',  os.path.join(DERIVED_DIR, 'permanova_all_pairwise.tsv'),          'tsv'),
    ('SD10_centroid_distances',     os.path.join(DERIVED_DIR, 'centroid_distances_over_time.tsv'),     'tsv'),
]

# Sanity-check sheet name lengths
for name, _, _ in TABLE_SPECS:
    assert len(name) <= 31, f"Sheet name '{name}' exceeds 31 chars ({len(name)})"


def load_tsv(path):
    return pd.read_csv(path, sep='\t')


def load_tsv_gz(path):
    return pd.read_csv(path, sep='\t', compression='gzip')


def load_multisection(path):
    """Parse section-delimited format into a single stacked DataFrame."""
    sections = []
    current_header = None
    current_lines = []

    def flush():
        if current_header is None or not current_lines:
            return
        from io import StringIO
        df = pd.read_csv(StringIO(''.join(current_lines)), sep='\t')
        sections.append((current_header, df))

    with open(path) as fh:
        for line in fh:
            if line.startswith('## '):
                flush()
                current_header = line[3:].rstrip('\n')
                current_lines = []
            elif line.startswith('#') or not line.strip():
                continue
            else:
                current_lines.append(line)
        flush()

    if not sections:
        return load_tsv(path)

    out_rows = []
    max_cols = max(len(df.columns) for _, df in sections)
    for header, df in sections:
        header_row = [header] + [''] * (max_cols - 1)
        out_rows.append(header_row)
        col_row = list(df.columns) + [''] * (max_cols - len(df.columns))
        out_rows.append(col_row)
        for _, r in df.iterrows():
            row = list(r.values) + [''] * (max_cols - len(df.columns))
            out_rows.append(row)
        out_rows.append([''] * max_cols)

    return pd.DataFrame(out_rows)


LOADERS = {
    'tsv': load_tsv,
    'tsv.gz': load_tsv_gz,
    'multisection': load_multisection,
}


# ----------------------------------------------------------------------------
# Build the workbook
# ----------------------------------------------------------------------------
xlsx_path = os.path.join(OUT_DIR, 'Zeichner_et_al_supplementary_tables.xlsx')
print(f"Writing {xlsx_path}")

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    # Cover sheet
    cover_rows = [
        ['Zeichner et al. — Supplementary Tables'],
        ['Manuscript: Substrate physical form drives qualitatively distinct '
         'soil microbial community trajectories during sugar processing'],
        [''],
        ['Tab', 'Description', 'Source file', 'Notes'],
    ]
    for sheet_name, src, fmt in TABLE_SPECS:
        desc = re.sub(r'^S\d+[a-z]?_', '', sheet_name).replace('_', ' ')
        desc = re.sub(r'^D\d+_', '', desc).replace('_', ' ')
        note = ''
        if fmt == 'multisection':
            note = 'Multi-section table; section headers are inline'
        cover_rows.append([sheet_name, desc, os.path.basename(src), note])
    cover_rows.append([''])
    cover_rows.append(['NOTE', 'Table S4 (full DESeq2 differential abundance, '
                       '227,600 rows) is provided as a separate compressed TSV: '
                       'TableS4_DESeq2_full_results.tsv.gz'])
    cover_df = pd.DataFrame(cover_rows)
    cover_df.to_excel(writer, sheet_name='_README', index=False, header=False)

    for sheet_name, src, fmt in TABLE_SPECS:
        if not os.path.exists(src):
            print(f"  WARNING: missing source for {sheet_name}: {src}")
            continue
        df = LOADERS[fmt](src)
        df.to_excel(writer, sheet_name=sheet_name, index=False,
                    header=(fmt != 'multisection'))
        print(f"  {sheet_name}: {df.shape[0]} rows x {df.shape[1]} cols  ({fmt})")

# ----------------------------------------------------------------------------
# Copy Table S4 .tsv.gz alongside the workbook
# ----------------------------------------------------------------------------
s4_src = os.path.join(SRC_DIR, 'TableS4_DESeq2_full_results.tsv.gz')
s4_dst = os.path.join(OUT_DIR, 'TableS4_DESeq2_full_results.tsv.gz')
if os.path.exists(s4_src):
    shutil.copy2(s4_src, s4_dst)
    print(f"Copied {os.path.basename(s4_src)} -> {OUT_DIR}")
    with gzip.open(s4_dst, 'rt') as fh:
        n_lines = sum(1 for _ in fh)
    print(f"  TableS4: {n_lines:,} lines ({os.path.getsize(s4_dst) / 1e6:.1f} MB)")

print(f"\nDone. Bundle written to {OUT_DIR}")
