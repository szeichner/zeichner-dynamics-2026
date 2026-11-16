#!/usr/bin/env python3
"""
Renumber supplemental figures to match the final manuscript order.

Maps source stems (script output filenames) to canonical S-numbers.
Copies source -> FigureS{N}.{pdf,png} in figures/output/.

21 figures (S1-S21): Methods/standalone -> Fig 1 (qSIP) -> Fig 2 (PCoA)
    -> Fig 3 (programs) -> Fig 4 (carbon fate) -> Fig 5 (coupling)

Matches Zeichner_Dynamics_Supplement_v14.docx ordering.

Run this AFTER regenerating any figures that need fixes.
"""

import os
import shutil

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
FIG_DIR = os.path.join(BASE, 'figures', 'output')
METAB_DIR = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /data/Metabolomics/'

# ============================================================================
# MASTER MAPPING: S-number -> (source filename stem, description)
# Matches Zeichner_Dynamics_Supplement_v14.docx
# ============================================================================

MAPPING = [
    # --- Methods / standalone (S1-S3) ---
    ('S1',  'FigureS1_metaT_QC',                        'MetaT QC summary'),
    ('S2',  'FigureS8_EA_13C_data',                      'Bulk soil 13C enrichment (EA-IRMS)'),
    ('S3',  'FigureS9_CO2_respiration',                   'CO2 respiration dynamics'),

    # --- Supports Fig 1: qSIP (S4-S7) ---
    ('S4',  'FigureS_weighted_vs_unweighted',             'Weighted vs unweighted mean EAF'),
    ('S5',  'FigureS_abundance_bin_EAF',                  'Abundance-bin-resolved EAF by treatment'),
    ('S6',  'FigureS3_qSIP_phylum_heatmaps_color',       'Phylum-level EAF heatmaps (color)'),
    ('S7',  'FigureS_qsip_genus_heatmaps',               'Genus-level EAF heatmaps'),

    # --- Supports Fig 2: PCoA (S8-S10) ---
    ('S8',  'FigureS2_NMDS',                              'NMDS ordination'),
    ('S9',  'FigureS_multivariate_summary',               'PERMANOVA treatment effects'),
    ('S10', 'FigureS_pairwise_permanova_heatmap',         'Pairwise PERMANOVA heatmap'),

    # --- Supports Fig 3: programs (S11-S17) ---
    ('S11', 'FigureS_k_selection_trajectories',           'K-means cluster selection + trajectories'),
    ('S12', 'FigureS_k5_functional_bubble',               'k=5 functional bubble heatmap'),
    ('S13', 'FigureS12_functional_category_temporal',     'Functional-category transcript allocation'),
    ('S14', 'FigureS13_CAZyme_combined',                  'CAZyme family dynamics (monomers pooled)'),
    ('S15', 'query_genes_bubble_P1',                      'P1 query-gene bubble heatmap'),
    ('S16', 'query_genes_bubble_P2',                      'P2 query-gene bubble heatmap'),
    ('S17', 'query_genes_bubble_P3',                      'P3 query-gene bubble heatmap'),

    # --- Supports Fig 4: carbon fate (S18-S20) ---
    # S18: calibration curves (handled separately via METAB_MAPPING)
    ('S19', 'FigureS_metabolite_concentrations_combined', 'Pore-water metabolite concentrations'),
    ('S20', 'FigureS6_aggregate_TCA_labeling',            'Aggregate TCA-cycle fractional labeling'),

    # --- Supports Fig 5: coupling (S21) ---
    ('S21', 'FigureS7_program_metabolite_heatmap',        'Program-metabolite correlation heatmap'),
]

# Metabolomics notebook figures (different source directory)
METAB_MAPPING = [
    ('S18', os.path.join(METAB_DIR, 'converter_plots', 'all_converters_grid'), 'Calibration curves', 'png'),
]

STEM_DIR = os.path.join(FIG_DIR, 'deprecated', 'source_stems')
DEPRECATED_DIR = os.path.join(FIG_DIR, 'deprecated')


def copy_figure(src_stem, dst_stem, fig_dir, extensions=('pdf', 'png')):
    """Copy src_stem.{ext} -> dst_stem.{ext}. Checks fig_dir, source_stems/, and deprecated/."""
    copied = []
    for ext in extensions:
        src = os.path.join(fig_dir, f'{src_stem}.{ext}')
        if not os.path.exists(src):
            src = os.path.join(STEM_DIR, f'{src_stem}.{ext}')
        if not os.path.exists(src):
            src = os.path.join(DEPRECATED_DIR, f'{src_stem}.{ext}')
        dst = os.path.join(fig_dir, f'{dst_stem}.{ext}')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(ext)
        else:
            print(f'  WARNING: {src_stem}.{ext} not found in output/, source_stems/, or deprecated/')
    return copied


def main():
    print("=" * 60)
    print("Renumbering supplemental figures (S1-S21)")
    print("=" * 60)

    os.makedirs(DEPRECATED_DIR, exist_ok=True)

    # Check all sources exist
    print("\nChecking source files...")
    missing = []
    for new_s, src_stem, desc in MAPPING:
        for ext in ('pdf', 'png'):
            src = os.path.join(FIG_DIR, f'{src_stem}.{ext}')
            src_alt = os.path.join(STEM_DIR, f'{src_stem}.{ext}')
            src_dep = os.path.join(DEPRECATED_DIR, f'{src_stem}.{ext}')
            if not os.path.exists(src) and not os.path.exists(src_alt) and not os.path.exists(src_dep):
                missing.append((new_s, f'{src_stem}.{ext}', desc))
    for new_s, src_path, desc, ext in METAB_MAPPING:
        src = f'{src_path}.{ext}'
        if not os.path.exists(src):
            missing.append((new_s, src, desc))

    if missing:
        print("\nMissing source files:")
        for s, path, desc in missing:
            print(f'  {s}: {path} ({desc})')
        print("\nProceeding with available files...\n")
    else:
        print("  All source files found.\n")

    # Copy with new names
    print("Copying figures to new S-numbers...")
    for new_s, src_stem, desc in MAPPING:
        dst_stem = f'Figure{new_s}'
        exts = copy_figure(src_stem, dst_stem, FIG_DIR)
        status = f'[{",".join(exts)}]' if exts else '[MISSING]'
        print(f'  {new_s}: {src_stem} -> {dst_stem}  {status}  ({desc})')

    # Metabolomics notebook figures
    print("\nCopying metabolomics notebook figures...")
    for new_s, src_path, desc, src_ext in METAB_MAPPING:
        dst_stem = f'Figure{new_s}'
        src = f'{src_path}.{src_ext}'
        if os.path.exists(src):
            dst = os.path.join(FIG_DIR, f'{dst_stem}.{src_ext}')
            shutil.copy2(src, dst)
            print(f'  {new_s}: {os.path.basename(src)} -> {dst_stem}.{src_ext}  ({desc})')
        else:
            print(f'  {new_s}: {src} NOT FOUND  ({desc})')

    print("\n" + "=" * 60)
    print("Done. Total: 21 supplemental figures (S1-S21)")
    print("Verify with: ls figures/output/FigureS*.{pdf,png} | sort")
    print("=" * 60)


if __name__ == '__main__':
    main()
