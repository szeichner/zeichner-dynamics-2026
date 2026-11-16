# Code and Data Archive: Substrate physical form drives qualitatively distinct soil microbial community trajectories during sugar processing

[![DOI](https://zenodo.org/badge/DOI/[DOI_PLACEHOLDER].svg)](https://doi.org/[DOI_PLACEHOLDER])

## Authors

Sarah S. Zeichner, Norah Chen, Katerina Estera-Molina, Mike Allen, Jessica R. Wollard, Steven J. Blazewicz, Ludmilla Aristilde, Patrick M. Shih, Jennifer Pett-Ridge

## Summary

This repository contains all code and data needed to reproduce the analyses and figures in Zeichner et al. "Substrate physical form drives qualitatively distinct soil microbial community trajectories during sugar processing."

Soil microcosms were amended with 13C-labeled glucose+xylose (monomers) or 13C-labeled hemicellulose (polymer) and sampled over 28 days. Metatranscriptomic, quantitative stable isotope probing (qSIP), extracellular pore water metabolomic, and 13C-CO2 respiration data were integrated to show that substrate physical form induces qualitatively distinct microbial community trajectories, not merely different rates along a common successional path.

## Repository Structure

```
zenodo_archive/
├── README.md                          # This file
├── LICENSE                            # MIT (code) + CC-BY-4.0 (data)
├── scripts/
│   ├── analysis/                      # Core analysis scripts (Python)
│   │   ├── compute_bc_pcoa.py         # Bray-Curtis PCoA + distance from baseline
│   │   ├── compute_multivariate_treatment_tests.py  # PERMANOVA per timepoint
│   │   ├── compute_pairwise_permanova_all.py        # All 171 pairwise PERMANOVA
│   │   ├── compute_program_metabolite_correlations_v2.py  # Program-metabolite correlations
│   │   ├── compute_co2_rate_bootstrap.py            # CO2 rate bootstrap
│   │   ├── compute_fractional_labeling.py           # Metabolite fractional labeling
│   │   ├── compute_normalized_bootstrap_GX_ratios.py  # G/X rate ratio bootstrap
│   │   ├── generate_figure_subprograms_from_lrt.py  # Subcategory assignment
│   │   ├── generate_tableSX_pairwise_DE.py          # Pairwise DE summary
│   │   └── generate_DESeq2_tables.py                # DESeq2 table generation
│   ├── figures/                       # Figure generation scripts (Python)
│   │   ├── generate_figure1_combined_logx.py        # Figure 1 (qSIP)
│   │   ├── generate_figure2_pcoa_standalone.py      # Figure 2 (PCoA)
│   │   ├── generate_figure3_programs_bubble_logx.py # Figure 3 (programs)
│   │   ├── generate_figure4_v8_mockup_logx.py       # Figure 4 C-L (metabolites)
│   │   ├── generate_figure4K_co2_rate.py            # Figure 4 B (CO2 rate)
│   │   ├── generate_figure5_multiomics_v10_logx.py  # Figure 5 (coupling)
│   │   ├── generate_figureS*.py                     # Supplementary figure scripts
│   │   ├── renumber_supplemental_figures.py          # S-figure renumbering pipeline
│   │   └── compile_supplementary_tables_xlsx.py      # Table compilation
│   └── R/                             # R scripts
│       ├── run_spline_lrt_vst.R       # DESeq2 spline-LRT + VST normalization
│       ├── run_qsip2.R               # qSIP2 analysis (16S)
│       ├── run_qsip2_ITS.R           # qSIP2 analysis (ITS)
│       ├── run_qsip2_per_timepoint.R  # qSIP per-timepoint analysis
│       ├── run_qsip2_per_treatment_timepoint.R  # qSIP per-treatment-timepoint
│       └── process_gas_bench_data_rates.R  # CO2 gas bench data processing
├── data/
│   ├── metatranscriptomics/           # Metatranscriptomic data
│   │   ├── Ko_clustering/             # IMG/M KO count matrices
│   │   ├── IMG_metadata.tsv           # Sample-to-IMG ID mapping
│   │   ├── cazy/                      # CAZyme family counts
│   │   └── qc/                        # QC metadata and reports
│   ├── qSIP/
│   │   ├── input/                     # Raw qSIP input (16S, ITS, density/qPCR)
│   │   └── output/                    # qSIP summary tables (<50 MB)
│   ├── metabolomics/                  # HPLC-Orbitrap metabolomics (all treatments)
│   ├── co2/                           # 13C-CO2 respiration data
│   ├── ea/                            # EA-IRMS bulk soil 13C data
│   ├── derived/                       # Computed intermediate files
│   ├── carbon_addition.csv            # Amendment carbon amounts
│   └── Cellulose soil degradation experiment SZ - Sample notes.csv
├── tables/                            # Supplementary tables (TSV + compiled XLSX)
│   └── Zeichner_et_al_supplementary_tables.xlsx  # 27-sheet compiled workbook
├── figures/
│   ├── main/                          # Main text figures 1-5 (PDF + PNG)
│   └── supplementary/                 # Supplementary figures S1-S21 (PDF + PNG)
└── environment/
    ├── requirements.txt               # Python package versions (pip freeze)
    └── r_packages.txt                 # R package versions
```

## Reproducing the Analyses

### Software Requirements

- **Python** >= 3.10 with packages listed in `environment/requirements.txt`
  - Key packages: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pydeseq2, statsmodels
- **R** >= 4.5 with packages listed in `environment/r_packages.txt`
  - Key packages: DESeq2, qSIP2, vegan, phyloseq

### Analysis Pipeline

1. **DESeq2 differential expression** (R): Run `scripts/R/run_spline_lrt_vst.R` to perform spline-LRT testing and VST normalization on KO count data.

2. **qSIP growth quantification** (R): Run `scripts/R/run_qsip2.R` and `scripts/R/run_qsip2_ITS.R` for 16S and ITS excess atom fraction (EAF) estimation.

3. **Ordination and multivariate statistics** (Python): Run `scripts/analysis/compute_bc_pcoa.py` for Bray-Curtis PCoA, then `scripts/analysis/compute_multivariate_treatment_tests.py` and `scripts/analysis/compute_pairwise_permanova_all.py` for PERMANOVA.

4. **Transcriptional program identification** (Python): K-means clustering assignments are in `data/derived/kmeans_program_assignments.tsv`. Subcategory assignment via `scripts/analysis/generate_figure_subprograms_from_lrt.py`.

5. **Metabolite labeling** (Python): Run `scripts/analysis/compute_fractional_labeling.py` for isotopologue fractions, then `scripts/analysis/compute_normalized_bootstrap_GX_ratios.py` for G/X rate ratios.

6. **CO2 rates** (Python/R): Run `scripts/R/process_gas_bench_data_rates.R` and `scripts/analysis/compute_co2_rate_bootstrap.py`.

7. **Program-metabolite correlations** (Python): Run `scripts/analysis/compute_program_metabolite_correlations_v2.py`.

### Generating Figures

Main figure scripts are in `scripts/figures/`. Each `generate_figure*.py` script reads from `data/` and `data/derived/` and outputs to `figures/`. Figure 4 Panel A (pathway schematic) and Figure 5 Panels D-E (conceptual models) were created manually in Affinity Designer.

## Data Description

- **Metatranscriptomics**: IMG/M-annotated KO-level transcript counts from 74 soil metatranscriptome samples (78 sequenced, 4 excluded by QC). Treatments: glucose+xylose monomers, hemicellulose polymer, unamended control. Timepoints: 0, 2, 8, 24, 72, 168, 336, 672 hours.
- **qSIP**: Quantitative stable isotope probing via 13C-DNA density fractionation with 16S rRNA gene and ITS amplicon sequencing. Provides taxon-resolved excess atom fraction (EAF) of 13C incorporation.
- **Metabolomics**: Extracellular pore water metabolite concentrations and 13C isotopologue distributions measured by HPLC-Orbitrap MS. 10 metabolites: pyruvate, lactate, citrate, alpha-ketoglutarate, succinate, fumarate, malate, aspartate, glutamate, N-acetylglutamate.
- **CO2**: delta-13C-CO2 and total CO2 flux from headspace gas sampling (gas bench IRMS).
- **EA-IRMS**: Bulk soil 13C enrichment from elemental analyzer isotope ratio mass spectrometry.

## Large Files Not in GitHub

The following files exceed GitHub's 100 MB limit and are available only via the Zenodo archive:

- `data/qSIP/output/qsip_16S_growth_results.tsv` (935 MB)
- `data/qSIP/output/qsip_16S_EAF_results.tsv` (151 MB)
- `data/qSIP/output/qsip_ITS_growth_results.tsv` (254 MB)

The corresponding RData files (`qsip2_16S_complete.RData`, 1.3 GB; `qsip2_ITS_complete.RData`, 217 MB) are not included but can be regenerated by running the qSIP R scripts.

## Citation

If you use this code or data, please cite:

> Zeichner, S.S., Chen, N., Estera-Molina, K., Allen, M., Wollard, J.R., Blazewicz, S.J., Aristilde, L., Shih, P.M., & Pett-Ridge, J. Substrate physical form drives qualitatively distinct soil microbial community trajectories during sugar processing. [DOI_PLACEHOLDER]

## License

Code is released under the MIT License. Data and figures are released under CC-BY-4.0. See [LICENSE](LICENSE) for details.
