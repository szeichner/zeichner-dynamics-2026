#!/usr/bin/env python3
"""
compute_program_metabolite_correlations_v2.py
==============================================

Two approaches to program-metabolite correlations:

Approach 1: Per-KO correlations summarized per program.
  For each KO in a program, correlate its LFC trajectory with each metabolite's
  labeled fraction trajectory (Spearman, n=6-7 timepoints). Then report per-program:
  median rho, mean rho, fraction of KOs with rho > 0, fraction significant (p < 0.05).

Approach 2: Spline-fitted smooth trajectories.
  Use the VST data to compute smooth program trajectories via natural cubic spline
  interpolation, then correlate with metabolite trajectories at matched timepoints.

Outputs:
  - data/derived/program_metabolite_perKO_summary.tsv (Approach 1)
  - data/derived/program_metabolite_perKO_distributions.tsv (per-KO detail)
  - data/derived/program_metabolite_spline_corr.tsv (Approach 2)
"""

import os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.interpolate import CubicSpline
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}
METABOLITES = ['PYR', 'LAT', 'Cit', 'aKG', 'Suc', 'Fum', 'Mal', 'Asp', 'GLUM', 'AGLUM', 'ASR', 'NAA']
METAB_NAMES = {
    'PYR': 'Pyruvate', 'LAT': 'Lactate',
    'Cit': 'Citrate', 'aKG': 'α-KG', 'Suc': 'Succinate',
    'Fum': 'Fumarate', 'Mal': 'Malate', 'Asp': 'Aspartate',
    'GLUM': 'Glutamate', 'AGLUM': 'N-AcGlut', 'ASR': 'O-Ac-Serine', 'NAA': 'N-Ac-Alanine',
}

TREATMENTS = ['GLUCOSE', 'XYLOSE', 'CELLULOSE']
TREAT_TO_SG = {'GLUCOSE': 'MONOMER', 'XYLOSE': 'MONOMER', 'CELLULOSE': 'HEMICELLULOSE'}
TREAT_FILES = {
    'GLUCOSE': ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv', 'GlucoseMetadata.csv'),
    'XYLOSE': ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'CelluloseMetadata.csv'),
}

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")

# Program assignments with per-condition LFC
assign = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')
lfc_cols = [c for c in assign.columns if c.startswith('MONOMER_') or c.startswith('HEMICELLULOSE_')]

# VST matrix for spline fitting
vst = pd.read_csv(os.path.join(DERIVED_DIR, 'vst_counts_R.tsv.gz'), sep='\t', index_col=0)
vst.index = vst.index.astype(str)

# Metadata
meta = pd.read_csv(os.path.join(BASE, 'data', 'metatranscriptomics', 'img data', 'IMG_metadata.tsv'), sep='\t')
meta['taxon_id'] = meta['taxon_id'].astype(str)
excluded = ['sample20_GLUCOSE_xylose_168hrs_rep4',
            'sample25_glucose_XYLOSE_2hrs_rep1',
            'sample71_CELLULOSE_336hrs_rep3']
meta = meta[~meta.sample_name.isin(excluded)].reset_index(drop=True)
meta['substrate_group'] = np.where(
    meta.treatment.isin(['GLUCOSE', 'XYLOSE']), 'MONOMER',
    np.where(meta.treatment == 'CELLULOSE', 'HEMICELLULOSE', 'control'))

# Control mean for LFC computation
ctrl_ids = meta[meta.treatment == 'control'].taxon_id.values
ctrl_mean_vst = vst.loc[ctrl_ids].mean(axis=0)

# Metabolite labeled fractions
def load_metab_means(treatment):
    frac_file, meta_file = TREAT_FILES[treatment]
    frac = pd.read_csv(os.path.join(METAB_DIR, frac_file))
    meta_m = pd.read_csv(os.path.join(METAB_DIR, meta_file))
    sample_cols = [c for c in frac.columns if c not in ['metabolite', 'isotopologue']]
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta_m.iterrows()}
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

print(f"  {len(assign)} KOs in {len(PROG_ORDER)} programs")
print(f"  Metabolites per treatment: " +
      ", ".join(f"{t}={len(metab_data[t])}" for t in TREATMENTS))

# ============================================================================
# Approach 1: Per-KO correlations
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 1: Per-KO Spearman correlations")
print("=" * 70)

per_ko_rows = []

for prog in PROG_ORDER:
    prog_kos = assign[assign.mnemonic == prog].ko_id.tolist()

    for treat in TREATMENTS:
        sg = TREAT_TO_SG[treat]

        for metab in METABOLITES:
            if metab not in metab_data[treat]:
                continue
            md = metab_data[treat][metab]
            metab_tps = sorted(md.keys())

            # Build per-KO LFC at matched timepoints
            matched_cols = []
            metab_vals = []
            for tp in metab_tps:
                col = f'{sg}_t{tp}'
                if col in lfc_cols:
                    matched_cols.append(col)
                    metab_vals.append(md[tp])

            if len(matched_cols) < 4:
                continue

            metab_arr = np.array(metab_vals)

            for ko in prog_kos:
                ko_row = assign[assign.ko_id == ko]
                if len(ko_row) == 0:
                    continue
                ko_lfc = ko_row[matched_cols].values.flatten()

                if np.all(ko_lfc == ko_lfc[0]):  # constant — skip
                    continue

                rho, p = spearmanr(ko_lfc, metab_arr)
                per_ko_rows.append({
                    'program': prog, 'ko_id': ko, 'metabolite': metab,
                    'treatment': treat, 'spearman_rho': rho, 'p_value': p,
                    'n_timepoints': len(matched_cols),
                })

per_ko_df = pd.DataFrame(per_ko_rows)
print(f"  {len(per_ko_df)} per-KO correlations computed")

# Save per-KO detail
per_ko_df.to_csv(os.path.join(DERIVED_DIR, 'program_metabolite_perKO_distributions.tsv'),
                 sep='\t', index=False)

# Summarize per (program, metabolite, treatment)
summary_rows = []
for prog in PROG_ORDER:
    for metab in METABOLITES:
        for treat in TREATMENTS:
            sub = per_ko_df[(per_ko_df.program == prog) &
                            (per_ko_df.metabolite == metab) &
                            (per_ko_df.treatment == treat)]
            if len(sub) < 5:
                continue

            rhos = sub.spearman_rho.values
            pvals = sub.p_value.values
            n_ko = len(sub)
            n_pos = (rhos > 0).sum()
            n_neg = (rhos < 0).sum()
            n_sig_pos = ((rhos > 0) & (pvals < 0.05)).sum()
            n_sig_neg = ((rhos < 0) & (pvals < 0.05)).sum()

            summary_rows.append({
                'program': prog, 'metabolite': metab,
                'metabolite_name': METAB_NAMES.get(metab, metab),
                'treatment': treat,
                'n_kos': n_ko,
                'median_rho': np.median(rhos),
                'mean_rho': np.mean(rhos),
                'sd_rho': np.std(rhos),
                'frac_positive': n_pos / n_ko,
                'frac_negative': n_neg / n_ko,
                'n_sig_positive': n_sig_pos,
                'n_sig_negative': n_sig_neg,
                'frac_sig_positive': n_sig_pos / n_ko,
                'frac_sig_negative': n_sig_neg / n_ko,
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(DERIVED_DIR, 'program_metabolite_perKO_summary.tsv'),
                  sep='\t', index=False)
print(f"  {len(summary_df)} program-metabolite-treatment summaries")

# Report top results
print("\n  Top positive median correlations (median rho > 0.3):")
top_pos = summary_df[summary_df.median_rho > 0.3].sort_values('median_rho', ascending=False)
for _, r in top_pos.head(15).iterrows():
    print(f"    {r.program:20s} × {r.metabolite_name:14s} [{r.treatment:10s}] "
          f"median_ρ={r.median_rho:+.3f}  {r.frac_positive:.0%} pos  "
          f"{r.n_sig_positive}/{r.n_kos} sig+ (n={r.n_kos})")

print("\n  Top negative median correlations (median rho < -0.3):")
top_neg = summary_df[summary_df.median_rho < -0.3].sort_values('median_rho')
for _, r in top_neg.head(15).iterrows():
    print(f"    {r.program:20s} × {r.metabolite_name:14s} [{r.treatment:10s}] "
          f"median_ρ={r.median_rho:+.3f}  {r.frac_negative:.0%} neg  "
          f"{r.n_sig_negative}/{r.n_kos} sig- (n={r.n_kos})")

# ============================================================================
# Approach 2: Spline-fitted trajectories
# ============================================================================
print("\n" + "=" * 70)
print("APPROACH 2: Spline-fitted program trajectories")
print("=" * 70)

# For each program, fit a cubic spline to the mean LFC trajectory per treatment,
# then evaluate at matched metabolite timepoints and correlate.

spline_rows = []

for prog in PROG_ORDER:
    prog_kos = assign[assign.mnemonic == prog].ko_id.tolist()
    # Filter to KOs present in VST
    prog_kos_in_vst = [k for k in prog_kos if k in vst.columns]

    for treat in TREATMENTS:
        sg = TREAT_TO_SG[treat]

        # Get timepoints for this substrate group
        sg_mask = meta.substrate_group == sg
        sg_tps = sorted(meta[sg_mask].time_hours.unique())
        if 0 in sg_tps:
            sg_tps.remove(0)

        # Compute mean LFC per timepoint for this program
        tp_lfc_means = []
        tp_hours = []
        for tp in sg_tps:
            tp_ids = meta[(meta.substrate_group == sg) & (meta.time_hours == tp)].taxon_id.values
            if len(tp_ids) == 0:
                continue
            mean_vst_tp = vst.loc[tp_ids, prog_kos_in_vst].mean(axis=0)
            mean_lfc = (mean_vst_tp - ctrl_mean_vst[prog_kos_in_vst]).mean()
            tp_lfc_means.append(mean_lfc)
            tp_hours.append(tp)

        if len(tp_hours) < 4:
            continue

        # Fit cubic spline to log(time) vs LFC
        log_t = np.log(np.array(tp_hours, dtype=float))
        lfc_arr = np.array(tp_lfc_means)

        try:
            cs = CubicSpline(log_t, lfc_arr, bc_type='natural')
        except Exception:
            continue

        # Evaluate spline at metabolite timepoints
        for metab in METABOLITES:
            if metab not in metab_data[treat]:
                continue
            md = metab_data[treat][metab]
            metab_tps = sorted(md.keys())

            # Only use timepoints within the spline's fitted range
            min_t, max_t = tp_hours[0], tp_hours[-1]
            valid_tps = [tp for tp in metab_tps if min_t <= tp <= max_t]

            if len(valid_tps) < 4:
                continue

            spline_vals = cs(np.log(np.array(valid_tps, dtype=float)))
            metab_vals = [md[tp] for tp in valid_tps]

            rho, p = spearmanr(spline_vals, metab_vals)

            spline_rows.append({
                'program': prog, 'metabolite': metab,
                'metabolite_name': METAB_NAMES.get(metab, metab),
                'treatment': treat,
                'n_timepoints': len(valid_tps),
                'spearman_rho': rho, 'p_value': p,
            })

spline_df = pd.DataFrame(spline_rows)
if len(spline_df) > 0:
    reject, padj, _, _ = multipletests(spline_df.p_value.values, method='fdr_bh')
    spline_df['padj'] = padj
    spline_df['significant'] = reject

spline_df.to_csv(os.path.join(DERIVED_DIR, 'program_metabolite_spline_corr.tsv'),
                 sep='\t', index=False)
print(f"  {len(spline_df)} spline-based correlations computed")

print("\n  Spline approach top correlations (|rho| > 0.7):")
top_spline = spline_df[spline_df.spearman_rho.abs() > 0.7].sort_values(
    'spearman_rho', key=abs, ascending=False)
for _, r in top_spline.head(20).iterrows():
    sig = '*' if r.get('significant', False) else ' '
    print(f"  {sig} {r.program:20s} × {r.metabolite_name:14s} [{r.treatment:10s}] "
          f"ρ={r.spearman_rho:+.3f}  p={r.p_value:.4f}  padj={r.padj:.4f}")

# ============================================================================
# Comparison summary
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Approach 1 (per-KO median) vs Approach 2 (spline)")
print("=" * 70)

# Merge for comparison
if len(summary_df) > 0 and len(spline_df) > 0:
    comp = summary_df[['program', 'metabolite', 'treatment', 'median_rho', 'mean_rho',
                        'frac_sig_positive', 'frac_sig_negative']].merge(
        spline_df[['program', 'metabolite', 'treatment', 'spearman_rho']].rename(
            columns={'spearman_rho': 'spline_rho'}),
        on=['program', 'metabolite', 'treatment'], how='inner'
    )

    # Correlation between the two approaches
    from scipy.stats import pearsonr
    r, p = pearsonr(comp.median_rho, comp.spline_rho)
    print(f"  Pearson r between per-KO median rho and spline rho: {r:.3f} (p={p:.1e})")
    print(f"  n = {len(comp)} matched comparisons")

    # Cases where they disagree substantially
    comp['diff'] = (comp.median_rho - comp.spline_rho).abs()
    disagree = comp[comp['diff'] > 0.4].sort_values('diff', ascending=False)
    if len(disagree) > 0:
        print(f"\n  Cases with |difference| > 0.4:")
        for _, r in disagree.iterrows():
            print(f"    {r.program:20s} × {METAB_NAMES.get(r.metabolite, r.metabolite):14s} "
                  f"[{r.treatment:10s}] perKO_median={r.median_rho:+.3f} spline={r.spline_rho:+.3f}")
    else:
        print("  No major disagreements (all |diff| < 0.4)")

print("\nDone.")
