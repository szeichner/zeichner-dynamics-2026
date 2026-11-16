#!/usr/bin/env python3
"""
Compute fractional 13C labeling for all metabolites × 3 treatments × all timepoints.

fractional_labeling = Σ(i × f_i) / n_C

Outputs: data/derived/metabolite_fractional_labeling.tsv
"""

import os
import numpy as np
import pandas as pd

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')
OUT_DIR = os.path.join(BASE, 'data', 'derived')
os.makedirs(OUT_DIR, exist_ok=True)

N_CARBONS = {
    'PYR': 3, 'LAT': 3, 'Cit': 6, 'aKG': 5, 'Suc': 4,
    'Fum': 4, 'Mal': 4, 'Asp': 4, 'GLUM': 5, 'AGLUM': 7
}

METABOLITE_NAMES = {
    'PYR': 'Pyruvate', 'LAT': 'Lactate', 'Cit': 'Citrate',
    'aKG': 'α-Ketoglutarate', 'Suc': 'Succinate', 'Fum': 'Fumarate',
    'Mal': 'Malate', 'Asp': 'Aspartate', 'GLUM': 'Glutamate',
    'AGLUM': 'N-Acetylglutamate'
}

TREATMENTS_INFO = {
    'GLUCOSE':   ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv', 'GlucoseMetadata.csv'),
    'XYLOSE':    ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv',  'XyloseMetadata.csv'),
    'CELLULOSE': ('CelluloseResults_ForIsoCor_res_isotopologue_fraction.csv', 'CelluloseMetadata.csv'),
}

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}


def compute_fractional_labeling(isotopologue_fractions_df, n_carbons_dict):
    """
    Compute fractional 13C labeling for each metabolite and sample.
    fractional_labeling = sum(i * f_i) / n_C
    """
    sample_cols = [c for c in isotopologue_fractions_df.columns
                   if c not in ['metabolite', 'isotopologue']]
    results = {}
    for metab, nc in n_carbons_dict.items():
        sub = isotopologue_fractions_df[
            isotopologue_fractions_df['metabolite'] == metab
        ].copy()
        if sub.empty:
            continue
        weighted = sub[sample_cols].multiply(sub['isotopologue'].values, axis=0)
        frac_label = weighted.sum(axis=0) / nc
        results[metab] = frac_label
    return pd.DataFrame(results).T


def load_1_minus_m0(frac_df, metab_code, sample_cols):
    """Get 1 - M+0 for a metabolite."""
    m0 = frac_df[(frac_df.metabolite == metab_code) & (frac_df.isotopologue == 0)]
    if len(m0) == 0:
        return {}
    return {sid: 1.0 - float(m0[sid].values[0]) for sid in sample_cols}


rows = []

for treatment, (frac_file, meta_file) in TREATMENTS_INFO.items():
    frac_path = os.path.join(METAB_DIR, frac_file)
    meta_path = os.path.join(METAB_DIR, meta_file)

    frac_df = pd.read_csv(frac_path)
    meta_df = pd.read_csv(meta_path)

    sample_cols = [c for c in frac_df.columns
                   if c not in ['metabolite', 'isotopologue']]

    # Compute fractional labeling
    fl_df = compute_fractional_labeling(frac_df, N_CARBONS)

    # Build sample → timepoint mapping
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta_df.iterrows()}

    for metab_code in N_CARBONS:
        if metab_code not in fl_df.index:
            continue

        # Get 1-M0 values for comparison
        m0_dict = load_1_minus_m0(frac_df, metab_code, sample_cols)

        # Group by timepoint
        tp_fl = {}
        tp_m0 = {}
        for sid in sample_cols:
            sid_str = str(sid)
            if sid_str not in stp:
                continue
            td = stp[sid_str]
            if td == 0 or td not in METAB_TO_METAT:
                continue
            mh = METAB_TO_METAT[td]
            tp_fl.setdefault(mh, []).append(float(fl_df.loc[metab_code, sid]))
            if sid in m0_dict:
                tp_m0.setdefault(mh, []).append(m0_dict[sid])

        for tp_hours in sorted(tp_fl.keys()):
            vals = tp_fl[tp_hours]
            m0_vals = tp_m0.get(tp_hours, [])
            rows.append({
                'treatment': treatment,
                'timepoint_hours': tp_hours,
                'metabolite': metab_code,
                'metabolite_name': METABOLITE_NAMES[metab_code],
                'n_carbons': N_CARBONS[metab_code],
                'mean_fractional_labeling': np.mean(vals),
                'sem': np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                'n_replicates': len(vals),
                'mean_1minusM0': np.mean(m0_vals) if m0_vals else np.nan,
            })

result_df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, 'metabolite_fractional_labeling.tsv')
result_df.to_csv(out_path, sep='\t', index=False)
print(f"Saved {len(result_df)} rows to {out_path}")

# Print summary for key metabolites
print("\n=== Key values ===")
for metab in ['Suc', 'Cit', 'Asp', 'PYR']:
    print(f"\n{METABOLITE_NAMES[metab]} ({metab}):")
    sub = result_df[result_df.metabolite == metab]
    for _, r in sub.iterrows():
        print(f"  {r['treatment']:10s} {r['timepoint_hours']:5.0f}h: "
              f"frac_label={r['mean_fractional_labeling']:.4f}  "
              f"1-M0={r['mean_1minusM0']:.4f}  "
              f"n={r['n_replicates']}")
