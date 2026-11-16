#!/usr/bin/env python3
"""
Compute bootstrap G/X initial-rate ratios using fractional ¹³C labeling,
both raw and normalized by mg ¹³C added per treatment.

Fractional labeling = Σ(i × f_i) / n_C
Initial rate = OLS slope of fractional_labeling vs log₁₀(time_hours),
restricted to timepoints ≤ 72 h (2, 8, 24, 72 h). The 72 h cutoff is
informed by the P1→P2 transcriptional program transition at ~3 d, after
which substrate exhaustion and community turnover cause labeling to plateau
or decline. Using only the rising phase isolates the filling rate from
subsequent pool turnover.

Normalization: divide each treatment's slope by mg ¹³C added.
  Glucose: 5.16 mg ¹³C
  Xylose:  1.79 mg ¹³C

Bootstrap: 10,000 iterations, seed=42, resample biological replicates
with replacement within each treatment-timepoint.

Also computes 1-M+0 based ratios for comparison with original values.

Output: data/derived/normalized_bootstrap_GX_ratios.tsv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
METAB_DIR = os.path.join(BASE, 'data', 'metabolomics')
OUT_DIR = os.path.join(BASE, 'data', 'derived')
os.makedirs(OUT_DIR, exist_ok=True)

# ¹³C added per treatment (mg)
C13_ADDED = {'GLUCOSE': 5.16, 'XYLOSE': 1.79, 'CELLULOSE': 8.89}

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
    'GLUCOSE':  ('GlucoseResults_ForIsoCor_res_isotopologue_fraction.csv', 'GlucoseMetadata.csv'),
    'XYLOSE':   ('XyloseResults_ForIsoCor_res_isotopologue_fraction.csv',  'XyloseMetadata.csv'),
}

METAB_TO_METAT = {0.08: 2, 0.33: 8, 1: 24, 2: 72, 7: 168, 14: 336, 28: 672}

TCA_METABOLITES = ['Cit', 'aKG', 'Suc', 'Fum', 'Mal']

N_BOOT = 10000
SEED = 42


def load_per_replicate_data(treatment, metab_code, metric='fractional'):
    """
    Return {metat_hour: [replicate values]} for a given treatment and metabolite.

    metric='fractional': Σ(i × f_i) / n_C
    metric='1minusM0': 1 - M+0
    """
    frac_file, meta_file = TREATMENTS_INFO[treatment]
    frac_df = pd.read_csv(os.path.join(METAB_DIR, frac_file))
    meta_df = pd.read_csv(os.path.join(METAB_DIR, meta_file))

    sample_cols = [c for c in frac_df.columns if c not in ['metabolite', 'isotopologue']]
    stp = {str(r['sample']): float(r['timepoint']) for _, r in meta_df.iterrows()}
    # Also build sample -> replicate mapping
    s2rep = {str(r['sample']): int(r['replicate']) for _, r in meta_df.iterrows()}

    sub = frac_df[frac_df['metabolite'] == metab_code].copy()
    if sub.empty:
        return {}, {}

    nc = N_CARBONS[metab_code]

    if metric == 'fractional':
        # Compute Σ(i × f_i) / n_C for each sample
        weighted = sub[sample_cols].multiply(sub['isotopologue'].values, axis=0)
        frac_label = weighted.sum(axis=0) / nc
        val_dict = {sid: float(frac_label[sid]) for sid in sample_cols}
    else:
        # 1 - M+0
        m0 = sub[sub['isotopologue'] == 0]
        if m0.empty:
            return {}, {}
        val_dict = {sid: 1.0 - float(m0[sid].values[0]) for sid in sample_cols}

    # Group by timepoint, returning both grouped values and replicate IDs
    tp_vals = {}
    tp_reps = {}
    for sid in sample_cols:
        sid_str = str(sid)
        if sid_str not in stp:
            continue
        td = stp[sid_str]
        if td == 0 or td not in METAB_TO_METAT:
            continue
        mh = METAB_TO_METAT[td]
        tp_vals.setdefault(mh, []).append(val_dict[sid])
        tp_reps.setdefault(mh, []).append(s2rep.get(sid_str, -1))

    return tp_vals, tp_reps


def compute_slope(tp_vals, max_hours=72):
    """OLS slope of labeling vs log10(time_hours), restricted to early timepoints.

    Only timepoints ≤ max_hours are used, capturing the initial filling rate
    before substrate exhaustion and community turnover (P1→P2 transition at ~72h).
    """
    tps = sorted([t for t in tp_vals.keys() if t <= max_hours])
    if len(tps) < 2:
        return np.nan
    x = np.array([np.log10(t) for t in tps])
    y = np.array([np.mean(tp_vals[t]) for t in tps])
    slope, _, _, _, _ = linregress(x, y)
    return slope


def bootstrap_ratio(g_tp_vals, x_tp_vals, n_boot, rng, normalize=False):
    """
    Bootstrap G/X rate ratio.
    For each iteration, resample replicates within each timepoint (with replacement),
    compute slope for G and X, optionally normalize by ¹³C added, compute ratio.
    """
    g_tps = sorted(g_tp_vals.keys())
    x_tps = sorted(x_tp_vals.keys())

    ratios = []
    for _ in range(n_boot):
        # Resample G
        g_resampled = {}
        for tp in g_tps:
            vals = g_tp_vals[tp]
            idx = rng.integers(0, len(vals), size=len(vals))
            g_resampled[tp] = [vals[i] for i in idx]

        # Resample X
        x_resampled = {}
        for tp in x_tps:
            vals = x_tp_vals[tp]
            idx = rng.integers(0, len(vals), size=len(vals))
            x_resampled[tp] = [vals[i] for i in idx]

        g_slope = compute_slope(g_resampled)
        x_slope = compute_slope(x_resampled)

        if normalize:
            g_slope /= C13_ADDED['GLUCOSE']
            x_slope /= C13_ADDED['XYLOSE']

        if x_slope != 0 and not np.isnan(x_slope) and not np.isnan(g_slope):
            ratios.append(g_slope / x_slope)

    return np.array(ratios)


def bootstrap_tca_aggregate(treatment_data_g, treatment_data_x, n_boot, rng, normalize=False):
    """
    Bootstrap G/X ratio on aggregate TCA labeling.
    For each iteration, resample replicates within each timepoint for each metabolite,
    compute the mean fractional labeling across TCA metabolites at each timepoint,
    then compute slope.
    """
    # Get union of timepoints across all TCA metabolites for each treatment
    all_tps_g = set()
    all_tps_x = set()
    for metab in TCA_METABOLITES:
        all_tps_g.update(treatment_data_g[metab].keys())
        all_tps_x.update(treatment_data_x[metab].keys())
    # Use only timepoints present in all metabolites
    common_tps_g = sorted(all_tps_g.intersection(*[set(treatment_data_g[m].keys()) for m in TCA_METABOLITES]))
    common_tps_x = sorted(all_tps_x.intersection(*[set(treatment_data_x[m].keys()) for m in TCA_METABOLITES]))

    ratios = []
    for _ in range(n_boot):
        # For each timepoint, resample and average across metabolites
        g_agg = {}
        for tp in common_tps_g:
            metab_means = []
            for metab in TCA_METABOLITES:
                vals = treatment_data_g[metab][tp]
                idx = rng.integers(0, len(vals), size=len(vals))
                metab_means.append(np.mean([vals[i] for i in idx]))
            g_agg[tp] = [np.mean(metab_means)]  # single aggregated value

        x_agg = {}
        for tp in common_tps_x:
            metab_means = []
            for metab in TCA_METABOLITES:
                vals = treatment_data_x[metab][tp]
                idx = rng.integers(0, len(vals), size=len(vals))
                metab_means.append(np.mean([vals[i] for i in idx]))
            x_agg[tp] = [np.mean(metab_means)]

        g_slope = compute_slope(g_agg)
        x_slope = compute_slope(x_agg)

        if normalize:
            g_slope /= C13_ADDED['GLUCOSE']
            x_slope /= C13_ADDED['XYLOSE']

        if x_slope != 0 and not np.isnan(x_slope) and not np.isnan(g_slope):
            ratios.append(g_slope / x_slope)

    return np.array(ratios)


# =============================================================================
# Main
# =============================================================================
print(f"Running bootstrap G/X rate ratios (n_boot={N_BOOT}, seed={SEED})")
print(f"  Glucose ¹³C added: {C13_ADDED['GLUCOSE']} mg")
print(f"  Xylose  ¹³C added: {C13_ADDED['XYLOSE']} mg")
print(f"  Normalization factor G/X: {C13_ADDED['GLUCOSE']/C13_ADDED['XYLOSE']:.3f}")
print()

rng = np.random.default_rng(SEED)

results = []

# Store per-metabolite data for TCA aggregate
tca_data_g_frac = {}
tca_data_x_frac = {}
tca_data_g_m0 = {}
tca_data_x_m0 = {}

for metab_code in N_CARBONS:
    name = METABOLITE_NAMES[metab_code]
    nc = N_CARBONS[metab_code]

    # Load fractional labeling data
    g_frac, _ = load_per_replicate_data('GLUCOSE', metab_code, metric='fractional')
    x_frac, _ = load_per_replicate_data('XYLOSE', metab_code, metric='fractional')

    # Load 1-M0 data
    g_m0, _ = load_per_replicate_data('GLUCOSE', metab_code, metric='1minusM0')
    x_m0, _ = load_per_replicate_data('XYLOSE', metab_code, metric='1minusM0')

    if not g_frac or not x_frac:
        print(f"  {metab_code}: skipping (no data)")
        continue

    # Store for TCA aggregate
    if metab_code in TCA_METABOLITES:
        tca_data_g_frac[metab_code] = g_frac
        tca_data_x_frac[metab_code] = x_frac
        tca_data_g_m0[metab_code] = g_m0
        tca_data_x_m0[metab_code] = x_m0

    # Point estimates
    g_slope_frac = compute_slope(g_frac)
    x_slope_frac = compute_slope(x_frac)
    raw_frac_ratio = g_slope_frac / x_slope_frac if x_slope_frac != 0 else np.nan

    g_slope_m0 = compute_slope(g_m0)
    x_slope_m0 = compute_slope(x_m0)
    old_m0_ratio = g_slope_m0 / x_slope_m0 if x_slope_m0 != 0 else np.nan

    norm_frac_ratio = (g_slope_frac / C13_ADDED['GLUCOSE']) / (x_slope_frac / C13_ADDED['XYLOSE']) if x_slope_frac != 0 else np.nan

    # Bootstrap: raw fractional
    boot_raw = bootstrap_ratio(g_frac, x_frac, N_BOOT, rng, normalize=False)
    raw_median = np.median(boot_raw)
    raw_lo, raw_hi = np.percentile(boot_raw, [2.5, 97.5])

    # Bootstrap: normalized fractional
    boot_norm = bootstrap_ratio(g_frac, x_frac, N_BOOT, rng, normalize=True)
    norm_median = np.median(boot_norm)
    norm_lo, norm_hi = np.percentile(boot_norm, [2.5, 97.5])

    # Bootstrap: old 1-M0
    boot_old = bootstrap_ratio(g_m0, x_m0, N_BOOT, rng, normalize=False)
    old_median = np.median(boot_old)
    old_lo, old_hi = np.percentile(boot_old, [2.5, 97.5])

    sig_raw = (raw_lo > 1.0) or (raw_hi < 1.0)
    sig_norm = (norm_lo > 1.0) or (norm_hi < 1.0)
    sig_old = (old_lo > 1.0) or (old_hi < 1.0)

    results.append({
        'metabolite': metab_code,
        'metabolite_name': name,
        'n_carbons': nc,
        'old_1minusM0_ratio': round(old_median, 2),
        'old_CI_lo': round(old_lo, 2),
        'old_CI_hi': round(old_hi, 2),
        'significant_old': sig_old,
        'raw_GX_ratio': round(raw_median, 2),
        'raw_GX_CI_lo': round(raw_lo, 2),
        'raw_GX_CI_hi': round(raw_hi, 2),
        'significant_raw': sig_raw,
        'norm_GX_ratio': round(norm_median, 2),
        'norm_GX_CI_lo': round(norm_lo, 2),
        'norm_GX_CI_hi': round(norm_hi, 2),
        'significant_norm': sig_norm,
    })

# TCA aggregate
print("\nComputing TCA aggregate...")
boot_tca_raw = bootstrap_tca_aggregate(tca_data_g_frac, tca_data_x_frac, N_BOOT, rng, normalize=False)
boot_tca_norm = bootstrap_tca_aggregate(tca_data_g_frac, tca_data_x_frac, N_BOOT, rng, normalize=True)
boot_tca_old = bootstrap_tca_aggregate(tca_data_g_m0, tca_data_x_m0, N_BOOT, rng, normalize=False)

tca_raw_med = np.median(boot_tca_raw)
tca_raw_lo, tca_raw_hi = np.percentile(boot_tca_raw, [2.5, 97.5])
tca_norm_med = np.median(boot_tca_norm)
tca_norm_lo, tca_norm_hi = np.percentile(boot_tca_norm, [2.5, 97.5])
tca_old_med = np.median(boot_tca_old)
tca_old_lo, tca_old_hi = np.percentile(boot_tca_old, [2.5, 97.5])

results.append({
    'metabolite': 'TCA_agg',
    'metabolite_name': 'TCA aggregate',
    'n_carbons': 0,
    'old_1minusM0_ratio': round(tca_old_med, 2),
    'old_CI_lo': round(tca_old_lo, 2),
    'old_CI_hi': round(tca_old_hi, 2),
    'significant_old': (tca_old_lo > 1.0) or (tca_old_hi < 1.0),
    'raw_GX_ratio': round(tca_raw_med, 2),
    'raw_GX_CI_lo': round(tca_raw_lo, 2),
    'raw_GX_CI_hi': round(tca_raw_hi, 2),
    'significant_raw': (tca_raw_lo > 1.0) or (tca_raw_hi < 1.0),
    'norm_GX_ratio': round(tca_norm_med, 2),
    'norm_GX_CI_lo': round(tca_norm_lo, 2),
    'norm_GX_CI_hi': round(tca_norm_hi, 2),
    'significant_norm': (tca_norm_lo > 1.0) or (tca_norm_hi < 1.0),
})

# Save results
df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, 'normalized_bootstrap_GX_ratios.tsv')
df.to_csv(out_path, sep='\t', index=False)
print(f"\nSaved results to {out_path}")

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 120)
print("COMPARISON TABLE: G/X Bootstrap Rate Ratios")
print("=" * 120)
print(f"{'Metabolite':<22s} | {'Old (1-M+0)':<22s} | {'Fractional (raw)':<22s} | {'Fractional (norm)':<22s}")
print(f"{'':22s} | {'ratio [95% CI]':<22s} | {'ratio [95% CI]':<22s} | {'ratio [95% CI]':<22s}")
print("-" * 120)

for r in results:
    sig_old = '*' if r['significant_old'] else ' '
    sig_raw = '*' if r['significant_raw'] else ' '
    sig_norm = '*' if r['significant_norm'] else ' '

    old_str = f"{r['old_1minusM0_ratio']:.1f}× [{r['old_CI_lo']:.1f}, {r['old_CI_hi']:.1f}]{sig_old}"
    raw_str = f"{r['raw_GX_ratio']:.1f}× [{r['raw_GX_CI_lo']:.1f}, {r['raw_GX_CI_hi']:.1f}]{sig_raw}"
    norm_str = f"{r['norm_GX_ratio']:.1f}× [{r['norm_GX_CI_lo']:.1f}, {r['norm_GX_CI_hi']:.1f}]{sig_norm}"

    print(f"{r['metabolite_name']:<22s} | {old_str:<22s} | {raw_str:<22s} | {norm_str:<22s}")

print("-" * 120)
print("* = 95% CI excludes 1.0 (significant)")
print(f"Normalization: G slope / {C13_ADDED['GLUCOSE']} mg ¹³C, X slope / {C13_ADDED['XYLOSE']} mg ¹³C")
print(f"  => normalization ratio = {C13_ADDED['GLUCOSE']/C13_ADDED['XYLOSE']:.3f}")
print(f"  => normalized G/X = raw G/X / {C13_ADDED['GLUCOSE']/C13_ADDED['XYLOSE']:.3f}")
