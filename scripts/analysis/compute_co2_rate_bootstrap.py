#!/usr/bin/env python3
"""
Compute ¹³C-CO₂ production rates and bootstrap G/X/H rate ratios.

Methodology:
- Convert δ¹³C-VPDB to fractional abundance F = R/(1+R) where R = R_VPDB*(1+δ/1000)
- Subtract baseline F (mean of basal/control t=0 jars)
- Compute excess ¹³C flux = (F - F_baseline) × CO₂_mg / Δt / soil_g
- Bootstrap G/X slope ratio (excess ¹³C flux vs log₁₀(time))

See CLAUDE.md "δ¹³C-CO₂ rate calculation methodology" for full protocol.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
CO2_FILE = os.path.join(BASE, 'data', 'co2', 'sugar_experiment_CO2_mgC_withJarInfo.csv')
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
TABLES_DIR = os.path.join(BASE, 'tables', 'output')

R_VPDB = 0.0112372
N_BOOT = 10_000
np.random.seed(42)

# ============================================================================
# Load and parse data
# ============================================================================
print("Loading CO₂ data...")
df = pd.read_csv(CO2_FILE)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'δ13CVPDB (‰)': 'delta13C',
    'time point': 'time_h',
    'Converted CO2 (mg)': 'co2_mg',
    'grams in jar': 'soil_g',
    'jar number': 'jar',
})

# Convert δ¹³C to fractional abundance
df['R'] = R_VPDB * (1 + df['delta13C'] / 1000)
df['F'] = df['R'] / (1 + df['R'])

# Baseline F from basal + control t=0
baseline_rows = df[df['treatment'].isin(['basal', 'control'])]
F_baseline = baseline_rows['F'].mean()
print(f"  Baseline F = {F_baseline:.6f} (from {len(baseline_rows)} measurements)")
print(f"  Baseline δ¹³C ≈ {baseline_rows['delta13C'].mean():.2f}‰")

# ============================================================================
# Define accumulation windows
# ============================================================================
# For paired timepoints: use the post-flush (later) measurement
# Δt is the accumulation window in hours

# Jar-tracked (8h jars): 0→1h, 1→4h, 4→8h
# Sacrificial: 0→2h, 0→24h
# Paired: 71→72h, 167→168h, 335→336h, 671→672h

# Map: nominal timepoint → (measurement time_h to use, Δt in hours)
RATE_CONFIG = {
    1:   {'use_t': 1,   'dt_h': 1},
    2:   {'use_t': 2,   'dt_h': 2},
    4:   {'use_t': 4,   'dt_h': 3},
    8:   {'use_t': 8,   'dt_h': 4},
    24:  {'use_t': 24,  'dt_h': 24},
    72:  {'use_t': 72,  'dt_h': 1},
    168: {'use_t': 168, 'dt_h': 1},
    336: {'use_t': 336, 'dt_h': 1},
    672: {'use_t': 672, 'dt_h': 1},
}

# ============================================================================
# Compute excess ¹³C flux at each timepoint
# ============================================================================
print("\nComputing excess ¹³C-CO₂ production rates...")

treatments = ['glucose', 'xylose', 'cellulose']
treat_labels = {'glucose': 'Glucose', 'xylose': 'Xylose', 'cellulose': 'Hemicellulose'}

rate_rows = []
for treat in treatments:
    tdf = df[df['treatment'] == treat]
    available_tps = sorted(tdf['time_h'].unique())

    for nominal_tp, cfg in RATE_CONFIG.items():
        use_t = cfg['use_t']
        dt_h = cfg['dt_h']

        # Check this timepoint exists for this treatment
        sub = tdf[tdf['time_h'] == use_t]
        if len(sub) == 0:
            continue

        for _, row in sub.iterrows():
            excess_F = row['F'] - F_baseline
            # excess ¹³C flux: (excess_F × co2_mg_C) / dt_h / soil_g
            # Convert to µg: × 1000
            excess_13C_flux = (excess_F * row['co2_mg'] * 1000) / dt_h / row['soil_g']

            rate_rows.append({
                'treatment': treat_labels[treat],
                'nominal_timepoint_h': nominal_tp,
                'measurement_t_h': use_t,
                'accumulation_window_h': dt_h,
                'jar': row['jar'],
                'replicate': row['replicate'],
                'delta13C': row['delta13C'],
                'F': row['F'],
                'F_baseline': F_baseline,
                'excess_F': excess_F,
                'co2_mg': row['co2_mg'],
                'soil_g': row['soil_g'],
                'excess_13C_flux_ug_per_h_per_g': excess_13C_flux,
            })

rates = pd.DataFrame(rate_rows)

# Print summary
print("\nExcess ¹³C-CO₂ flux (µg ¹³C / h / g soil) — timepoint means:")
print("-" * 70)
for treat in ['Glucose', 'Xylose', 'Hemicellulose']:
    t = rates[rates['treatment'] == treat]
    means = t.groupby('nominal_timepoint_h')['excess_13C_flux_ug_per_h_per_g'].agg(['mean', 'std', 'count'])
    print(f"\n  {treat}:")
    for tp, row in means.iterrows():
        print(f"    t={tp:>4d}h: {row['mean']:8.4f} ± {row['std']:7.4f} (n={int(row['count'])})")

# Save rates
rates_file = os.path.join(DERIVED_DIR, 'co2_excess_13C_flux_rates.tsv')
rates.to_csv(rates_file, sep='\t', index=False, float_format='%.6f')
print(f"\nSaved rates to {rates_file}")

# ============================================================================
# Bootstrap G/X and G/H rate ratios
# ============================================================================
print("\n" + "=" * 70)
print("Bootstrap rate ratios (excess ¹³C-CO₂ flux vs log₁₀(time))...")
print("=" * 70)

# Use timepoints common to metabolomics for comparability: 2, 8, 24, 72, 168, 336h
# (exclude 1h, 4h which are from jar-tracked protocol and not in metabolomics)
BOOTSTRAP_TPS = [2, 8, 24, 72, 168, 336]

def bootstrap_slope(data, timepoints, n_boot=N_BOOT):
    """Bootstrap regression slope of excess ¹³C flux vs log₁₀(time)."""
    slopes = []
    for _ in range(n_boot):
        boot_t, boot_v = [], []
        for tp in timepoints:
            tp_vals = data[data['nominal_timepoint_h'] == tp]['excess_13C_flux_ug_per_h_per_g'].values
            if len(tp_vals) > 0:
                resampled = np.random.choice(tp_vals, len(tp_vals), replace=True)
                boot_t.append(tp)
                boot_v.append(np.mean(resampled))
        if len(boot_t) >= 3:
            X = np.log10(np.array(boot_t)).reshape(-1, 1)
            y = np.array(boot_v)
            slopes.append(LinearRegression().fit(X, y).coef_[0])
    return np.array(slopes)

# Point estimates
ratio_rows = []
for pair_name, treat_a, treat_b in [
    ('G/X', 'Glucose', 'Xylose'),
    ('G/H', 'Glucose', 'Hemicellulose'),
    ('X/H', 'Xylose', 'Hemicellulose'),
]:
    data_a = rates[rates['treatment'] == treat_a]
    data_b = rates[rates['treatment'] == treat_b]

    # Use timepoints available for both treatments
    tps = BOOTSTRAP_TPS

    # Point estimate slopes
    a_avg = data_a[data_a['nominal_timepoint_h'].isin(tps)].groupby(
        'nominal_timepoint_h')['excess_13C_flux_ug_per_h_per_g'].mean().reset_index()
    b_avg = data_b[data_b['nominal_timepoint_h'].isin(tps)].groupby(
        'nominal_timepoint_h')['excess_13C_flux_ug_per_h_per_g'].mean().reset_index()

    if len(a_avg) >= 3 and len(b_avg) >= 3:
        a_slope = LinearRegression().fit(
            np.log10(a_avg['nominal_timepoint_h'].values).reshape(-1, 1),
            a_avg['excess_13C_flux_ug_per_h_per_g'].values).coef_[0]
        b_slope = LinearRegression().fit(
            np.log10(b_avg['nominal_timepoint_h'].values).reshape(-1, 1),
            b_avg['excess_13C_flux_ug_per_h_per_g'].values).coef_[0]
    else:
        a_slope = b_slope = np.nan

    # Bootstrap
    a_boots = bootstrap_slope(data_a, tps)
    b_boots = bootstrap_slope(data_b, tps)

    n_min = min(len(a_boots), len(b_boots))
    ratio_arr = a_boots[:n_min] / b_boots[:n_min]
    ratio_valid = ratio_arr[np.isfinite(ratio_arr) & (ratio_arr > 0) & (ratio_arr < 100)]

    result = {
        'comparison': pair_name,
        f'{treat_a}_slope': round(a_slope, 6),
        f'{treat_b}_slope': round(b_slope, 6),
        'ratio_median': round(np.median(ratio_valid), 3),
        'CI_lo_2.5pct': round(np.percentile(ratio_valid, 2.5), 3),
        'CI_hi_97.5pct': round(np.percentile(ratio_valid, 97.5), 3),
        'p_A_greater_B': round(np.mean(a_boots[:n_min] > b_boots[:n_min]), 4),
        'significant': 'yes' if np.percentile(ratio_valid, 2.5) > 1.0 else 'no',
        'n_valid_bootstraps': len(ratio_valid),
    }
    ratio_rows.append(result)

    print(f"\n  {pair_name}: {result['ratio_median']}× [{result['CI_lo_2.5pct']}, {result['CI_hi_97.5pct']}]"
          f"  p={result['p_A_greater_B']}  sig={result['significant']}")
    print(f"    {treat_a} slope: {a_slope:.6f}")
    print(f"    {treat_b} slope: {b_slope:.6f}")

# Save bootstrap results
ratio_df = pd.DataFrame(ratio_rows)
ratio_file = os.path.join(TABLES_DIR, 'CO2_GX_bootstrap_rate_ratios.tsv')
ratio_df.to_csv(ratio_file, sep='\t', index=False)
print(f"\nSaved bootstrap results to {ratio_file}")

# ============================================================================
# Also compute with ALL timepoints (including 1h, 4h) for comparison
# ============================================================================
print("\n" + "=" * 70)
print("Sensitivity check: using ALL timepoints (1, 2, 4, 8, 24, 72, 168, 336h)...")
print("=" * 70)

ALL_TPS = [1, 2, 4, 8, 24, 72, 168, 336]

for pair_name, treat_a, treat_b in [
    ('G/X', 'Glucose', 'Xylose'),
    ('G/H', 'Glucose', 'Hemicellulose'),
]:
    data_a = rates[rates['treatment'] == treat_a]
    data_b = rates[rates['treatment'] == treat_b]

    a_boots = bootstrap_slope(data_a, ALL_TPS)
    b_boots = bootstrap_slope(data_b, ALL_TPS)

    n_min = min(len(a_boots), len(b_boots))
    ratio_arr = a_boots[:n_min] / b_boots[:n_min]
    ratio_valid = ratio_arr[np.isfinite(ratio_arr) & (ratio_arr > 0) & (ratio_arr < 100)]

    print(f"  {pair_name} (all TPs): {np.median(ratio_valid):.3f}× "
          f"[{np.percentile(ratio_valid, 2.5):.3f}, {np.percentile(ratio_valid, 97.5):.3f}]")

print("\nDone.")
