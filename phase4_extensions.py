"""
PHASE 4: Extensions — Lagged ARIMAX + COVID Dummy + Granger Causality
US Housing & Consumer Market Indicators — Time Series Analysis
Ruchita Anil Zingade | MSE Semester III

What this script does:
1. Granger Causality Test — does mortgage rate statistically cause housing starts?
2. COVID Dummy Variable — isolates pandemic shock from normal model behaviour
3. Lagged ARIMAX — uses mortgage rate at t-3 and t-6 (more realistic transmission lag)
4. Compares lagged ARIMAX vs original ARIMAX
5. Saves all results to Excel
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 4: Extensions")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading prepared data ...")

data = pd.read_csv('fred_data_prepared.csv', index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index)
data.index.freq = 'MS'

y        = data['housing_starts'].dropna()
mortgage = data['mortgage_rate'].dropna()
fed      = data['fed_funds'].dropna()

common_idx = y.index.intersection(mortgage.index).intersection(fed.index)
y        = y[common_idx]
mortgage = mortgage[common_idx]
fed      = fed[common_idx]

print(f"    Loaded: {len(y)} observations | {y.index[0].strftime('%Y-%m')} to {y.index[-1].strftime('%Y-%m')}")

# ─────────────────────────────────────────────
# EXTENSION 1: GRANGER CAUSALITY TEST
#
# Granger causality answers: "Does knowing past values of
# mortgage_rate help predict housing_starts, BEYOND what
# housing_starts alone can predict?"
#
# It does NOT prove economic causation — it proves
# statistical predictive causation (hence "Granger" causality).
#
# We test at lags 1, 3, 6, 12 months.
#
# H0: mortgage_rate does NOT Granger-cause housing_starts
# If p < 0.05 → reject H0 → mortgage_rate DOES Granger-cause starts
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXTENSION 1: Granger Causality Test")
print("=" * 60)
print("\n  Does mortgage rate Granger-cause housing starts?")
print("  (i.e., do past mortgage rates help predict future starts?)")
print("-" * 60)

# Granger test needs a 2-column dataframe: [dependent, causal]
granger_df = pd.DataFrame({
    'housing_starts': y,
    'mortgage_rate':  mortgage
}).dropna()

# Test at multiple lags
granger_lags = [1, 3, 6, 12]
granger_results = []

gc_test = grangercausalitytests(granger_df, maxlag=12)

print(f"\n  {'Lag (months)':<15} {'F-stat':>10} {'p-value':>10} {'Result':>20}")
print("  " + "-" * 58)

for lag in granger_lags:
    # ssr_ftest is the F-test version (most common)
    f_stat = gc_test[lag][0]['ssr_ftest'][0]
    p_val  = gc_test[lag][0]['ssr_ftest'][1]
    result = "✓ Granger-causes" if p_val < 0.05 else "✗ Does not cause"
    sig    = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  {lag:<15} {f_stat:>10.3f} {p_val:>10.4f} {result:>20} {sig}")
    granger_results.append({'Lag': lag, 'F_stat': round(f_stat,3),
                            'p_value': round(p_val,4), 'Result': result})

granger_df_out = pd.DataFrame(granger_results)

# Also test: does housing starts Granger-cause mortgage rate?
print("\n  Does housing starts Granger-cause mortgage rate?")
print("  (reverse direction — checking for reverse causality)")
print("-" * 60)

granger_df2 = pd.DataFrame({
    'mortgage_rate':  mortgage,
    'housing_starts': y
}).dropna()

gc_test2 = grangercausalitytests(granger_df2, maxlag=6)

print(f"\n  {'Lag (months)':<15} {'F-stat':>10} {'p-value':>10} {'Result':>20}")
print("  " + "-" * 58)

for lag in [1, 3, 6]:
    f_stat = gc_test2[lag][0]['ssr_ftest'][0]
    p_val  = gc_test2[lag][0]['ssr_ftest'][1]
    result = "✓ Granger-causes" if p_val < 0.05 else "✗ Does not cause"
    print(f"  {lag:<15} {f_stat:>10.3f} {p_val:>10.4f} {result:>20}")

# ─────────────────────────────────────────────
# EXTENSION 2: COVID DUMMY VARIABLE
#
# March 2020 to June 2021 = extreme outlier period
# We add a binary dummy variable:
#   covid_dummy = 1 during this period
#   covid_dummy = 0 otherwise
#
# This tells the model: "these months were structurally
# different — don't let them distort the normal pattern."
#
# The coefficient on covid_dummy captures the average
# deviation during COVID relative to the model's baseline.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXTENSION 2: COVID Dummy Variable")
print("=" * 60)

# Create dummy
covid_dummy = pd.Series(0, index=y.index, name='covid_dummy')
covid_dummy['2020-03':'2021-06'] = 1

print(f"\n  COVID dummy = 1 for: 2020-03 to 2021-06 ({covid_dummy.sum()} months)")
print(f"  COVID dummy = 0 for all other months")

# Fit ARIMA(0,1,1) with COVID dummy
exog_covid = covid_dummy.copy()

model_covid = SARIMAX(
    y,
    exog=exog_covid,
    order=(0, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

covid_coef  = model_covid.params['covid_dummy']
covid_pval  = model_covid.pvalues['covid_dummy']
covid_aic   = model_covid.aic

print(f"\n  ARIMA(0,1,1) + COVID dummy results:")
print(f"    COVID dummy coefficient: {covid_coef:.1f}")
print(f"    p-value:                 {covid_pval:.4f} {'*** significant' if covid_pval < 0.01 else ('** significant' if covid_pval < 0.05 else 'not significant')}")
print(f"    AIC:                     {covid_aic:.2f}  (vs 3694.61 without dummy)")

if covid_coef < 0:
    print(f"\n  Interpretation: During COVID, housing starts were on average")
    print(f"  {abs(covid_coef):.0f} units LOWER than the model would otherwise predict.")
else:
    print(f"\n  Interpretation: During COVID, housing starts were on average")
    print(f"  {abs(covid_coef):.0f} units HIGHER than the model would otherwise predict.")
    print(f"  (This reflects the 2020-21 housing boom driven by remote work demand.)")

# ─────────────────────────────────────────────
# EXTENSION 3: LAGGED ARIMAX
#
# In Phase 3, we used mortgage_rate at time t (current month).
# But in reality, a rate hike today affects housing starts
# 3-6 months later (permits take time, builders adjust plans).
#
# So we add:
#   mortgage_rate_lag3  = mortgage rate 3 months ago
#   mortgage_rate_lag6  = mortgage rate 6 months ago
#   fed_funds (current) = still included as control
#
# If these lagged values are significant, it confirms that
# monetary policy transmission has a 3-6 month delay.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXTENSION 3: Lagged ARIMAX")
print("=" * 60)

# Create lagged variables
data_lagged = pd.DataFrame({
    'housing_starts':     y,
    'mortgage_rate':      mortgage,
    'mortgage_lag3':      mortgage.shift(3),   # rate 3 months ago
    'mortgage_lag6':      mortgage.shift(6),   # rate 6 months ago
    'fed_funds':          fed,
    'covid_dummy':        covid_dummy
}).dropna()   # dropna removes first 6 rows (no lag data yet)

y_lagged    = data_lagged['housing_starts']
exog_lagged = data_lagged[['mortgage_rate', 'mortgage_lag3', 'mortgage_lag6',
                            'fed_funds', 'covid_dummy']]

print(f"\n  Variables included:")
print(f"    mortgage_rate      (current month)")
print(f"    mortgage_lag3      (rate 3 months ago)")
print(f"    mortgage_lag6      (rate 6 months ago)")
print(f"    fed_funds          (current)")
print(f"    covid_dummy        (pandemic control)")

# Fit lagged ARIMAX
model_lagged = SARIMAX(
    y_lagged,
    exog=exog_lagged,
    order=(0, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

lagged_aic = model_lagged.aic

print(f"\n  Lagged ARIMAX results  |  AIC: {lagged_aic:.2f}")
print(f"\n  {'Variable':<22} {'Coefficient':>12} {'p-value':>10} {'Significance':>15}")
print("  " + "-" * 62)

lag_coef_results = []
for name in ['mortgage_rate', 'mortgage_lag3', 'mortgage_lag6', 'fed_funds', 'covid_dummy']:
    coef = model_lagged.params[name]
    pval = model_lagged.pvalues[name]
    sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "n.s."))
    print(f"  {name:<22} {coef:>12.3f} {pval:>10.4f} {sig:>15}")
    lag_coef_results.append({'Variable': name, 'Coefficient': round(coef,3),
                             'p_value': round(pval,4), 'Significance': sig})

lag_coef_df = pd.DataFrame(lag_coef_results)

# ─────────────────────────────────────────────
# STEP 2: SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison = pd.DataFrame({
    'Model':       ['ARIMA(0,1,1)',
                    'ARIMAX (Phase 3)',
                    'ARIMA + COVID dummy',
                    'Lagged ARIMAX + COVID'],
    'AIC':         [3694.61, 3672.18, round(covid_aic,2), round(lagged_aic,2)],
    'Key addition': ['Baseline',
                     'Mortgage rate + fed funds (contemporaneous)',
                     'COVID structural break controlled',
                     'Lagged rates + COVID (most realistic)']
})

print("\n" + comparison.to_string(index=False))
best_idx = comparison['AIC'].idxmin()
print(f"\n  → Best model: {comparison.loc[best_idx, 'Model']} (AIC = {comparison.loc[best_idx, 'AIC']})")

# ─────────────────────────────────────────────
# STEP 3: PLOT — COVID dummy effect
# ─────────────────────────────────────────────
print("\n[7] Generating plots ...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Phase 4 Extensions — US Housing Starts\nRuchita Anil Zingade | MSE Semester III',
             fontsize=12, fontweight='bold')

# Plot 1: Actual series with COVID period highlighted
ax1 = axes[0]
ax1.plot(y, color='#37474F', linewidth=1, label='Housing Starts')
ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'),
            alpha=0.2, color='#E53935', label='COVID dummy period')
ax1.axhline(y=y.mean(), color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Long-run mean')
ax1.set_title('Housing Starts with COVID Structural Break Identified', fontsize=10, fontweight='bold')
ax1.set_ylabel('Housing Starts (000s)', fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45, labelsize=8)

# Plot 2: Mortgage rate and lags
ax2 = axes[1]
ax2.plot(mortgage, color='#1E88E5', linewidth=1,   label='Mortgage rate (current)')
ax2.plot(mortgage.shift(3), color='#43A047', linewidth=1, linestyle='--', label='Mortgage rate (lag 3)')
ax2.plot(mortgage.shift(6), color='#FB8C00', linewidth=1, linestyle=':', label='Mortgage rate (lag 6)')
ax2.set_title('Mortgage Rate and Lagged Values (3-month, 6-month)', fontsize=10, fontweight='bold')
ax2.set_ylabel('30-Year Mortgage Rate (%)', fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig('phase4_extensions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: phase4_extensions.png")

# ─────────────────────────────────────────────
# STEP 4: SAVE TO EXCEL
# ─────────────────────────────────────────────
print("\n[8] Saving to Excel ...")

with pd.ExcelWriter('phase4_results.xlsx', engine='openpyxl') as writer:
    granger_df_out.to_excel(writer, sheet_name='Granger_Causality', index=False)
    lag_coef_df.to_excel(writer, sheet_name='Lagged_ARIMAX_Coefficients', index=False)
    comparison.to_excel(writer, sheet_name='Model_Comparison', index=False)

print("    ✓ Saved: phase4_results.xlsx (3 sheets)")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print("""
What you can now say about this project:

1. GRANGER CAUSALITY
   "Granger causality tests were conducted to assess
   whether mortgage rate changes statistically precede
   movements in housing starts."

2. COVID STRUCTURAL BREAK
   "A binary dummy variable was introduced for March 2020
   to June 2021 to control for the pandemic-induced
   structural break in housing market activity."

3. LAGGED ARIMAX
   "Recognising that monetary policy operates with a
   transmission lag, mortgage rates at t-3 and t-6
   were included as regressors in an extended ARIMAX
   specification."
""")
print("=" * 60)
