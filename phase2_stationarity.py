"""
PHASE 2: Decomposition + Stationarity Testing
US Housing & Consumer Market Indicators — Time Series Analysis
Ruchita Anil Zingade | MSE Semester III

What this script does:
1. Loads the data we pulled in Phase 1 (fred_data.csv)
2. Decomposes each series into: Trend + Seasonality + Residual
3. Runs ADF (Augmented Dickey-Fuller) test to check if series is stationary
4. First-differences the non-stationary series
5. Saves decomposition plots + stationarity summary table
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 2: Decomposition + Stationarity Testing")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading data from fred_data.csv ...")

data = pd.read_csv('fred_data.csv', index_col=0, parse_dates=True)
data.index = pd.DatetimeIndex(data.index).to_period('M')  # Monthly period index

print(f"    Loaded: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"    Period: {data.index[0]} to {data.index[-1]}")
print(f"    Columns: {list(data.columns)}")

# ─────────────────────────────────────────────
# STEP 2: SEASONAL DECOMPOSITION
# What is decomposition?
#   Any time series = Trend + Seasonality + Residual (noise)
#   Trend    → long-run direction (rising, falling)
#   Seasonal → repeating pattern every year (e.g., more houses built in summer)
#   Residual → what's left after removing trend + seasonality (random noise)
#
# We use "additive" model: Y = Trend + Season + Residual
# (Use "multiplicative" if values are always positive and variance grows with level)
# ─────────────────────────────────────────────
print("\n[2] Running seasonal decomposition for each series ...")

# We'll decompose 3 key series (the most economically interesting ones)
series_to_decompose = {
    'housing_starts':  'Housing Starts (HOUST)',
    'mortgage_rate':   '30-Yr Mortgage Rate',
    'shelter_cpi':     'House Price Index (USSTHPI)'
}

fig, axes = plt.subplots(nrows=len(series_to_decompose), ncols=4, figsize=(20, 12))
fig.suptitle('Seasonal Decomposition — Key Housing Indicators\n(Trend | Seasonal | Residual | Original)',
             fontsize=13, fontweight='bold', y=1.01)

for row_idx, (col_name, label) in enumerate(series_to_decompose.items()):
    series = data[col_name].dropna()

    # Convert PeriodIndex to DatetimeIndex for statsmodels compatibility
    series_dt = series.copy()
    series_dt.index = series_dt.index.to_timestamp()

    # Decompose — period=12 means annual seasonality (12 months)
    decomposition = seasonal_decompose(series_dt, model='additive', period=12)

    components = {
        'Original':   series_dt,
        'Trend':      decomposition.trend,
        'Seasonal':   decomposition.seasonal,
        'Residual':   decomposition.resid
    }

    col_order = ['Trend', 'Seasonal', 'Residual', 'Original']
    colors     = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

    for col_idx, (comp_name, comp_data) in enumerate(zip(col_order, [components[c] for c in col_order])):
        ax = axes[row_idx][col_idx]
        ax.plot(comp_data, color=colors[col_idx], linewidth=0.9)
        if row_idx == 0:
            ax.set_title(comp_name, fontsize=10, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decomposition_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: decomposition_plots.png")

# ─────────────────────────────────────────────
# STEP 3: ADF STATIONARITY TEST
#
# What is stationarity?
#   A series is "stationary" if its mean, variance, and autocorrelation
#   do NOT change over time. ARIMA requires stationary data.
#
# ADF Test (Augmented Dickey-Fuller):
#   H0 (null hypothesis):  Series HAS a unit root → NOT stationary
#   H1 (alternative):      Series does NOT have unit root → IS stationary
#
#   If p-value < 0.05 → reject H0 → series IS stationary ✓
#   If p-value ≥ 0.05 → fail to reject H0 → series is NOT stationary ✗
#                        → need to first-difference it
# ─────────────────────────────────────────────
print("\n[3] Running ADF Stationarity Tests ...")
print("-" * 60)

def run_adf_test(series, name, differenced=False):
    """
    Run ADF test on a series.
    Returns: dict with test results
    """
    series_clean = series.dropna()
    series_dt = series_clean.copy()
    if hasattr(series_dt.index, 'to_timestamp'):
        series_dt.index = series_dt.index.to_timestamp()

    result = adfuller(series_dt, autolag='AIC')  # AIC picks optimal lag automatically

    adf_stat  = result[0]   # Test statistic (more negative = more likely stationary)
    p_value   = result[1]   # p-value
    n_lags    = result[2]   # Number of lags used
    n_obs     = result[3]   # Number of observations used
    crit_vals = result[4]   # Critical values at 1%, 5%, 10%

    is_stationary = p_value < 0.05
    status = "✓ STATIONARY" if is_stationary else "✗ NON-STATIONARY"

    label = f"{name} {'(1st diff)' if differenced else '(level)'}"
    print(f"  {label:<45} p={p_value:.4f}  {status}")

    return {
        'Series':         label,
        'ADF Statistic':  round(adf_stat, 4),
        'p-value':        round(p_value, 4),
        'Lags Used':      n_lags,
        'Obs Used':       n_obs,
        'Crit Val 1%':    round(crit_vals['1%'], 3),
        'Crit Val 5%':    round(crit_vals['5%'], 3),
        'Crit Val 10%':   round(crit_vals['10%'], 3),
        'Stationary?':    'Yes' if is_stationary else 'No',
        'Differenced':    differenced
    }

# ─────────────────────────────────────────────
# STEP 4: TEST ALL SERIES — LEVEL (original)
# ─────────────────────────────────────────────
print("\n  --- LEVEL (original series) ---")

series_names = {
    'housing_starts': 'Housing Starts',
    'mortgage_rate':  'Mortgage Rate',
    'shelter_cpi':    'House Price Index',
    'pce':            'PCE',
    'unemployment':   'Unemployment Rate',
    'fed_funds':      'Fed Funds Rate'
}

results_list = []
non_stationary = []   # We'll track which ones need differencing

for col, name in series_names.items():
    res = run_adf_test(data[col], name, differenced=False)
    results_list.append(res)
    if res['Stationary?'] == 'No':
        non_stationary.append(col)

# ─────────────────────────────────────────────
# STEP 5: FIRST DIFFERENCE non-stationary series
#
# What is first differencing?
#   Instead of analyzing the level (e.g., price = 250),
#   we analyze the CHANGE (e.g., price this month − price last month = +3)
#   This removes trend and often makes a series stationary.
#
#   Δy_t = y_t − y_{t-1}
#
# If 1st diff is still non-stationary → 2nd difference (rare)
# Most macro series become stationary after 1 difference → "I(1)" series
# ─────────────────────────────────────────────
print(f"\n  --- 1st DIFFERENCE (for non-stationary series: {[series_names[c] for c in non_stationary]}) ---")

data_diff = data.copy()
diff_results = []

for col in non_stationary:
    data_diff[col + '_d1'] = data[col].diff()   # .diff() = subtract previous value
    res = run_adf_test(data_diff[col + '_d1'], series_names[col], differenced=True)
    diff_results.append(res)
    results_list.append(res)

# ─────────────────────────────────────────────
# STEP 6: SAVE STATIONARITY SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n[4] Saving stationarity summary ...")

results_df = pd.DataFrame(results_list)
results_df.to_csv('stationarity_summary.csv', index=False)
results_df.to_excel('stationarity_summary.xlsx', index=False)
print("    ✓ Saved: stationarity_summary.csv")
print("    ✓ Saved: stationarity_summary.xlsx")

# ─────────────────────────────────────────────
# STEP 7: PLOT — LEVEL vs. DIFFERENCED (for key series)
# Shows visually why differencing helps
# ─────────────────────────────────────────────
print("\n[5] Plotting level vs. differenced comparison ...")

fig, axes = plt.subplots(nrows=len(non_stationary), ncols=2, figsize=(16, 3.5 * len(non_stationary)))
if len(non_stationary) == 1:
    axes = [axes]

fig.suptitle('Level vs. First Difference — Non-Stationary Series',
             fontsize=13, fontweight='bold')

for i, col in enumerate(non_stationary):
    name = series_names[col]

    # Level
    series_level = data[col].dropna()
    series_level_dt = series_level.copy()
    series_level_dt.index = series_level_dt.index.to_timestamp()

    axes[i][0].plot(series_level_dt, color='#E53935', linewidth=0.9)
    axes[i][0].set_title(f'{name} — Level (non-stationary)', fontsize=10)
    axes[i][0].set_ylabel('Level', fontsize=8)
    axes[i][0].grid(True, alpha=0.3)
    axes[i][0].tick_params(axis='x', rotation=45, labelsize=7)

    # First difference
    series_d1 = data_diff[col + '_d1'].dropna()
    series_d1_dt = series_d1.copy()
    series_d1_dt.index = series_d1_dt.index.to_timestamp()

    axes[i][1].plot(series_d1_dt, color='#1E88E5', linewidth=0.9)
    axes[i][1].axhline(y=0, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
    axes[i][1].set_title(f'{name} — 1st Difference (stationary ✓)', fontsize=10)
    axes[i][1].set_ylabel('Δ (Change)', fontsize=8)
    axes[i][1].grid(True, alpha=0.3)
    axes[i][1].tick_params(axis='x', rotation=45, labelsize=7)

plt.tight_layout()
plt.savefig('stationarity_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: stationarity_comparison.png")

# ─────────────────────────────────────────────
# STEP 8: SAVE PREPARED DATA FOR ARIMA
# The differenced columns are now ready for modelling
# ─────────────────────────────────────────────
data_diff.index = data_diff.index.to_timestamp()
data_diff.to_csv('fred_data_prepared.csv')
data_diff.to_excel('fred_data_prepared.xlsx')
print("\n    ✓ Saved: fred_data_prepared.csv  (use this for ARIMA in Phase 3)")
print("    ✓ Saved: fred_data_prepared.xlsx (use this for Power BI)")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print("\nFiles generated:")
print("  decomposition_plots.png      → trend/seasonal/residual charts")
print("  stationarity_comparison.png  → level vs. differenced")
print("  stationarity_summary.csv     → ADF test results table")
print("  stationarity_summary.xlsx    → same (for Power BI)")
print("  fred_data_prepared.csv       → clean data ready for ARIMA")
print("  fred_data_prepared.xlsx      → same (for Power BI)")
print("\nNext step → Phase 3: ARIMA / ARIMAX modelling")
print("=" * 60)
