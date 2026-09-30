"""
PHASE 3: ARIMA / SARIMA / ARIMAX Modelling
US Housing & Consumer Market Indicators — Time Series Analysis
Ruchita Anil Zingade | MSE Semester III

What this script does:
1. Loads the prepared data from Phase 2 (fred_data_prepared.csv)
2. Fits ARIMA model on Housing Starts (baseline)
3. Fits SARIMA model (captures seasonal pattern)
4. Fits ARIMAX model (adds mortgage rate + fed funds as external drivers)
5. Compares models using AIC
6. Forecasts 12 months ahead
7. Saves all outputs + forecast to Excel for Power BI
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 3: ARIMA / SARIMA / ARIMAX Modelling")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: LOAD PREPARED DATA
# ─────────────────────────────────────────────
print("\n[1] Loading prepared data ...")

data = pd.read_csv('fred_data_prepared.csv', index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index)
data.index.freq = 'MS'   # Monthly Start frequency

# Our target: Housing Starts
# Exogenous variables: mortgage rate, fed funds rate
y = data['housing_starts'].dropna()
exog_vars = data[['mortgage_rate', 'fed_funds']].dropna()

# Align all series to same date range
common_idx = y.index.intersection(exog_vars.index)
y = y[common_idx]
exog_vars = exog_vars.loc[common_idx]

print(f"    Target: housing_starts | {len(y)} observations")
print(f"    Period: {y.index[0].strftime('%Y-%m')} to {y.index[-1].strftime('%Y-%m')}")

# ─────────────────────────────────────────────
# STEP 2: TRAIN / TEST SPLIT
#
# We hold out the last 12 months as "test" set.
# The model trains on everything before that,
# then we see how well it predicts the held-out period.
# This is called "out-of-sample evaluation."
# ─────────────────────────────────────────────
FORECAST_HORIZON = 12   # months to forecast ahead

train_y    = y[:-FORECAST_HORIZON]
test_y     = y[-FORECAST_HORIZON:]
train_exog = exog_vars[:-FORECAST_HORIZON]
test_exog  = exog_vars[-FORECAST_HORIZON:]

print(f"\n    Train: {train_y.index[0].strftime('%Y-%m')} to {train_y.index[-1].strftime('%Y-%m')} ({len(train_y)} obs)")
print(f"    Test:  {test_y.index[0].strftime('%Y-%m')} to {test_y.index[-1].strftime('%Y-%m')} ({len(test_y)} obs)")

# ─────────────────────────────────────────────
# STEP 3: MODEL 1 — ARIMA (baseline)
#
# auto_arima tries all combinations of p and q
# (with d=1 since we found housing starts needs 1 difference)
# and picks the one with lowest AIC.
#
# AIC = Akaike Information Criterion
# Lower AIC = better model (balances fit vs complexity)
# ─────────────────────────────────────────────
print("\n[2] Fitting Model 1: ARIMA (auto-selecting p, d, q) ...")
print("    This may take 30-60 seconds ...")

arima_auto = auto_arima(
    train_y,
    d=1,                  # we know d=1 from ADF test
    start_p=0, max_p=4,
    start_q=0, max_q=4,
    seasonal=False,       # no seasonality in plain ARIMA
    information_criterion='aic',
    stepwise=True,        # faster search
    suppress_warnings=True,
    error_action='ignore'
)

arima_order = arima_auto.order   # (p, d, q)
arima_aic   = arima_auto.aic()

print(f"    ✓ Best ARIMA order: {arima_order}  |  AIC: {arima_aic:.2f}")

# Fit using statsmodels SARIMAX (more flexible, same result)
model_arima = SARIMAX(
    train_y,
    order=arima_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Forecast
arima_forecast = model_arima.get_forecast(steps=FORECAST_HORIZON)
arima_pred     = arima_forecast.predicted_mean
arima_ci       = arima_forecast.conf_int(alpha=0.05)   # 95% confidence interval
arima_pred.index = test_y.index
arima_ci.index   = test_y.index

# ─────────────────────────────────────────────
# STEP 4: MODEL 2 — SARIMA (with seasonality)
#
# SARIMA adds seasonal terms: (P, D, Q, m)
# m=12 means annual seasonality (12 months)
# P, D, Q are the seasonal equivalents of p, d, q
#
# From our decomposition plot, housing starts clearly
# has a seasonal pattern (peak in summer, dip in winter)
# SARIMA explicitly models this.
# ─────────────────────────────────────────────
print("\n[3] Fitting Model 2: SARIMA (with seasonal component) ...")
print("    This may take 1-2 minutes ...")

sarima_auto = auto_arima(
    train_y,
    d=1, D=1,             # 1 regular + 1 seasonal difference
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    m=12,                 # 12-month seasonal period
    seasonal=True,
    information_criterion='aic',
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

sarima_order         = sarima_auto.order
sarima_seasonal_order = sarima_auto.seasonal_order
sarima_aic           = sarima_auto.aic()

print(f"    ✓ Best SARIMA order: {sarima_order} x {sarima_seasonal_order}  |  AIC: {sarima_aic:.2f}")

model_sarima = SARIMAX(
    train_y,
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarima_forecast = model_sarima.get_forecast(steps=FORECAST_HORIZON)
sarima_pred     = sarima_forecast.predicted_mean
sarima_ci       = sarima_forecast.conf_int(alpha=0.05)
sarima_pred.index = test_y.index
sarima_ci.index   = test_y.index

# ─────────────────────────────────────────────
# STEP 5: MODEL 3 — ARIMAX (with external variables)
#
# ARIMAX = ARIMA + eXogenous variables
# We add mortgage_rate and fed_funds as regressors.
#
# Economic logic:
#   Higher mortgage rates → borrowing costs rise → fewer homes built
#   Higher fed funds rate → tighter credit → same effect
#
# The model estimates coefficients for each:
#   housing_starts = ARIMA(...) + β1*mortgage_rate + β2*fed_funds
#
# NOTE: We need to provide the exogenous variables for the
# forecast period too (test_exog) — that's fine because we
# have actual data for those months.
# ─────────────────────────────────────────────
print("\n[4] Fitting Model 3: ARIMAX (ARIMA + mortgage rate + fed funds) ...")

model_arimax = SARIMAX(
    train_y,
    exog=train_exog,
    order=arima_order,    # use same order as best ARIMA
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

arimax_aic = model_arimax.aic

print(f"    ✓ ARIMAX fitted  |  AIC: {arimax_aic:.2f}")

# Print coefficients — these are the interesting numbers
print("\n    ARIMAX Coefficients (effect on Housing Starts):")
for name, coef, pval in zip(
    model_arimax.param_names,
    model_arimax.params,
    model_arimax.pvalues
):
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    if name in ['mortgage_rate', 'fed_funds']:
        print(f"      {name:<20} coef={coef:>10.3f}   p={pval:.4f} {sig}")

arimax_forecast = model_arimax.get_forecast(steps=FORECAST_HORIZON, exog=test_exog)
arimax_pred     = arimax_forecast.predicted_mean
arimax_ci       = arimax_forecast.conf_int(alpha=0.05)
arimax_pred.index = test_y.index
arimax_ci.index   = test_y.index

# ─────────────────────────────────────────────
# STEP 6: MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n[5] Model Comparison:")
print("-" * 50)

def rmse(actual, predicted):
    """Root Mean Squared Error — average forecast error in original units"""
    return np.sqrt(np.mean((actual.values - predicted.values) ** 2))

def mae(actual, predicted):
    """Mean Absolute Error — average absolute forecast error"""
    return np.mean(np.abs(actual.values - predicted.values))

results = {
    'Model':  ['ARIMA',         'SARIMA',         'ARIMAX'],
    'Order':  [str(arima_order), f"{sarima_order}x{sarima_seasonal_order}", f"{arima_order}+exog"],
    'AIC':    [round(arima_aic, 2), round(sarima_aic, 2), round(arimax_aic, 2)],
    'RMSE':   [round(rmse(test_y, arima_pred), 1),
               round(rmse(test_y, sarima_pred), 1),
               round(rmse(test_y, arimax_pred), 1)],
    'MAE':    [round(mae(test_y, arima_pred), 1),
               round(mae(test_y, sarima_pred), 1),
               round(mae(test_y, arimax_pred), 1)]
}

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
print(f"\n    → Best model by AIC: {best_model_name}")

# ─────────────────────────────────────────────
# STEP 7: FUTURE FORECAST (12 months beyond data)
#
# Now we refit the best model on ALL data (not just train)
# and forecast 12 months into the future.
# For ARIMAX we need future values of exog vars —
# we use the last known values as a simple assumption.
# ─────────────────────────────────────────────
print("\n[6] Generating 12-month future forecast ...")

# Refit SARIMA on full data
final_model = SARIMAX(
    y,
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future_forecast = final_model.get_forecast(steps=FORECAST_HORIZON)
future_pred     = future_forecast.predicted_mean
future_ci       = future_forecast.conf_int(alpha=0.05)

# Create future date index
last_date    = y.index[-1]
future_index = pd.date_range(start=last_date + pd.DateOffset(months=1),
                             periods=FORECAST_HORIZON, freq='MS')
future_pred.index = future_index
future_ci.index   = future_index

print(f"    ✓ Forecast period: {future_index[0].strftime('%Y-%m')} to {future_index[-1].strftime('%Y-%m')}")
print(f"\n    12-Month Housing Starts Forecast:")
print(f"    {'Month':<12} {'Forecast':>10} {'Lower 95%':>12} {'Upper 95%':>12}")
print("    " + "-" * 48)
for date, pred, lo, hi in zip(future_index, future_pred,
                               future_ci.iloc[:, 0], future_ci.iloc[:, 1]):
    print(f"    {date.strftime('%Y-%m'):<12} {pred:>10.0f} {lo:>12.0f} {hi:>12.0f}")

# ─────────────────────────────────────────────
# STEP 8: PLOTS
# ─────────────────────────────────────────────
print("\n[7] Generating plots ...")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# --- Plot 1: ARIMA fit vs actual ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_y[-60:], color='#37474F', linewidth=1, label='Actual (train)')
ax1.plot(test_y,        color='#37474F', linewidth=1, linestyle='--', label='Actual (test)')
ax1.plot(arima_pred,    color='#1E88E5', linewidth=1.5, label=f'ARIMA{arima_order} forecast')
ax1.fill_between(arima_ci.index, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1],
                 alpha=0.2, color='#1E88E5', label='95% CI')
ax1.set_title(f'Model 1: ARIMA{arima_order}  |  AIC={arima_aic:.0f}', fontsize=10, fontweight='bold')
ax1.set_ylabel('Housing Starts (000s)', fontsize=8)
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45, labelsize=7)

# --- Plot 2: SARIMA fit vs actual ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(train_y[-60:], color='#37474F', linewidth=1, label='Actual (train)')
ax2.plot(test_y,        color='#37474F', linewidth=1, linestyle='--', label='Actual (test)')
ax2.plot(sarima_pred,   color='#43A047', linewidth=1.5, label=f'SARIMA forecast')
ax2.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                 alpha=0.2, color='#43A047', label='95% CI')
ax2.set_title(f'Model 2: SARIMA{sarima_order}x{sarima_seasonal_order}  |  AIC={sarima_aic:.0f}',
              fontsize=10, fontweight='bold')
ax2.set_ylabel('Housing Starts (000s)', fontsize=8)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45, labelsize=7)

# --- Plot 3: ARIMAX fit vs actual ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(train_y[-60:], color='#37474F', linewidth=1, label='Actual (train)')
ax3.plot(test_y,        color='#37474F', linewidth=1, linestyle='--', label='Actual (test)')
ax3.plot(arimax_pred,   color='#E53935', linewidth=1.5, label='ARIMAX forecast')
ax3.fill_between(arimax_ci.index, arimax_ci.iloc[:, 0], arimax_ci.iloc[:, 1],
                 alpha=0.2, color='#E53935', label='95% CI')
ax3.set_title(f'Model 3: ARIMAX (+ mortgage & fed funds)  |  AIC={arimax_aic:.0f}',
              fontsize=10, fontweight='bold')
ax3.set_ylabel('Housing Starts (000s)', fontsize=8)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45, labelsize=7)

# --- Plot 4: Model comparison (RMSE bar) ---
ax4 = fig.add_subplot(gs[1, 1])
models = ['ARIMA', 'SARIMA', 'ARIMAX']
rmses  = [results['RMSE'][0], results['RMSE'][1], results['RMSE'][2]]
colors = ['#1E88E5', '#43A047', '#E53935']
bars   = ax4.bar(models, rmses, color=colors, width=0.5)
for bar, val in zip(bars, rmses):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_title('Model Comparison: RMSE (lower = better)', fontsize=10, fontweight='bold')
ax4.set_ylabel('RMSE (Housing Starts units)', fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, max(rmses) * 1.3)

# --- Plot 5: 12-month future forecast ---
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(y[-36:], color='#37474F', linewidth=1.2, label='Historical (last 3 years)')
ax5.plot(future_pred, color='#7B1FA2', linewidth=2, label='12-Month Forecast (SARIMA)')
ax5.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                 alpha=0.2, color='#7B1FA2', label='95% Confidence Interval')
ax5.axvline(x=y.index[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax5.text(y.index[-1], ax5.get_ylim()[0] if ax5.get_ylim()[0] > 0 else 800,
         '  Forecast →', fontsize=8, color='gray')
ax5.set_title('12-Month Ahead Forecast: US Housing Starts', fontsize=11, fontweight='bold')
ax5.set_ylabel('Housing Starts (000s annualised)', fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45, labelsize=8)

fig.suptitle('US Housing Starts — ARIMA Family Model Results\nRuchita Anil Zingade | MSE Semester III',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('arima_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: arima_results.png")

# ─────────────────────────────────────────────
# STEP 9: SAVE ALL RESULTS TO EXCEL
# ─────────────────────────────────────────────
print("\n[8] Saving results to Excel ...")

with pd.ExcelWriter('arima_results.xlsx', engine='openpyxl') as writer:

    # Sheet 1: Model comparison
    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)

    # Sheet 2: Future forecast
    forecast_df = pd.DataFrame({
        'Date':          future_index.strftime('%Y-%m'),
        'Forecast':      future_pred.values.round(0),
        'Lower_95pct':   future_ci.iloc[:, 0].values.round(0),
        'Upper_95pct':   future_ci.iloc[:, 1].values.round(0)
    })
    forecast_df.to_excel(writer, sheet_name='Forecast_12M', index=False)

    # Sheet 3: ARIMA test predictions
    test_preds_df = pd.DataFrame({
        'Date':          test_y.index.strftime('%Y-%m'),
        'Actual':        test_y.values,
        'ARIMA_pred':    arima_pred.values.round(0),
        'SARIMA_pred':   sarima_pred.values.round(0),
        'ARIMAX_pred':   arimax_pred.values.round(0)
    })
    test_preds_df.to_excel(writer, sheet_name='Test_Predictions', index=False)

    # Sheet 4: ARIMAX coefficients
    coef_df = pd.DataFrame({
        'Variable': model_arimax.param_names,
        'Coefficient': model_arimax.params.round(4),
        'Std_Error': model_arimax.bse.round(4),
        'p_value': model_arimax.pvalues.round(4)
    })
    coef_df.to_excel(writer, sheet_name='ARIMAX_Coefficients', index=False)

print("    ✓ Saved: arima_results.xlsx (4 sheets)")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print("\nFiles generated:")
print("  arima_results.png   → all model plots + 12-month forecast")
print("  arima_results.xlsx  → model comparison, forecast, coefficients")
print(f"\nBest model: {best_model_name} (lowest AIC)")
print(f"12-month forecast range: {future_pred.min():.0f} – {future_pred.max():.0f} (000s)")
print("\nNext step → Phase 4: ARCH/GARCH volatility modelling")
print("=" * 60)
