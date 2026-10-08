# US Housing & Consumer Market Indicators: Time Series Analysis

**Ruchita Anil Zingade | MA Economics, Madras School of Economics (Semester III)**  
**Course: Applied Macro and Financial Econometrics**

---

## Project Overview

This project analyses the dynamics of the US residential housing market using monthly macroeconomic data from the Federal Reserve Economic Data (FRED) database, spanning January 2000 to September 2026 (321 observations).

The central research question is:

> **Do monetary policy changes, specifically mortgage rate movements, predict housing starts, and if so, with what transmission lag?**

This is an economics question about monetary policy transmission. When the Fed raises rates, mortgage rates follow. Higher mortgage rates make home loans more expensive. Developers and buyers pull back. But this doesn't happen overnight, there's a delay between the rate move and the construction response. The entire project is built to measure that delay.

The analysis proceeds in four phases, from raw data extraction through ARIMA/ARIMAX modelling and sub-national extensions, combining classical time series econometrics with model diagnostics.

---

## Key Findings

- US housing starts follow an **ARIMA(0,1,1)** process a single lagged shock carries over month to month after first differencing
- **Granger causality** from mortgage rates to housing starts is significant at lag 6 (F=2.803, p=0.011), but not at lags 1 or 3, consistent with a 6-month monetary policy transmission lag
- **Lagged ARIMAX** confirms: mortgage rate at t-3 (coef=-62.5, p=0.003) and t-6 (coef=-73.4, p=0.002) are both highly significant; the contemporaneous rate is not (p=0.34)
- A **COVID structural break dummy** (March 2020 to June 2021) is highly significant (coef=-109.5, p=0.002), isolating the pandemic shock from underlying market dynamics
- **12-month forecast** (Oct 2026 to Sep 2027): housing starts projected in the range of 1,204 to 1,402 thousand units annualised

---

## Data Sources

All data extracted via the [FRED API](https://fred.stlouisfed.org/) (fredapi, Python):

| Variable | FRED Series | Description |
|----------|------------|-------------|
| Housing Starts | `HOUST` | New residential units started (000s, annualised, monthly) |
| 30-Yr Mortgage Rate | `MORTGAGE30US` | Weekly average, interpolated to monthly |
| US House Price Index | `USSTHPI` | All-transactions HPI (used as shelter price proxy) |
| Personal Consumption | `PCE` | Personal Consumption Expenditures ($bn) |
| Unemployment Rate | `UNRATE` | Civilian unemployment rate (%) |
| Fed Funds Rate | `FEDFUNDS` | Effective federal funds rate (%) |

**Period:** January 2000 to September 2026 | **Frequency:** Monthly

![Macroeconomic Overview](macro_overview.png)

The key transmission chain is: Fed Funds Rate → Mortgage Rate → Housing Starts. We can't just put Fed Funds directly into a housing model without the mortgage rate as the intermediary, because the Fed doesn't lend to homebuyers, banks do, at the mortgage rate.
---

## Project Structure

```
US_housing/
|
+-- phase1_data_pull.py          # FRED API extraction + cleaning
+-- phase2_stationarity.py       # Seasonal decomposition + ADF stationarity tests
+-- phase3_arima.py              # ARIMA / SARIMA / ARIMAX modelling + 12M forecast
+-- phase4_extensions.py         # Granger causality + COVID dummy + lagged ARIMAX
|
+-- timeseries_analysis.ipynb    # Time series methods applied to FRED data
|                                  (White noise, AR, MA, ARMA, ARIMA, ARIMAX)
|
+-- outputs/
|   +-- fred_data.csv                  # Raw extracted data
|   +-- fred_data_prepared.csv         # Cleaned + differenced data (model-ready)
|   +-- stationarity_summary.xlsx      # ADF test results for all series
|   +-- arima_results.xlsx             # Model comparison, forecast, coefficients
|   +-- phase4_results.xlsx            # Granger causality, lagged ARIMAX, comparison
|   |
|   +-- decomposition_plots.png        # Trend / seasonal / residual decomposition
|   +-- stationarity_comparison.png    # Level vs. first-differenced series
|   +-- arima_results.png              # ARIMA/ARIMAX model outputs + forecast chart
|   +-- phase4_extensions.png          # COVID dummy + lagged mortgage rate visuals
|
+-- README.md
```

---

## Methods

### Phase 1: Data Extraction
Monthly FRED data pulled via API for 6 macroeconomic series. Missing values handled via forward-fill (2 months or fewer); series aligned to a common monthly index. 

Forward-fill for missing values: up to 2 months. This handles the fact that some series (mortgage rate is weekly, interpolated to monthly) occasionally have gaps.
Aligned all series to a common monthly index: since FRED series don't all start or end on the same date.
First differences computed: for each series, you stored both the level (housing_starts) and the first difference (housing_starts_d1). This is important because most time series models require stationary data

### Phase 2: Decomposition and Stationarity
- **Seasonal decomposition** (additive model, period=12): trend, seasonal, and residual components extracted for housing starts, mortgage rate, and HPI.
- used an additive decomposition (as opposed to multiplicative) with period=12 because the data is monthly. Additive is appropriate when the seasonal swings don't grow proportionally with the level, winter suppression of starts is roughly the same ±200k whether the overall level is 600k or 1,600k.
- **What the decomposition plots showed:**
Housing starts trend: clear boom (pre-2006), collapse (2008–2012), slow recovery, COVID disruption
The seasonal component of housing starts is stable and regular: roughly +100 to +150k in spring, -100 to -150k in winter
Mortgage rate has almost no seasonality, it's driven by policy, not the calendar
Residuals for housing starts spike dramatically in 2020 (COVID): the model can't explain that from trend + seasonal alone, which motivates the COVID dummy later

**Why does stationarity matter?**

A time series is stationary if its statistical properties — mean, variance, autocorrelation structure, don't change over time.

If we run a regression on two non-stationary series, we can get a spurious regression: a high R², significant t-statistics, but the relationship is completely meaningless. The classic example: global temperature and the number of pirates both trend over time, regress them and you'd find a "significant" negative relationship. That's not causal, it's just two trending series moving together.

To run valid ARIMA models and regressions, we need stationary data.

What does non-stationary look like?

A series with a unit root is non-stationary. **Informally: the series doesn't revert to a mean. It just wanders.** Housing starts over 2000–2026 wanders — it was 2,000+ in 2006 and 490 in 2009. It doesn't hover around a fixed value.

The Augmented Dickey-Fuller (ADF) Test
The ADF test formally tests for a unit root.

**Null hypothesis (H₀):** the series has a unit root (is non-stationary)
**Alternative hypothesis (H₁):** the series is stationary

You reject H₀ when the ADF statistic is sufficiently negative (more negative than the critical value) OR when the p-value is below your threshold (usually 0.05).
- **Augmented Dickey-Fuller (ADF) test** applied to all series at level and after first differencing
- Housing starts: p=0.47 at level (non-stationary) → p<0.001 after first differencing (stationary) → integrated of order I(1)

## Stationarity Test Results (ADF Test)

| Series | Level p-value | Stationary at level? | 1st diff p-value | Stationary after diff? |
|---|---|---|---|---|
| Housing Starts | 0.623 | No | 0.000 | Yes → I(1) |
| Mortgage Rate | 0.250 | No | 0.000 | Yes → I(1) |
| HPI | 0.969 | No | 0.090 | Borderline |
| PCE | 1.000 | No | 0.002 | Yes → I(1) |
| Unemployment | 0.036 | Yes → I(0) | — | — |
| Fed Funds | 0.004 | Yes → I(0) | — | — |

> **I(0)** = integrated of order 0 = stationary at level  
> **I(1)** = integrated of order 1 = stationary only after first differencing

Housing starts, mortgage rate, and PCE are all I(1). This is expected — they all have clear long-run trends. Unemployment and Fed Funds are I(0), which also makes sense — they mean-revert (unemployment fluctuates around a structural rate, Fed Funds gets cut and raised in cycles).

![Seasonal Decomposition](decomposition_plots.png)

![Stationarity: Levels vs First Differences](stationarity_comparison.png)

### Phase 3: ARIMA / SARIMA / ARIMAX Modelling
Train/test split: last 12 months held out for out-of-sample evaluation.
ARIMAX
ARIMAX = ARIMA with eXogenous variables. add external predictors (mortgage rate, fed funds rate) to the right-hand side.
This is now asking: after controlling for the ARIMA dynamics (the inertia and shock-propagation), do mortgage rates add additional predictive power?

AIC (Akaike Information Criterion) balances model fit against complexity:

$$AIC = 2k - 2\ln(L)$$

where $k$ = number of parameters, $L$ = likelihood of the model. **Lower AIC is better.**

 ARIMAX: ARIMA(0,1,1) + mortgage rate + fed funds rate
| Model | Specification | AIC | Test RMSE |
|-------|--------------|-----|-----------|
| ARIMA | (0,1,1) | 3694.6 | 88.5 |
| SARIMA | (3,1,0)x(2,1,0,12) | 3673.3 | 112.9 |
| **ARIMAX** | **(0,1,1) + mortgage + fed funds** | **3672.2** | **90.3** |

ARIMAX selected as best model by AIC. 12-month forecast produced using SARIMA refitted on full sample.
> **ARIMAX wins on AIC.** Note that SARIMA has *worse* out-of-sample RMSE than plain ARIMA despite a better AIC, a classic sign of overfitting: the seasonal model fits training data well but generalises poorly to new data. ARIMAX strikes the best balance between fit and parsimony.
>  **Note on the forecast chart:** The 12-month fan chart (Oct 2026–Sep 2027) is generated 
> from SARIMA(3,1,0)(2,1,0,12) refitted on the full sample, not from ARIMAX. ARIMAX is the 
> best model by AIC and captures the mortgage rate transmission mechanism — but it requires 
> future values of mortgage rate and fed funds rate as inputs, which are unknown over the 
> forecast horizon. SARIMA is purely univariate and self-contained, making it operationally 
> suitable for a standalone multi-step forecast. The two models serve different purposes: 
> ARIMAX answers *why* starts move, SARIMA answers *where* starts are headed.
> 
![ARIMA/ARIMAX Results and 12-Month Forecast](arima_results.png)

### Phase 4: Extensions
## Granger Causality: Does Mortgage Rate Predict Housing Starts?

Granger causality asks a precise question: does knowing past mortgage rates help predict housing starts, **beyond what housing starts' own history already tells us?** It is a statement about **predictive content**, not true mechanism.

### The Two-Regression Setup

**Restricted model**  housing starts predicted by its own lags only:

    Starts_t = a₁·Starts_{t-1} + a₂·Starts_{t-2} + ... + aₖ·Starts_{t-k} + ε

**Unrestricted model** adds lagged mortgage rates:

    Starts_t = a₁·Starts_{t-1} + ... + aₖ·Starts_{t-k}
             + b₁·Mortgage_{t-1} + ... + bₖ·Mortgage_{t-k} + ε

An **F-test** compares whether adding mortgage rate lags meaningfully reduces forecast error. If yes → mortgage rate Granger-causes housing starts.

### Results by Lag

| Lag | p-value | Significant? | Interpretation |
|---|---|---|---|
| Lag 1 | 0.612 | No | Too soon, market hasn't had time to respond |
| Lag 3 | 0.169 | No | Still within decision pipeline, not yet visible in starts |
| Lag 6 | 0.011 | ✅ Yes (5%) | Rate change → permit pulled → groundbreaking ~6 months later |
| Lag 12 | 0.057 | Marginal (10%) | Signal weakens, too many intervening factors |

### Reverse Direction: Bidirectionality

Running the test in reverse (do housing starts Granger-cause mortgage rates?) yields significance at lags 1–3. A construction boom raises mortgage demand, tightening credit and pushing rates up. This bidirectionality means mortgage rate is not a fully clean exogenous predictor — a **structural VAR** would be needed to cleanly separate cause from effect; flagged as a limitation of the ARIMAX specification.

### Key Finding

> Mortgage rates from **6 months prior** have statistically significant predictive content for housing starts (p = 0.011). This 3–6 month transmission lag reflects the decision-to-groundbreaking pipeline: rate rise → builder recalculates financing → permit withdrawn → starts fall.

## Phase 4 Extensions

### Structural Break: COVID Dummy

A structural break is a sudden, permanent shift in the relationship between variables, not just a large residual, but a change in the model's parameters or intercept.

COVID (March 2020) was a massive supply-side shock to construction: workers couldn't show up, lumber prices spiked, supply chains broke. This suppressed starts in a way entirely unrelated to mortgage rates.

**Dummy variable added:**

    COVID_dummy = 1 for March 2020 – June 2021, else 0

**Result:** coefficient = −109.5, p = 0.002. The pandemic suppressed housing starts by approximately **110,000 units per month** on average during that period, holding everything else constant.

Without this dummy, the model would try to explain the 2020 crash through mortgage rates and AR terms — which it can't. Residuals would spike, diagnostics would fail, and standard errors would inflate. The dummy isolates the pandemic as a separate regime, letting the model fit pre- and post-COVID periods cleanly.

---

### Lagged ARIMAX: Quantifying the Transmission Lag

Mortgage rate entered the model at three time points simultaneously:

- **Mortgage(t)** — the rate this month
- **Mortgage(t−3)** — the rate 3 months ago  
- **Mortgage(t−6)** — the rate 6 months ago

| Variable | Coefficient | p-value | Significant? |
|---|---|---|---|
| Mortgage rate (t) | −23.6 | 0.34 | No |
| Mortgage rate (t−3) | −62.5 | 0.003 | ✅ Yes |
| Mortgage rate (t−6) | −73.4 | 0.002 | ✅ Yes |
| Fed funds rate | +61.6 | <0.01 | ✅ Yes |
| COVID dummy | −100.8 | 0.027 | ✅ Yes |

### Key Finding

> The contemporaneous mortgage rate is **insignificant**, but lags at 3 and 6 months are highly significant with large negative coefficients. A 1 percentage point rise in mortgage rates reduces starts by ~62k units after 3 months and ~73k units after 6 months. This directly confirms the Granger causality result and **quantifies the transmission lag**.

The positive fed funds coefficient (+61.6) may appear counterintuitive. Once the mortgage rate is controlled for, the fed funds rate partly captures broader economic expansion, a rising funds rate can coincide with a stronger economy that supports construction activity. This is a partial effect and multicollinearity interpretation issue, flagged as a limitation.
![Phase 4: COVID Dummy and Lagged Mortgage Rate](phase4_extensions.png)

---

## Reproducibility

### Requirements
```
Python 3.9+
fredapi
pandas
numpy
matplotlib
statsmodels
pmdarima
openpyxl
```

### Setup
```bash
# Clone the repo
git clone https://github.com/ruchitaanil03/us-housing-timeseries.git
cd us-housing-timeseries

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate        # Mac/Linux
myenv\Scripts\activate           # Windows

# Install dependencies
pip install fredapi pandas numpy matplotlib statsmodels pmdarima openpyxl jupyter

# Run phases in order
python3 phase1_data_pull.py
python3 phase2_stationarity.py
python3 phase3_arima.py
python3 phase4_extensions.py

# Open the time series analysis notebook
jupyter notebook timeseries_analysis.ipynb
```

**FRED API Key:** A free key is required. Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and replace the `FRED_API_KEY` variable in `phase1_data_pull.py`.

---

## Economic Motivation

The analysis is motivated by the housing market's central role in consumer spending and home improvement retail. Key transmission mechanisms examined:

1. **Rate → Permits → Starts:** A mortgage rate rise raises financing costs, reducing permit applications; this flows through to housing starts with a 3 to 6 month lag
2. **Starts → Home improvement demand:** New construction and housing transactions are the primary driver of large-ticket home improvement spending (flooring, appliances, fixtures)

## The Home Improvement Connection

This project is not purely a forecasting exercise. The time series econometrics serves as evidence for a broader consumer behaviour argument.

### The Logic Chain

Housing starts is a leading indicator — when builders break ground, downstream spending follows: furniture, appliances, paint, fixtures, landscaping. But the argument runs deeper than new construction alone.

The **mortgage rate → housing starts** link has a second channel: the **lock-in effect**. When rates rise sharply (as they did 2022–2023), existing homeowners stop moving — refinancing their current mortgage at 7% to buy elsewhere makes no financial sense. So they stay. And instead of buying new, they spend on **renovating what they have**: kitchens, bathrooms, extensions.

### How the Variables Connect

**Mortgage rate (MORTGAGE30US)** is the primary driver. High rates suppress not just new construction but existing home sales too — fewer transactions means fewer move-in purchases and more "make do and improve" spending.

**HPI (House Price Index)** is the wealth channel. Rising home values make owners feel wealthier and more willing to invest in improvements — the home is appreciating, so spending on it feels rational.

**PCE (Personal Consumption Expenditure)** is where home improvement spending actually shows up in the data — durable goods (appliances, fixtures) and services (contractors, landscaping). The KPSS conflict on PCE, non-stationary even after differencing, reflects how persistent consumer spending trends are: consistent with people committing to multi-year improvement cycles, not one-off purchases.

### What the Forecast Implies

The SARIMA forecast of **1,204–1,402k starts** for Oct 2026–Sep 2027 sits below pre-pandemic norms (1,500k+ during 2017–2019). That sustained suppression of new construction, combined with elevated rates keeping existing homeowners in place, is the macro backdrop for the home improvement thesis.

> **Demand shifts from "buy new" to "improve existing."** The Granger causality and lagged ARIMAX results establish mortgage rates as a statistically significant predictor of housing starts. The level of starts — combined with the lock-in effect — then predicts where consumer spending on home improvement goes. The econometrics is the evidence base for that story.

---

## Limitations and Future Work

- **Contemporaneous exogeneity:** Mortgage rate and fed funds are treated as exogenous; in practice, the Fed responds to economic conditions that also affect housing. A structural VAR framework would address this.
- **ARCH/GARCH:** Volatility clustering in housing starts residuals is not modelled here; conditional heteroskedasticity models are a natural extension.
- **MSA-level modelling:** Individual ARIMA models per metro and panel time series approaches are left for future work.
- **Supply-side variables:** Lumber costs, construction employment, and zoning stringency indices are omitted; their inclusion would improve structural interpretation.

---

## Author

**Ruchita Anil Zingade**  
MA Economics (2025-27), Madras School of Economics, Chennai
Email: ge25ruchita@mse.ac.in

---

*Data: Federal Reserve Bank of St. Louis (FRED). All analysis conducted in Python 3. Charts generated with matplotlib and statsmodels.*
