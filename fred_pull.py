# ============================================================
# STEP 1: IMPORT LIBRARIES
# These are pre-built tools we use so we don't write from scratch
# ============================================================

import pandas as pd                          # for data tables (like Excel in Python)
import matplotlib.pyplot as plt              # for plotting graphs
from fredapi import Fred                     # to connect to FRED database
import warnings
warnings.filterwarnings('ignore')            # suppresses minor warnings

# ============================================================
# STEP 2: CONNECT TO FRED
# Paste your API key here after registering on FRED website
# ============================================================

fred = Fred(api_key='YOUR_FRED_API_KEY_HERE')

# ============================================================
# STEP 3: PULL ALL DATA SERIES
# Each string like 'HOUST' is the FRED series code
# observation_start = from which date you want data
# ============================================================

print("Pulling data from FRED...")

housing_starts   = fred.get_series('HOUST',          observation_start='2000-01-01')
mortgage_rate    = fred.get_series('MORTGAGE30US',    observation_start='2000-01-01')
shelter_cpi      = fred.get_series('USSTHPI', observation_start='2000-01-01')
pce              = fred.get_series('PCE',             observation_start='2000-01-01')
unemployment     = fred.get_series('UNRATE',          observation_start='2000-01-01')
fed_funds        = fred.get_series('FEDFUNDS',        observation_start='2000-01-01')

print("Data pulled successfully!")

# ============================================================
# STEP 4: COMBINE INTO ONE TABLE
# pd.DataFrame puts all series into one neat table
# Each series becomes a column
# ============================================================

data = pd.DataFrame({
    'housing_starts': housing_starts,
    'mortgage_rate':  mortgage_rate,
    'shelter_cpi':    shelter_cpi,
    'pce':            pce,
    'unemployment':   unemployment,
    'fed_funds':      fed_funds
})

# ============================================================
# STEP 5: CLEAN THE DATA
# Mortgage rate is weekly, everything else is monthly
# We resample it to monthly by taking the average of each month
# ffill = forward fill — fills any empty cells with previous value
# ============================================================

data = data.resample('MS').mean()   # MS = Month Start frequency
data = data.ffill()                  # fill any missing values
data = data.dropna()                 # drop any remaining empty rows

print(f"\nData shape: {data.shape}")        # tells you rows x columns
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print("\nFirst 5 rows:")
print(data.head())                   # shows first 5 rows so you can verify

# ============================================================
# STEP 6: SAVE TO CSV AND EXCEL
# CSV for backup, Excel for Power BI import
# ============================================================

data.to_csv('fred_data.csv')
data.to_excel('fred_data_powerbi.xlsx')   # Power BI reads this directly

print("\nData saved to fred_data.csv and fred_data_powerbi.xlsx")

# ============================================================
# STEP 7: PLOT ALL SERIES — QUICK VISUAL CHECK
# Just to see what the data looks like before analysis
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))  
# 3 rows, 2 columns = 6 subplots, one per series

fig.suptitle('U.S. Macro Indicators — FRED Data (2000–2026)', 
             fontsize=14, fontweight='bold')

# Plot each series in its own subplot
data['housing_starts'].plot(ax=axes[0,0], title='Housing Starts (HOUST)', color='steelblue')
data['mortgage_rate'].plot(ax=axes[0,1],  title='30-Year Mortgage Rate',   color='darkorange')
data['shelter_cpi'].plot(ax=axes[1,0],    title='Shelter CPI',             color='green')
data['pce'].plot(ax=axes[1,1],            title='Personal Consumption (PCE)', color='purple')
data['unemployment'].plot(ax=axes[2,0],   title='Unemployment Rate',       color='red')
data['fed_funds'].plot(ax=axes[2,1],      title='Federal Funds Rate',      color='brown')

plt.tight_layout()               # auto-adjusts spacing between subplots
plt.savefig('macro_overview.png', dpi=150)   # saves the chart as image
plt.show()

print("\nChart saved as macro_overview.png")
print("\nPhase 1 Complete!")