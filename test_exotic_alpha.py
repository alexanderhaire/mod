"""
Exotic Alpha Test
=================

Exploring "Deep" and "Absurd" Alpha sources.

1.  **Alternative Data**: Correlating Stock Returns with "Weird" Annual Data (Butter, Netflix, etc.)
2.  **Fractal Analysis**: Calculating Hurst Exponent (Trend Persistence).
3.  **HMM**: Hidden Markov Models for Regime Detection.

Universe: XLB, XLE, JNK, GLD, MTUM (The Holy Grail)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- 1. ALTERNATIVE DATA (Hardcoded from chemical scanner) ---
WEIRD_DATA = {
    "butter_consumption": {2010: 4.9, 2011: 5.0, 2012: 5.2, 2013: 5.3, 2014: 5.5, 2015: 5.6, 2016: 5.7, 2017: 5.8, 2018: 5.9, 2019: 6.0, 2020: 6.1, 2021: 6.2, 2022: 6.3, 2023: 6.5, 2024: 6.8},
    "netflix_subscribers": {2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5, 2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5, 2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0},
    "national_park_visits": {2010: 281.3, 2011: 278.7, 2012: 282.8, 2013: 273.6, 2014: 292.8, 2015: 307.2, 2016: 331.0, 2017: 330.9, 2018: 318.2, 2019: 327.5, 2020: 237.0, 2021: 297.1, 2022: 312.0, 2023: 325.0, 2024: 331.9},
    "starbucks_stores": {2010: 10.6, 2011: 10.8, 2012: 11.2, 2013: 11.6, 2014: 12.0, 2015: 12.5, 2016: 13.0, 2017: 13.5, 2018: 14.2, 2019: 15.0, 2020: 15.3, 2021: 15.7, 2022: 15.9, 2023: 16.4, 2024: 16.9},
}

ASSETS = ['XLB', 'XLE', 'JNK', 'GLD', 'MTUM', 'XLK'] # Added XLK for Netflix check

def fetch_data() -> pd.DataFrame:
    # 2010 to present to match weird data
    data = yf.download(ASSETS, start="2010-01-01", progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.ffill().dropna()

def test_weird_correlations(prices):
    print("\n   --- TEST 1: WEIRD CORRELATIONS (Annual) ---")
    annual_prices = prices.resample('YE').last()
    annual_idx = annual_prices.index.year
    
    for asset in ASSETS:
        if asset not in prices.columns: continue
        
        # Series of price (or return?)
        # Let's correlate Weird Data Level with Asset Price Level (Spurious?)
        # Or Weird Growth with Asset Return (Cleaner).
        # Let's do Level vs Level (Classic "Butter predicts S&P")
        
        y_vals = []
        common_years = []
        for y in range(2010, 2025):
            if y in annual_idx:
                # Find the row with year y
                try:
                    p = annual_prices.loc[str(y)].iloc[-1][asset] if isinstance(annual_prices.loc[str(y)], pd.DataFrame) else annual_prices.loc[str(y)][asset]
                    # Handle Series/DataFrame ambiguity with resample
                    # resample('YE') result index is last day of year.
                    # annual_prices is a DataFrame with DatetimeIndex.
                    # We need integer year matching.
                    row = annual_prices[annual_prices.index.year == y]
                    if not row.empty:
                        y_vals.append(row[asset].values[0])
                        common_years.append(y)
                except:
                    pass
                    
        if len(common_years) < 10: continue

        for weird_name, weird_dict in WEIRD_DATA.items():
            x_vals = [weird_dict[y] for y in common_years if y in weird_dict]
            # align y_vals
            # (Re-align carefully)
            xy = []
            for y in common_years:
                if y in weird_dict:
                    # Look up price
                    price = annual_prices[annual_prices.index.year==y][asset].values[0]
                    xy.append((weird_dict[y], price))
            
            if len(xy) < 10: continue
            
            X = [p[0] for p in xy]
            Y = [p[1] for p in xy]
            
            corr, pval = spearmanr(X, Y)
            if abs(corr) > 0.8 and pval < 0.05:
                print(f"   🤯 SPURIOUS ALERT: {weird_name} vs {asset}: corr={corr:.2f} (p={pval:.3f})")

# --- 2. FRACTAL ANALYSIS ---
def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def test_fractals(prices):
    print("\n   --- TEST 2: FRACTAL MARKET HYPOTHESIS ---")
    print(f"   {'Asset':<8} {'Hurst':<8} {'Regime'}")
    print("   " + "-" * 40)
    
    for asset in ASSETS:
        if asset not in prices.columns: continue
        series = prices[asset].values
        # Use log prices
        h = get_hurst_exponent(np.log(series), max_lag=100)
        
        regime = "Random"
        if h > 0.55: regime = "Trending 📈"
        elif h < 0.45: regime = "Reverting 📉"
        
        print(f"   {asset:<8} {h:.3f}    {regime}")
        
# --- 3. HIDDEN MARKOV MODELS ---
def test_hmm(prices):
    print("\n   --- TEST 3: HIDDEN MARKOV MODELS ---")
    # Test on GLD (Gold)
    asset = 'GLD'
    if asset not in prices.columns: return
    
    returns = prices[asset].pct_change().dropna().values.reshape(-1, 1)
    
    # Fit HMM (2 States: Bull/Bear or Low/High Vol)
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    hmm.fit(returns)
    
    hidden_states = hmm.predict(returns)
    
    # Analyze States
    means = hmm.means_.flatten()
    vars = hmm.covars_.flatten() # For diag, this is just array of variances
    
    print(f"   Asset: {asset}")
    for i in range(hmm.n_components):
        print(f"   State {i}: Mean Ret {means[i]*252:.1%} | Vol {np.sqrt(vars[i])*np.sqrt(252):.1%}")
        
    # Last State
    print(f"   Current Regime: State {hidden_states[-1]}")
    
def run_exotic_tests():
    print("=" * 80)
    print("   EXOTIC ALPHA FRONTIERS")
    print("=" * 80)
    
    prices = fetch_data()
    
    test_weird_correlations(prices)
    test_fractals(prices)
    test_hmm(prices)
    
    print("\n   ✅ EXOTIC TESTS COMPLETE.")

if __name__ == "__main__":
    run_exotic_tests()
