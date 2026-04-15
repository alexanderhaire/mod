"""
Optimized Alpha Sniper (StatArb)
================================

Advanced filtering for Semantic High-Correlation pairs.
Goal: Filter "Noise" (Random Walk) from "Signal" (Mean Reversion).

Techniques:
1. Cointegration (Engle-Granger / ADF Test): Are they economically bound?
2. Hurst Exponent: Is the spread mean-reverting (H < 0.5)?
3. Half-Life: How fast does the spread revert?
4. Z-Score: Timing entry at > 2.0 Sigma.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

def calculate_hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def calculate_halflife(spread):
    """Calculate Half-Life of Mean Reversion via OU Process"""
    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    spread_ret = spread_ret.dropna()
    spread_lag = spread_lag.dropna()
    
    model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
    res = model.fit()
    halflife = -np.log(2) / res.params.iloc[1]
    return halflife

def check_cointegration(series_a, series_b):
    """Perform Engle-Granger Cointegration Test"""
    # 1. Regress A on B to find Hedge Ratio
    try:
        model = sm.OLS(series_a, sm.add_constant(series_b))
        res = model.fit()
        if len(res.params) < 2:
            return 1.0, pd.Series([0]*len(series_a)), 0.0
        hedge_ratio = res.params.iloc[1]
        spread = series_a - (hedge_ratio * series_b)
        
        # 2. Test Stationarity of Spread (ADF)
        adf_result = adfuller(spread)
        p_value = adf_result[1]
    except:
        return 1.0, pd.Series([0]*len(series_a)), 0.0
    
    return p_value, spread, hedge_ratio

def load_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        return df
    except:
        return None

def main():
    print("🎯 OPTIMIZED ALPHA SNIPER (STAT ARB)")
    print("====================================")
    
    df = load_data()
    if df is None: return

    # NLP Semantic Matching (Same as before)
    print("1️⃣  Scanning for Semantic Candidates...")
    df['text_feature'] = (df['Event'] + " " + df['Question']).fillna('')
    events = df.groupby('Question').agg({'text_feature': 'first'}).reset_index()
    
    if len(events) < 2:
        print("   Not enough events.")
        return

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(events['text_feature'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    pairs = []
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            if cosine_sim[i, j] > 0.4:
                pairs.append((events.iloc[i], events.iloc[j], cosine_sim[i, j]))
                
    print(f"   Found {len(pairs)} Semantic Pairs. Applying Stat Filters...")
    
    # Prepare Data
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    price_matrix = df.pivot_table(index='datetime', columns='Question', values='Price')
    price_matrix = price_matrix.resample('5min').last().ffill().dropna(axis=0, how='all')
    
    # Filter
    print("\n🔬 STATISTICAL ARBITRAGE REPORT")
    print(f"{'Pair (Short)':<40} {'Coint (p)':<10} {'Hurst':<6} {'HalfLife':<8} {'Z-Score':<8} {'Verdict'}")
    print("-" * 100)
    
    count = 0
    for p in pairs[:50]:
        q_a = p[0]['Question']
        q_b = p[1]['Question']
        
        if q_a not in price_matrix.columns or q_b not in price_matrix.columns:
            continue
            
        sa = price_matrix[q_a].dropna()
        sb = price_matrix[q_b].dropna()
        
        common = sa.index.intersection(sb.index)
        if len(common) < 30: # Need data for stats
            continue
            
        sa = sa.loc[common]
        sb = sb.loc[common]
        
        # 1. Cointegration (P-Value < 0.05 is good)
        coint_p, spread, hedge = check_cointegration(sa, sb)
        
        # 2. Hurst (< 0.5 is mean reverting)
        hurst = calculate_hurst(spread.values)
        
        # 3. Half Life (Time to revert)
        hl = calculate_halflife(spread)
        
        # 4. Z-Score (Current deviation)
        zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
        
        label = f"{q_a[:15]}... vs {q_b[:15]}..."
        
        verdict = "🔴 Noise"
        if coint_p < 0.1 and hurst < 0.5:
             if abs(zscore) > 2.0:
                 verdict = "🔥 FIRE (Entry)"
             else:
                 verdict = "🟢 Tradable (Wait)"
        elif coint_p < 0.2:
             verdict = "🟡 Weak"
             
        print(f"{label:<40} {coint_p:<10.3f} {hurst:<6.2f} {hl:<8.1f} {zscore:<8.2f} {verdict}")
        count += 1
        
    if count == 0:
        print("   No pairs with sufficient data history for StatArb.")

if __name__ == "__main__":
    main()
