"""
Lead-Lag Alpha Hunt
===================

Testing Inter-market "Crystal Ball" effects.
Hypothesis: Smart/Upstream assets (Leader) move *before* Broad/Downstream assets (Laggard).

Pairs:
1. Semis (SOXX) -> Tech (QQQ)
2. High Yield (HYG) -> Stocks (SPY)
3. Copper (COPX) -> Emerging Markets (EEM)
4. Lumber (WOOD) -> Homebuilders (XHB)

Test: Correlation of Leader(T) with Laggard(T+1).
Strategy: If Leader Return > 1.0 StdDev, Long Laggard at Close. Correlation is not enough, need Signal.

RUN: python lead_lag_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA
# =============================================================================

def fetch_pairs_data():
    print("🔮 Fetching Lead-Lag Pairs...")
    pairs = {
        'Semis_Tech': ['SOXX', 'QQQ'],
        'Credit_Equity': ['HYG', 'SPY'],
        'Copper_EM': ['COPX', 'EEM'],
        'Lumber_Housing': ['WOOD', 'XHB'] # WOOD is Global Timber & Forestry
    }
    
    all_tickers = [t for pair in pairs.values() for t in pair]
    data = yf.download(all_tickers, start='2018-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna()
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices, pairs

# =============================================================================
# 2. ANALYSIS
# =============================================================================

def analyze_pair(prices, name, leader, laggard):
    # Returns
    r_lead = prices[leader].pct_change()
    r_lag = prices[laggard].pct_change()
    
    # Lagged Correlation: Leader(T) vs Laggard(T+1)
    df = pd.DataFrame({'Lead_T': r_lead, 'Lag_T+1': r_lag.shift(-1)}).dropna()
    
    corr = df.corr().iloc[0, 1]
    
    # Strategy Backtest
    # Logic: If Leader moves > 1 StdDev Up, Buy Laggard at Close(T) (or Open T+1), Hold 1 Day.
    # Assumption: We can trade Laggard at Close T (Leader move is known).
    
    vol = r_lead.rolling(20).std().shift(1) # Ex-ante vol
    
    # Signal
    z_score = (r_lead / vol).fillna(0)
    
    signal = pd.Series(0, index=r_lead.index)
    signal[z_score > 1.0] = 1.0  # Buy
    signal[z_score < -1.0] = -1.0 # Short
    
    # Returns (Trade Laggard T+1)
    # We enter at Close T. Return is Laggard(T+1).
    # Strat Ret = Signal(T) * Lag_Ret(T+1)
    # Since r_lag is (P_t / P_t-1)-1, r_lag.shift(-1) is Return at T+1.
    strat_ret = signal * r_lag.shift(-1)
    strat_ret = strat_ret.dropna()
    
    # Metrics
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    ann_ret = strat_ret.mean() * 252
    
    print(f"\n🔗 PAIR: {name} ({leader} -> {laggard})")
    print(f"   Lagged Correlation: {corr:.3f}")
    print(f"   Strategy Sharpe: {sharpe:.2f}")
    
    if sharpe > 0.6 and abs(corr) > 0.05:
         print("   ✅ POTENTIAL EDGE")
         return {'Name': name, 'Sharpe': sharpe, 'Corr': corr}
    else:
         print("   ❌ NO EDGE (Market is efficient)")
         return None

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🔮 LEAD-LAG ALPHA HUNT")
    print("="*60)
    
    prices, pairs = fetch_pairs_data()
    
    results = []
    
    for name, p in pairs.items():
        leader, laggard = p
        if leader in prices.columns and laggard in prices.columns:
            res = analyze_pair(prices, name, leader, laggard)
            if res: results.append(res)
            
    if not results:
        print("\n🏁 FINAL VERDICT: Efficient Market Hypothesis wins. No crystal balls found.")
    
    print("\n" + "=" * 60)
