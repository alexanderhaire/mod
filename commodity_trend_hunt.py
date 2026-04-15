"""
Commodity Trend Alpha Hunt
==========================

Testing Trend Following on Hard Assets.
Hypothesis: Commodities exhibit strong serial correlation (trends) due to supply/demand lags.
Challenge: High Volatility.
Solution: Trend Filter (SMA 200) + Volatility Targeting (15%).

Assets:
- GLD (Gold)
- USO (Oil)
- DBA (Agriculture)
- DBC (Broad Commodity Index)

RUN: python commodity_trend_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA
# =============================================================================

def fetch_data():
    print("🛢️ Fetching Commodity Data...")
    tickers = ['GLD', 'USO', 'DBA', 'DBC', 'SPY']
    data = yf.download(tickers, start='2007-01-01', progress=False) # Long history
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    return prices

# =============================================================================
# 2. STRATEGY ENGINE
# =============================================================================

def run_trend_strategy(prices, asset, target_vol=0.15):
    if asset not in prices.columns: return None
    
    p = prices[asset]
    ret = p.pct_change()
    
    # 1. Trend Signal (SMA 200)
    sma = p.rolling(200).mean()
    signal = pd.Series(0, index=p.index)
    signal[p > sma] = 1 # Long
    signal[p < sma] = 0 # Cash (or -1 for Short, but Commodities are hard to short via ETF cost)
    # Let's stick to Long/Flat for ETF verify
    
    # 2. Volatility Targeting
    # Realized Vol (21 days)
    # We want position size * RealizedVol = TargetVol
    # Size = Target / Realized
    
    hist_vol = ret.rolling(21).std() * np.sqrt(252)
    
    # Avoid div by zero
    hist_vol = hist_vol.replace(0, 0.01) 
    
    size = target_vol / hist_vol
    size = size.clip(0, 2.0) # Cap leverage at 2x
    
    # Combined Weight
    weight = signal * size
    
    # Lag
    weight = weight.shift(1).fillna(0)
    
    # Returns
    strat_ret = weight * ret
    
    # Cost
    # Turnover approx: delta weight
    turnover = weight.diff().abs().fillna(0)
    cost = turnover * 0.0010 # 10bps
    
    net_ret = strat_ret - cost
    
    return net_ret.fillna(0)

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_commodities(prices):
    print("\n🛢️ COMMODITY TREND RESULTS (Long/Flat + Vol Target):")
    print("-" * 75)
    
    spy_ret = prices['SPY'].pct_change().dropna()
    
    assets = [c for c in prices.columns if c != 'SPY']
    
    for asset in assets:
        # Run Strat
        strat = run_trend_strategy(prices, asset)
        bh = prices[asset].pct_change().dropna()
        
        # Metrics
        def get_metrics(r):
            ann = r.mean() * 252
            vol = r.std() * np.sqrt(252)
            sharpe = ann / vol if vol > 0 else 0
            dd = (1+r).cumprod().iloc[-1] - 1 # Total
            return ann, vol, sharpe
            
        s_ann, s_vol, s_sharpe = get_metrics(strat)
        b_ann, b_vol, b_sharpe = get_metrics(bh)
        
        # Correlation to SPY
        corr = strat.corr(spy_ret)
        
        print(f"   {asset:<5} | Trend Sharpe {s_sharpe:.2f} (vs BH {b_sharpe:.2f}) | Vol {s_vol:.1%} | Corr SPY {corr:.2f}")
        
    print("-" * 75)
    
    # Combined Portfolio of Commodities?
    # Equal Weight the Trend Strategies
    combined = pd.DataFrame()
    for asset in assets:
        s = run_trend_strategy(prices, asset)
        combined[asset] = s
        
    port = combined.mean(axis=1)
    
    p_ann = port.mean() * 252
    p_vol = port.std() * np.sqrt(252)
    p_sharpe = p_ann / p_vol if p_vol > 0 else 0
    p_corr = port.corr(spy_ret)
    
    print(f"   COMBO | Trend Sharpe {p_sharpe:.2f} | Ann {p_ann:.1%} | Vol {p_vol:.1%} | Corr SPY {p_corr:.2f}")

    if p_sharpe > 0.6 and p_corr < 0.3:
        print("\n✅ ALPHA FOUND: Commodities provide uncorrelated returns.")
    else:
        print("\n❌ FAILED: Commodities are a drag or too correlated.")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_commodities(prices)
