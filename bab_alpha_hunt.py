"""
Betting Against Beta (BAB) Alpha Hunt
=====================================

Testing the "Low Volatility Anomaly".
Academic Theory: High Beta assets are overpriced "Lottery Tickets".
Low Beta assets offer superior risk-adjusted returns (higher Sharpe).

Data (Invesco ETFs, Inception ~2011):
- SPLV: S&P 500 Low Volatility
- SPHB: S&P 500 High Beta
- SPY: S&P 500 Benchmark

RUN: python bab_alpha_hunt.py
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

def fetch_factor_data():
    print("📉 Fetching Factor Data (SPLV, SPHB, SPY)...")
    tickers = ['SPLV', 'SPHB', 'SPY']
    data = yf.download(tickers, start='2011-05-01', progress=False) # Inception May 2011
    
    # Robust extraction
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
# 2. ANALYSIS
# =============================================================================

def test_bab(prices):
    if 'SPLV' not in prices.columns or 'SPHB' not in prices.columns:
        print("   ❌ Missing Data")
        return
        
    splv = prices['SPLV']
    sphb = prices['SPHB']
    spy = prices['SPY']
    
    # Returns
    splv_ret = splv.pct_change().dropna()
    sphb_ret = sphb.pct_change().dropna()
    spy_ret = spy.pct_change().dropna()
    
    # Align
    common_idx = splv_ret.index.intersection(sphb_ret.index).intersection(spy_ret.index)
    splv_ret = splv_ret.loc[common_idx]
    sphb_ret = sphb_ret.loc[common_idx]
    spy_ret = spy_ret.loc[common_idx]
    
    # Strategy: BAB Factor (Long Low Vol / Short High Beta)
    # Rebalanced daily (conceptual)
    # We want to catch the spread.
    bab_factor = (splv_ret - sphb_ret) / 2
    
    # Metrics
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1 # Simple total return proxy
        max_dd = ((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()
        print(f"   {name:<20} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | MaxDD {max_dd:.1%}")
        return sharpe

    print("\n📉 BETTING AGAINST BETA RESULTS:")
    print("-" * 75)
    
    s_spy = get_stats(spy_ret, "SPY (Benchmark)")
    s_splv = get_stats(splv_ret, "SPLV (Low Vol)")
    s_sphb = get_stats(sphb_ret, "SPHB (High Beta)")
    s_bab = get_stats(bab_factor, "BAB (L/S Factor)")
    
    print("-" * 75)
    
    # Conclusion
    if s_splv > s_spy:
        print("✅ ANOMALY CONFIRMED: Low Vol has higher Sharpe than Market.")
    else:
        print("❌ ANOMALY BROKEN: Low Vol underperformed Market.")
        
    if s_sphb < s_spy:
        print("✅ THEORY CONFIRMED: High Beta (Lottery) has lower Sharpe.")
    
    corr = splv_ret.corr(spy_ret)
    print(f"\n   Stats: Low Vol Correlation to SPY: {corr:.2f}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("📉 BETTING AGAINST BETA (FACTOR ALPHA)")
    print("="*60)
    
    prices = fetch_factor_data()
    test_bab(prices)
    
    print("\n" + "=" * 60)
