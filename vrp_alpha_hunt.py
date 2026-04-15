"""
VRP Alpha Hunt (Volatility Arbitrage)
=====================================

Exploiting the Volatility Risk Premium (VRP).
Theory: Implied Volatility (VIX) > Realized Volatility (RV) on average. 
This spread is the "Insurance Premium".

Strategy:
- Calculate VRP = VIX - Realized_Vol(21d).
- If VRP > 5: Long Short-Vol (SVXY). (Harvest Premium).
- If VRP < 0: Cash. (High Risk of Spike).

RUN: python vrp_alpha_hunt.py
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

def fetch_vol_data():
    print("🌋 Fetching Volatility Data (SPY, VIX, SVXY)...")
    tickers = ['SPY', '^VIX', 'SVXY']
    data = yf.download(tickers, start='2011-01-01', progress=False) # SVXY inception ~2011
    
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

def backtest_vrp(prices):
    if 'SPY' not in prices.columns or '^VIX' not in prices.columns or 'SVXY' not in prices.columns:
        print("   ❌ Missing Data")
        return
        
    spy = prices['SPY']
    vix = prices['^VIX']
    svxy = prices['SVXY']
    
    # 1. Calculate Realized Volatility (RV)
    # Annualized rolling standard deviation of SPY returns
    spy_ret = spy.pct_change()
    rv = spy_ret.rolling(21).std() * np.sqrt(252) * 100 # In percentage terms
    
    # 2. Calculate VRP
    # VIX is already annualized percentage
    vrp = vix - rv
    
    print(f"   Avg VRP Spread: {vrp.mean():.2f} pts (Normally Positive)")
    
    # 3. Strategy
    # Valid Index
    valid_idx = vrp.dropna().index
    
    # Signal
    # If VRP > 5: We sell insurance (Long SVXY)
    # Why? Market is pricing in more fear than is realized.
    signal = (vrp > 5.0).astype(int)
    
    # Returns
    # Trade SVXY
    # Shift signal to trade next day
    svxy_ret = svxy.pct_change()
    strat_ret = signal.shift(1).loc[valid_idx] * svxy_ret.loc[valid_idx]
    
    strat_ret = strat_ret.dropna()
    
    # Benchmark 1: Buy & Hold SPY
    spy_bh = spy_ret.loc[valid_idx]
    
    # Benchmark 2: Buy & Hold SVXY (Short Vol always)
    svxy_bh = svxy_ret.loc[valid_idx]
    
    # Metrics
    def get_stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        return ann, sharpe, dd
        
    s_ann, s_sharpe, s_tot = get_stats(strat_ret)
    b_ann, b_sharpe, b_tot = get_stats(spy_bh)
    v_ann, v_sharpe, v_tot = get_stats(svxy_bh)
    
    print("\n🌋 VRP STRATEGY RESULTS:")
    print(f"   Avg VRP: {vrp.mean():.2f}")
    print("-" * 50)
    print(f"   Buy & Hold SPY:   Ann {b_ann:.1%} | Sharpe {b_sharpe:.2f} | Total {b_tot:.0%}")
    print(f"   Hold Short Vol (SVXY): Ann {v_ann:.1%} | Sharpe {v_sharpe:.2f} | Total {v_tot:.0%}")
    print(f"   VRP Harvesting Strat:  Ann {s_ann:.1%} | Sharpe {s_sharpe:.2f} | Total {s_tot:.0%}")
    
    if s_sharpe > b_sharpe + 0.2:
        print("\n✅ EDGE FOUND: Selling Fear pays well.")
    else:
        print("\n❌ NO EDGE: VRP is efficiently priced or SVXY drag kills it.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🌋 VRP ALPHA HUNT (VOLATILITY ARBITRAGE)")
    print("="*60)
    
    prices = fetch_vol_data()
    backtest_vrp(prices)
    
    print("\n" + "=" * 60)
