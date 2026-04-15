"""
Factor Rotation Alpha Hunt
==========================

Testing "Smart Equity": Switching between Growth (VUG) and Value (VTV).
Hypothesis: Factor leadership persists. One style usually dominates for months/years.
Strategy: Relative Momentum (6-Month Lookback).
Rule:
- If Return(VUG, 6m) > Return(VTV, 6m): Hold VUG.
- Else: Hold VTV.

Benchmark: SPY (Buy & Hold).

RUN: python factor_rotation_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA
# =============================================================================

def fetch_data():
    print("🧬 Fetching Factor Data (VUG, VTV, SPY)...")
    tickers = ['VUG', 'VTV', 'SPY']
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
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

def run_factor_rotation(prices):
    if 'VUG' not in prices.columns or 'VTV' not in prices.columns: return None
    
    vug = prices['VUG']
    vtv = prices['VTV']
    
    # Returns
    ret_vug = vug.pct_change()
    ret_vtv = vtv.pct_change()
    
    # Momentum (126 days / 6 months)
    mom_vug = vug.pct_change(126)
    mom_vtv = vtv.pct_change(126)
    
    # Signal
    signal = pd.Series(0, index=prices.index)
    signal[mom_vug > mom_vtv] = 1.0 # Growth Mode
    
    # Allocation
    # If 1 -> VUG, Else -> VTV (0)
    # Lag 1 day
    w_vug = signal.shift(1).fillna(0.5)
    w_vtv = 1.0 - w_vug
    
    # Strategy Return
    strat_ret = w_vug * ret_vug + w_vtv * ret_vtv
    
    return strat_ret.fillna(0), w_vug

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_factors(prices):
    print("\n🧬 FACTOR ROTATION RESULTS (6M Momentum):")
    print("-" * 75)
    
    spy_ret = prices['SPY'].pct_change().fillna(0)
    rot_ret, w_vug = run_factor_rotation(prices)
    
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        
        # Max Drawdown
        cum = (1+r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        mdd = drawdown.min()
        
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | MDD {mdd:>6.1%}")
        return sharpe

    s_spy = get_stats(spy_ret, "SPY (Benchmark)")
    s_vug = get_stats(prices['VUG'].pct_change(), "VUG (Growth Hold)")
    s_vtv = get_stats(prices['VTV'].pct_change(), "VTV (Value Hold)")
    s_rot = get_stats(rot_ret, "Factor Rotation (Smart)")
    
    print("-" * 75)
    
    # 2022 Check
    y22_spy = spy_ret.loc['2022-01-01':'2022-12-31'].sum()
    y22_rot = rot_ret.loc['2022-01-01':'2022-12-31'].sum()
    
    print(f"   2022 Return: SPY {y22_spy:.1%} vs Rotation {y22_rot:.1%}")
    
    if s_rot > s_spy + 0.05:
         print("✅ ALPHA FOUND: Rotation beats the Average.")
         print("   Recommendation: Replace SPY with Factor Rotation engine.")
    else:
         print("❌ FAILED: Rotation churn destroys alpha. Stick to SPY.")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_factors(prices)
