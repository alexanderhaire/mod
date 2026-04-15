"""
IC-Weighted Portfolio Test
==========================

Testing if weighting assets by their ML Predictive Power (IC) improves performance.
Theory: Bet bigger on assets the model understands better.

Weights:
XLB: 0.357
XLI: 0.308
XLE: 0.230
JNK: 0.228
USMV: 0.210
MTUM: 0.193
FXA: 0.214
GLD: 0.127
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def run_test():
    print("=" * 80)
    print("   PORTFOLIO OPTIMIZATION: IC-WEIGHTING")
    print("=" * 80)
    
    # Hardcoded ICs from Discovery
    ics = {
        'XLB': 0.357,
        'XLI': 0.308,
        'XLE': 0.230,
        'JNK': 0.228,
        'USMV': 0.210,
        'MTUM': 0.193,
        'FXA': 0.214,
        'GLD': 0.127
    }
    
    tickers = list(ics.keys())
    
    # Calculate Weights
    total_ic = sum(ics.values())
    weights = {k: v / total_ic for k, v in ics.items()}
    
    print("\n   Proposed Weights (based on Predictive Power):")
    for t in tickers:
        print(f"   {t}: {weights[t]:.1%}")
        
    # Validation
    end = datetime.now()
    start = end - timedelta(days=5*365)
    data = yf.download(tickers, start=start, end=end, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
        
    prices = prices.ffill().dropna()
    returns = prices.pct_change()
    
    # 1. Equal Weight
    ew_ret = returns.mean(axis=1)
    ew_sharpe = ew_ret.mean() / ew_ret.std() * np.sqrt(252)
    ew_cagr = (1+ew_ret).cumprod().iloc[-1]**(252/len(ew_ret)) - 1
    
    # 2. IC Weighted
    w_vec = np.array([weights[c] for c in returns.columns])
    ic_ret = (returns * w_vec).sum(axis=1)
    ic_sharpe = ic_ret.mean() / ic_ret.std() * np.sqrt(252)
    ic_cagr = (1+ic_ret).cumprod().iloc[-1]**(252/len(ic_ret)) - 1
    
    print("\n   === RESULTS (Passive Hold) ===")
    print(f"   Equal Weight Sharpe: {ew_sharpe:.2f} (CAGR {ew_cagr:.1%})")
    print(f"   IC Weighted Sharpe:  {ic_sharpe:.2f} (CAGR {ic_cagr:.1%})")
    
    diff = ic_sharpe - ew_sharpe
    print(f"   Improvement: {diff:.2f}")
    
    if diff > 0.05:
        print("   ✅ RECOMMENDATION: Use IC-Weighted Allocation.")
    else:
        print("   ❌ RECOMMENDATION: Stick to Equal Weight (Robustness).")

if __name__ == "__main__":
    run_test()
