"""
Volatility Target Sweep
=======================

Finding the optimal Volatility Target for the Cyclical/Macro Strategy.
Testing: 10%, 12.5%, 15%, 17.5%, 20%, 25%

Hypothesis: Cyclicals are volatile. 15% might be too low (choking returns).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Assets (equal weight)
ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']

def run_test():
    print("=" * 80)
    print("   VOLATILITY TARGET SWEEP")
    print("=" * 80)
    
    end = datetime.now()
    start = end - timedelta(days=5*365)
    data = yf.download(ASSETS, start=start, end=end, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.dropna(how='all').ffill().dropna()
    
    returns = prices.pct_change().dropna()
    
    # Simulate Equal Weight Portfolio (Daily Rebalance approx)
    # Port Return = Mean(Asset Returns)
    port_ret = returns.mean(axis=1)
    
    targets = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
    
    print(f"\n   {'Target Vol':<12} {'Realized Vol':<12} {'CAGR':<10} {'Sharpe':<8} {'MaxDD':<8}")
    print("   " + "-" * 60)
    
    best_sharpe = 0
    best_target = 0.15
    
    for tv in targets:
        # Vol Targeting Loop
        weights = []
        lookback = 20
        
        # Rolling Volatility (20 days)
        rolling_std = port_ret.rolling(lookback).std() * np.sqrt(252)
        
        # Scaling Factor (lagged 1 day to be realistic)
        # Scale = Target / Realized
        scale = (tv / rolling_std).shift(1).clip(0, 2.0) # Cap leverage at 2x
        
        # Strategy Returns
        strat_ret = port_ret * scale
        strat_ret = strat_ret.dropna()
        
        # Metrics
        ann_vol = strat_ret.std() * np.sqrt(252)
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
        cagr = (1 + strat_ret).cumprod().iloc[-1]**(252/len(strat_ret)) - 1
        max_dd = (1 + strat_ret).cumprod() / (1 + strat_ret).cumprod().cummax() - 1
        max_dd = max_dd.min()
        
        print(f"   {tv:<12.1%} {ann_vol:<12.1%} {cagr:<10.1%} {sharpe:<8.2f} {max_dd:<8.1%}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_target = tv
            
    print("\n" + "=" * 80)
    print(f"   OPTIMAL TARGET VOLATILITY: {best_target:.1%}")
    print("=" * 80)
    
    if best_target > 0.15:
        print("   ✅ RECOMMENDATION: Increase Risk Target (Aggressive).")
    elif best_target < 0.15:
        print("   ✅ RECOMMENDATION: Decrease Risk Target (Conservative).")
    else:
        print("   ✅ RECOMMENDATION: Stick to 15% (Balanced).")

if __name__ == "__main__":
    run_test()
