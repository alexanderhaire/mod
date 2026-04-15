"""
Final Stress Test
=================

Extensive Robustness Checks for the "Winning Strategy".

1.  **Rolling Stability**: does the edge decay?
2.  **Parameter Sensitivity**: was 20% Vol Target just lucky?
3.  **Crisis Anatomy**: how did it handle Covid (2020) and Inflation (2022)?

Universe: XLB, XLI, XLE, JNK, GLD
Model: GBM + Macro (Rates/Credit/Dollar)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD']
MACRO = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data() -> pd.DataFrame:
    tickers = list(set(ASSETS + MACRO))
    data = yf.download(tickers, start="2018-01-01", progress=False) # Enough for 2019-2024 backtest
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.ffill().dropna()

def get_strategy_returns(prices, vol_target=0.20):
    # Quick reconstruction of the engine for efficient looping
    df = pd.DataFrame(index=prices.index)
    
    # Macro Features
    if '^TNX' in prices.columns:
        df['rate_change'] = prices['^TNX'].diff(20)
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        df['credit_spread'] = prices['JNK'] / prices['IEF']
        
    asset_map = {}
    for t in ASSETS:
        if t not in prices.columns: continue
        r = prices[t].pct_change()
        df[f'{t}_mom'] = r.rolling(20).mean()
        asset_map[t] = [f'{t}_mom', 'rate_change', 'credit_spread']
    
    # Target
    targets = prices.pct_change().shift(-1)
    
    # Train/Test Split (Fixed for consistency)
    features = df.dropna()
    common = features.index.intersection(targets.index)
    
    # Walk Forward 2020-2024
    # Train on 2018-2019 using Walk Forward?
    # For speed, let's just train on 2018-2022 and test on broad range to see fit?
    # No, strict OOS. 
    # Let's simple split: Train 2018-2019. Test 2020-Present (Includes Covid + 2022)
    
    split_date = "2020-01-01"
    train_end = features.index.searchsorted(pd.Timestamp(split_date))
    
    # Train Models
    models = {}
    for t in ASSETS:
        cols = [c for c in asset_map[t] if c in features.columns]
        X = features.iloc[:train_end][cols]
        y = targets.iloc[:train_end][t]
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X, y)
        models[t] = m
    
    # Predict OOS
    oos_idx = features.index[train_end:]
    signals = pd.DataFrame(index=oos_idx, columns=ASSETS)
    
    for i in range(len(oos_idx)):
        idx = oos_idx[i]
        curr_idx = features.index.get_loc(idx)
        for t in ASSETS:
            cols = [c for c in asset_map[t] if c in features.columns]
            row = features.iloc[curr_idx:curr_idx+1][cols]
            signals.iloc[i][t] = models[t].predict(row)[0]
            
    # Proportional Allocation
    weights = signals.copy()
    weights[weights < 0] = 0
    row_sums = weights.sum(axis=1)
    # Avoid zero division
    row_sums[row_sums == 0] = 1.0
    weights = weights.div(row_sums, axis=0).fillna(0)
    
    # Vol Targeting
    aligned_rets = targets.loc[oos_idx]
    strat_ret_unscaled = (weights * aligned_rets).sum(axis=1)
    
    univ_ret = aligned_rets.mean(axis=1)
    rolling_vol = univ_ret.rolling(20).std() * np.sqrt(252)
    lagged_vol = rolling_vol.shift(1).fillna(0.15)
    
    scale = (vol_target / lagged_vol).clip(0, 2.0)
    strat_ret = strat_ret_unscaled * scale
    
    return strat_ret.dropna(), univ_ret.dropna()

def run_stress_test():
    print("=" * 80)
    print("   FINAL STRESS TEST MATRIX")
    print("=" * 80)
    
    prices = fetch_data()
    
    # 1. Baseline
    print("   Running Baseline (Vol=20%)...")
    strat, bench = get_strategy_returns(prices, vol_target=0.20)
    
    # 2. Rolling Stability
    print("\n   --- TEST 1: ROLLING 12M SHARPE ---")
    roll_sharpe = strat.rolling(252).mean() / strat.rolling(252).std() * np.sqrt(252)
    min_s = roll_sharpe.min()
    max_s = roll_sharpe.max()
    curr_s = roll_sharpe.iloc[-1]
    print(f"   Min Sharpe: {min_s:.2f} | Max Sharpe: {max_s:.2f} | Current: {curr_s:.2f}")
    if min_s < 0:
        print("   ⚠️ WARNING: Strategy had period of negative risk-adjusted returns.")
    else:
        print("   ✅ STABLE: Sharpe Ratio never turned negative in rolling windows.")
        
    # 3. Crisis Anatomy
    print("\n   --- TEST 2: CRISIS PERFORMANCE ---")
    # Covid: Feb 20 - Mar 23 2020
    covid = strat.loc['2020-02-20':'2020-03-23']
    covid_bench = bench.loc['2020-02-20':'2020-03-23']
    print(f"   COVID Crash (Feb-Mar '20): Strat {covid.sum():.1%} vs Bench {covid_bench.sum():.1%}")
    
    # Inflation Bear: Jan - June 2022
    bear = strat.loc['2022-01-01':'2022-06-30']
    bear_bench = bench.loc['2022-01-01':'2022-06-30']
    print(f"   Inflation Bear ('22 H1):   Strat {bear.sum():.1%} vs Bench {bear_bench.sum():.1%}")
    
    # 4. Parameter Sensitivity
    print("\n   --- TEST 3: PARAMETER HEATMAP (Vol Target) ---")
    print(f"   {'Vol Target':<12} {'Sharpe':<8} {'CAGR':<8} {'MaxDD':<8}")
    print("   " + "-" * 40)
    
    for v in [0.10, 0.15, 0.20, 0.25, 0.30]:
        s, _ = get_strategy_returns(prices, vol_target=v)
        sharpe = s.mean() / s.std() * np.sqrt(252)
        cagr = s.mean() * 252
        dd = (1+s).cumprod() / (1+s).cumprod().cummax() - 1
        print(f"   {v:<12.0%} {sharpe:<8.2f} {cagr:<8.1%} {dd.min():<8.1%}")
        
    print("\n   ✅ STRESS TESTS COMPLETE.")

if __name__ == "__main__":
    run_stress_test()
