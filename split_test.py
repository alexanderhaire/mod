"""
Pre-2020 vs Post-2020 Split Test
================================

Tests whether ERP weird-data timing is:
1. Robust (works pre- and post-2020)
2. Regime-specific (only works post-2020)
3. Coincidence (VIX overlay does all the work)

Compares three strategy versions:
A) Base allocation only (equal weight SPY/XLE/GLD/TLT)
B) Base + VIX overlay 
C) Full ERP with weird-data

RUN: python split_test.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# WEIRD DATA
# =============================================================================

WEIRD_DATA = {
    "netflix": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0, 
        2025: 320.0, 2026: 340.0
    },
    "cheese": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
        2025: 43.0, 2026: 43.5
    },
    "coffee": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
        2025: 6.50, 2026: 6.70
    },
}

def get_erp_signal(date):
    """Get ERP signal for a date."""
    year = date.year
    netflix_yoy = cheese_yoy = coffee_yoy = 0
    
    if year in WEIRD_DATA['netflix'] and year-1 in WEIRD_DATA['netflix']:
        netflix_yoy = (WEIRD_DATA['netflix'][year] - WEIRD_DATA['netflix'][year-1]) / WEIRD_DATA['netflix'][year-1]
    if year in WEIRD_DATA['cheese'] and year-1 in WEIRD_DATA['cheese']:
        cheese_yoy = (WEIRD_DATA['cheese'][year] - WEIRD_DATA['cheese'][year-1]) / WEIRD_DATA['cheese'][year-1]
    if year in WEIRD_DATA['coffee'] and year-1 in WEIRD_DATA['coffee']:
        coffee_yoy = (WEIRD_DATA['coffee'][year] - WEIRD_DATA['coffee'][year-1]) / WEIRD_DATA['coffee'][year-1]
    
    xle_signal = -netflix_yoy * 0.5 + cheese_yoy * 0.3 + coffee_yoy * 0.2
    return xle_signal

# =============================================================================
# THREE STRATEGY VERSIONS
# =============================================================================

def strategy_A_base_allocation(prices, vix):
    """
    Strategy A: Base allocation only
    Equal weight SPY/XLE/GLD/TLT, no signals
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(60, len(prices)), len(prices)):
        for a in assets:
            weights.iloc[i][a] = 1.0 / len(assets)
    
    return weights.shift(1).fillna(0)

def strategy_B_base_plus_vix(prices, vix):
    """
    Strategy B: Base + VIX overlay
    Equal weight base, but tilt to TLT when VIX > 25
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(60, len(prices)), len(prices)):
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        
        # VIX overlay: defensive tilt when VIX > 25
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
            if 'SPY' in w: w['SPY'] *= 0.8
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

def strategy_C_full_erp(prices, vix):
    """
    Strategy C: Full ERP with weird-data
    Base + VIX overlay + Netflix/Cheese/Coffee timing
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(60, len(prices)), len(prices)):
        date = prices.index[i]
        xle_signal = get_erp_signal(date)
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        
        # ERP weird-data signal
        if 'XLE' in w:
            if xle_signal > 0.02:
                w['XLE'], w['SPY'] = 0.35, 0.20
            elif xle_signal < -0.02:
                w['XLE'], w['GLD'] = 0.10, 0.35
        
        # VIX overlay
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def backtest(prices, weights, warmup=60):
    """Backtest and return metrics."""
    returns = prices.pct_change()
    
    if len(weights) <= warmup:
        return None
    
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm = weights[common].div(abs_sum, axis=0)
    
    port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
    port_ret = port_ret.dropna()
    
    if len(port_ret) < 20 or port_ret.std() == 0:
        return None
    
    # Metrics
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    cagr = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
    
    cum = (1 + port_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        'sharpe': sharpe,
        'cagr': cagr * 100,
        'max_dd': max_dd * 100,
        'n_days': len(port_ret)
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   PRE-2020 VS POST-2020 SPLIT TEST")
    print("   Isolating Weird-Data Contribution")
    print("=" * 80)
    
    # Fetch all data
    print("\n📊 Fetching data...")
    tickers = ['SPY', 'XLE', 'GLD', 'TLT']
    
    # Get max history
    data = yf.download(tickers + ['^VIX'], start='2010-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    if vix is not None:
        prices = prices.drop('^VIX', axis=1)
    
    print(f"   Full data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Total days: {len(prices)}")
    
    # ==========================================================================
    # DEFINE TIME WINDOWS
    # ==========================================================================
    
    windows = {
        "Pre-2020 (2014-2019)": (pd.Timestamp('2014-01-01'), pd.Timestamp('2019-12-31')),
        "2020 Only": (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')),
        "Post-2020 (2021-2026)": (pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31')),
        "Pre-Jun 2022 (IS)": (pd.Timestamp('2014-01-01'), pd.Timestamp('2022-06-07')),
        "Post-Jun 2022 (OOS)": (pd.Timestamp('2022-06-08'), pd.Timestamp('2026-12-31')),
    }
    
    strategies = {
        'A: Base Allocation': strategy_A_base_allocation,
        'B: Base + VIX': strategy_B_base_plus_vix,
        'C: Full ERP': strategy_C_full_erp,
    }
    
    # ==========================================================================
    # RUN TESTS
    # ==========================================================================
    
    results = {}
    
    for window_name, (start, end) in windows.items():
        window_prices = prices[(prices.index >= start) & (prices.index <= end)]
        window_vix = vix[(vix.index >= start) & (vix.index <= end)] if vix is not None else None
        
        if len(window_prices) < 100:
            continue
        
        results[window_name] = {}
        
        for strat_name, strat_func in strategies.items():
            weights = strat_func(window_prices, window_vix)
            metrics = backtest(window_prices, weights)
            results[window_name][strat_name] = metrics
    
    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   RESULTS BY TIME WINDOW")
    print("=" * 80)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
            
        print(f"\n   {window_name}")
        print("   " + "-" * 65)
        print(f"   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'Days':>8}")
        print("   " + "-" * 65)
        
        for strat_name in strategies.keys():
            m = results[window_name].get(strat_name)
            if m:
                print(f"   {strat_name:<25} {m['sharpe']:>10.2f} {m['cagr']:>9.1f}% {m['max_dd']:>9.1f}% {m['n_days']:>8}")
            else:
                print(f"   {strat_name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
    
    # ==========================================================================
    # INCREMENTAL CONTRIBUTION OF WEIRD-DATA (C - B)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   INCREMENTAL CONTRIBUTION OF WEIRD-DATA (C - B)")
    print("=" * 80)
    print("\n   This isolates what the Netflix/Cheese/Coffee signal adds")
    print("   beyond the VIX defensive overlay.\n")
    
    print(f"   {'Window':<30} {'Sharpe Δ':>12} {'CAGR Δ':>12} {'MaxDD Δ':>12}")
    print("   " + "-" * 70)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        b = results[window_name].get('B: Base + VIX')
        c = results[window_name].get('C: Full ERP')
        
        if b and c:
            d_sharpe = c['sharpe'] - b['sharpe']
            d_cagr = c['cagr'] - b['cagr']
            d_dd = c['max_dd'] - b['max_dd']
            
            # Color coding
            sharpe_sign = "+" if d_sharpe >= 0 else ""
            cagr_sign = "+" if d_cagr >= 0 else ""
            
            print(f"   {window_name:<30} {sharpe_sign}{d_sharpe:>11.2f} {cagr_sign}{d_cagr:>10.1f}% {d_dd:>11.1f}%")
        else:
            print(f"   {window_name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   VERDICT INTERPRETATION")
    print("=" * 80)
    
    # Calculate key differentials
    pre_2020 = results.get("Pre-2020 (2014-2019)", {})
    post_2020 = results.get("Post-2020 (2021-2026)", {})
    
    pre_b = pre_2020.get('B: Base + VIX', {})
    pre_c = pre_2020.get('C: Full ERP', {})
    post_b = post_2020.get('B: Base + VIX', {})
    post_c = post_2020.get('C: Full ERP', {})
    
    if pre_b and pre_c and post_b and post_c:
        pre_delta = pre_c['sharpe'] - pre_b['sharpe']
        post_delta = post_c['sharpe'] - post_b['sharpe']
        
        print(f"""
   Pre-2020 weird-data increment:  {pre_delta:+.2f} Sharpe
   Post-2020 weird-data increment: {post_delta:+.2f} Sharpe
   
   Analysis:
   ─────────────────────────────────────────────────────────────""")
        
        if pre_delta > 0.1 and post_delta > 0.1:
            print("""
   ✅ ROBUST: Weird-data signal adds value in BOTH regimes.
   The Netflix/Cheese/Coffee timing appears to be a genuine,
   transportable signal that works across different macro environments.""")
        elif pre_delta < 0.1 and post_delta > 0.2:
            print("""
   ⚠️  REGIME-SPECIFIC: Weird-data only helps post-2020.
   The signal works during the inflation regime but not before.
   Use as a CONDITIONAL regime tool, not a universal alpha claim.""")
        elif abs(pre_delta) < 0.1 and abs(post_delta) < 0.1:
            print("""
   ❌ NO INCREMENTAL VALUE: VIX overlay does all the work.
   The weird-data adds essentially nothing beyond Base + VIX.
   The signal is likely narrative-fit coincidence.""")
        else:
            print(f"""
   📊 MIXED: Requires careful interpretation.
   Pre-2020 delta: {pre_delta:+.2f}
   Post-2020 delta: {post_delta:+.2f}
   The weird-data effect varies significantly by regime.""")
    else:
        print("\n   Could not compute all windows for verdict.")
    
    print("\n" + "=" * 80)
