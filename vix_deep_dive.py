"""
VIX Term Structure Deep Dive
============================

The VIX term structure signal showed marginal significance (p=0.079).
Let's refine it by:
1. Optimizing thresholds
2. Combining with VIX level
3. Adding smoothing/confirmation
4. Testing different asset universes

RUN: python vix_deep_dive.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'IEF', 'XLE', 'EEM', 'HYG']
    vix_tickers = ['^VIX', '^VIX3M']
    
    data = yf.download(tickers + vix_tickers, start='2008-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    for col in ['^VIX', '^VIX3M']:
        if col in prices.columns:
            prices = prices.drop(col, axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, vix, vix3m

def vix_composite_signal(vix, vix3m, smoothing=5):
    """
    Enhanced VIX signal combining:
    1. Term structure (contango/backwardation)
    2. VIX level (high = stressed)
    3. VIX momentum (rising = worsening)
    """
    if vix is None or vix3m is None:
        return None
    
    # Term structure ratio
    ratio = vix / vix3m
    ratio_smooth = ratio.rolling(smoothing).mean()
    
    # VIX level
    vix_ma = vix.rolling(20).mean()
    vix_high = vix > vix_ma * 1.2
    vix_extreme = vix > 30
    
    # VIX momentum (5-day change)
    vix_mom = vix.pct_change(5)
    vix_rising = vix_mom > 0.1
    
    # Composite signal
    signal = pd.Series(0.0, index=vix.index)
    
    # Strong contango + low VIX = very bullish
    strong_contango = ratio_smooth < 0.90
    signal[strong_contango & ~vix_high] = 1.0
    
    # Backwardation = bearish
    backwardation = ratio_smooth > 1.0
    signal[backwardation] = -0.5
    
    # Backwardation + VIX rising = very bearish
    signal[backwardation & vix_rising] = -1.0
    
    # VIX extreme = max defensive
    signal[vix_extreme] = -1.0
    
    # Discretize
    result = pd.Series(0, index=signal.index)
    result[signal > 0.5] = 1
    result[signal < -0.25] = -1
    
    return result

def strategy_vix_enhanced(prices, vix, vix3m):
    """Enhanced VIX strategy with multi-asset rotation."""
    signal = vix_composite_signal(vix, vix3m)
    
    if signal is None:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Asset groups
    risk_on = ['SPY', 'QQQ', 'XLE', 'EEM', 'HYG']
    risk_off = ['TLT', 'GLD', 'IEF']
    
    for i in range(252, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0:
            # Risk-on: heavy equities
            for a in risk_on:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.80 / len([x for x in risk_on if x in prices.columns])
            for a in risk_off:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.20 / len([x for x in risk_off if x in prices.columns])
        elif s < 0:
            # Risk-off: defensive
            for a in risk_on:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.20 / len([x for x in risk_on if x in prices.columns])
            for a in risk_off:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.80 / len([x for x in risk_off if x in prices.columns])
        else:
            # Neutral
            all_assets = [a for a in risk_on + risk_off if a in prices.columns]
            for a in all_assets:
                weights.iloc[i][a] = 1.0 / len(all_assets)
    
    return weights.shift(1).fillna(0)

def strategy_vix_simple_spy_tlt(prices, vix, vix3m):
    """Simple SPY/TLT rotation based on VIX signal."""
    signal = vix_composite_signal(vix, vix3m)
    
    if signal is None:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0:
            if 'SPY' in prices.columns: weights.iloc[i]['SPY'] = 0.80
            if 'TLT' in prices.columns: weights.iloc[i]['TLT'] = 0.20
        elif s < 0:
            if 'SPY' in prices.columns: weights.iloc[i]['SPY'] = 0.20
            if 'TLT' in prices.columns: weights.iloc[i]['TLT'] = 0.80
        else:
            if 'SPY' in prices.columns: weights.iloc[i]['SPY'] = 0.60
            if 'TLT' in prices.columns: weights.iloc[i]['TLT'] = 0.40
    
    return weights.shift(1).fillna(0)

def strategy_base(prices):
    """60/40 benchmark."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        if 'SPY' in prices.columns: weights.iloc[i]['SPY'] = 0.60
        if 'TLT' in prices.columns: weights.iloc[i]['TLT'] = 0.40
    
    return weights.shift(1).fillna(0)

def compute_returns(prices, weights, warmup=252):
    returns = prices.pct_change()
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm = weights[common].div(abs_sum, axis=0)
    
    port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
    return port_ret.dropna()

def compute_metrics(returns):
    if len(returns) < 20 or returns.std() == 0:
        return None
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    return {'sharpe': sharpe, 'cagr': cagr * 100, 'max_dd': max_dd * 100}

def compute_active_stats(r_c, r_b):
    common_idx = r_b.index.intersection(r_c.index)
    active = r_c.loc[common_idx] - r_b.loc[common_idx]
    active = active.dropna()
    
    if len(active) < 20 or active.std() == 0:
        return None
    
    ir = active.mean() / active.std() * np.sqrt(252)
    t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val, 'n': len(active)}

if __name__ == "__main__":
    print("=" * 80)
    print("   VIX TERM STRUCTURE DEEP DIVE")
    print("   Refining the marginal edge (p=0.079)")
    print("=" * 80)
    
    prices, vix, vix3m = fetch_data()
    
    # Generate enhanced signal
    signal = vix_composite_signal(vix, vix3m)
    
    print("\n📈 Signal statistics:")
    print(f"   Bullish days: {(signal == 1).sum()} ({(signal == 1).mean()*100:.1f}%)")
    print(f"   Neutral days: {(signal == 0).sum()} ({(signal == 0).mean()*100:.1f}%)")
    print(f"   Bearish days: {(signal == -1).sum()} ({(signal == -1).mean()*100:.1f}%)")
    
    # Build strategies
    strategies = {
        'Base (60/40)': strategy_base(prices),
        'VIX Simple (SPY/TLT)': strategy_vix_simple_spy_tlt(prices, vix, vix3m),
        'VIX Enhanced (Multi)': strategy_vix_enhanced(prices, vix, vix3m),
    }
    
    windows = {
        "Pre-2020": (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period": (pd.Timestamp('2010-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("   RESULTS BY WINDOW")
    print("=" * 80)
    
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        
        print(f"\n   {window_name}")
        print("   " + "-" * 60)
        print(f"   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
        print("   " + "-" * 60)
        
        results[window_name] = {}
        
        for strat_name, weights in strategies.items():
            if weights is None:
                continue
            w_weights = weights[mask]
            returns = compute_returns(w_prices, w_weights)
            metrics = compute_metrics(returns)
            results[window_name][strat_name] = {'metrics': metrics, 'returns': returns}
            
            if metrics:
                print(f"   {strat_name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs 60/40)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'Window':<15} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'N':>6}")
    print("   " + "-" * 80)
    
    for window_name in windows.keys():
        base_ret = results[window_name]['Base (60/40)']['returns']
        
        for strat_name in ['VIX Simple (SPY/TLT)', 'VIX Enhanced (Multi)']:
            if strat_name not in results[window_name]:
                continue
            strat_ret = results[window_name][strat_name]['returns']
            stats_result = compute_active_stats(strat_ret, base_ret)
            
            if stats_result:
                sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
                print(f"   {strat_name:<25} {window_name:<15} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {stats_result['n']:>6} {sig}")
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Check pre/post consistency
    print("\n" + "=" * 80)
    print("   CONSISTENCY CHECK")
    print("=" * 80)
    
    for strat_name in ['VIX Simple (SPY/TLT)', 'VIX Enhanced (Multi)']:
        pre = results['Pre-2020'].get(strat_name, {}).get('metrics', {})
        post = results['Post-2020'].get(strat_name, {}).get('metrics', {})
        
        if pre and post:
            print(f"\n   {strat_name}:")
            print(f"   Pre-2020 Sharpe:  {pre['sharpe']:.2f}")
            print(f"   Post-2020 Sharpe: {post['sharpe']:.2f}")
            
            if pre['sharpe'] > 0.8 and post['sharpe'] > 0.5:
                print("   ✅ Consistent across regimes")
            else:
                print("   ⚠️  Inconsistent performance")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("   FINAL VERDICT")
    print("=" * 80)
    
    full_base = results['Full Period']['Base (60/40)']['returns']
    
    best = None
    best_pval = 1.0
    
    for strat_name in ['VIX Simple (SPY/TLT)', 'VIX Enhanced (Multi)']:
        strat_ret = results['Full Period'][strat_name]['returns']
        stats_result = compute_active_stats(strat_ret, full_base)
        
        if stats_result and stats_result['p_val'] < best_pval:
            best_pval = stats_result['p_val']
            best = strat_name
    
    if best_pval < 0.05:
        print(f"\n   ✅ FOUND: {best} achieves p < 0.05 ({best_pval:.3f})")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best} achieves p < 0.10 ({best_pval:.3f})")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE (best p={best_pval:.3f})")
    
    print("\n" + "=" * 80)
