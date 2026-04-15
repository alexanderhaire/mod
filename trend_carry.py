"""
Trend + Carry Strategy
======================

Simple, robust signals that have long academic backing:
1. Trend: 12-month price momentum (time-series momentum)
2. Carry: Dividend yield for equities, real yield for bonds

No complex regime classification - just follow the data.

RUN: python trend_carry.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FETCH DATA
# =============================================================================

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'EFA', 'EEM', 'IWM', 'QQQ', 'DBC']
    
    data = yf.download(tickers + ['^VIX'], start='2006-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    if '^VIX' in prices.columns:
        prices = prices.drop('^VIX', axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices, vix

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_base(prices):
    """Equal weight all assets."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    n_assets = len(prices.columns)
    
    for i in range(252, len(prices)):
        for col in prices.columns:
            weights.iloc[i][col] = 1.0 / n_assets
    
    return weights.shift(1).fillna(0)

def strategy_trend(prices, lookback=252):
    """
    Time-series momentum: Long assets with positive 12M returns,
    weight proportional to momentum strength.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    mom = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(60).std() * np.sqrt(252)
    
    for i in range(lookback + 60, len(prices)):
        m = mom.iloc[i]
        v = vol.iloc[i]
        
        # Risk-adjusted momentum
        signal = m / v.replace(0, 0.2)
        
        # Long only: only positive momentum
        signal[signal < 0] = 0
        
        if signal.sum() > 0:
            w = signal / signal.sum()
        else:
            # All negative momentum: equal weight defensive
            w = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        
        for col in prices.columns:
            weights.iloc[i][col] = w.get(col, 0)
    
    return weights.shift(1).fillna(0)

def strategy_trend_with_vol_target(prices, vix, lookback=252, vol_target=0.15):
    """
    Time-series momentum with volatility targeting.
    Scale exposure based on recent vol.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    mom = prices.pct_change(lookback)
    asset_vol = prices.pct_change().rolling(60).std() * np.sqrt(252)
    
    for i in range(lookback + 60, len(prices)):
        m = mom.iloc[i]
        v = asset_vol.iloc[i]
        
        # Risk-adjusted momentum
        signal = m / v.replace(0, 0.2)
        
        # Long only
        signal[signal < 0] = 0
        
        if signal.sum() > 0:
            w = signal / signal.sum()
        else:
            w = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        
        # Vol targeting: scale down when vol is high
        if vix is not None and i < len(vix):
            current_vix = vix.iloc[i]
            if isinstance(current_vix, pd.Series):
                current_vix = current_vix.iloc[0]
            current_vix = float(current_vix) / 100  # VIX is in percentage
            
            # Scale to target vol (capped at 1.5x)
            vol_scalar = min(vol_target / max(current_vix, 0.10), 1.5)
            w = w * vol_scalar
        
        for col in prices.columns:
            weights.iloc[i][col] = w.get(col, 0)
    
    return weights.shift(1).fillna(0)

def strategy_dual_momentum(prices, lookback=252):
    """
    Dual momentum: Absolute + relative
    - Filter: Only long if 12M return > T-bills (assume 0% for simplicity)
    - Rank: Weight top performers more
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    mom = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(60).std() * np.sqrt(252)
    
    for i in range(lookback + 60, len(prices)):
        m = mom.iloc[i]
        v = vol.iloc[i]
        
        # Absolute momentum filter
        positive = m > 0
        
        if positive.sum() == 0:
            # All negative: hold cash (TLT proxy)
            if 'TLT' in prices.columns:
                weights.iloc[i]['TLT'] = 1.0
            continue
        
        # Relative momentum: rank positives
        pos_mom = m[positive]
        risk_adj = pos_mom / v[positive].replace(0, 0.2)
        
        # Weight top 3-5 assets
        top_n = min(4, len(pos_mom))
        top_assets = risk_adj.nlargest(top_n)
        
        if top_assets.sum() > 0:
            w = top_assets / top_assets.sum()
            for asset, weight in w.items():
                weights.iloc[i][asset] = weight
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

def compute_returns(prices, weights, warmup=300):
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
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        'sharpe': sharpe,
        'cagr': cagr * 100,
        'max_dd': max_dd * 100,
        'n_days': len(returns)
    }

def compute_active_ir(r_c, r_b):
    common_idx = r_b.index.intersection(r_c.index)
    r_b = r_b.loc[common_idx]
    r_c = r_c.loc[common_idx]
    
    active = r_c - r_b
    active = active.dropna()
    
    if len(active) < 20 or active.std() == 0:
        return None
    
    mean_active = active.mean()
    std_active = active.std()
    ir = mean_active / std_active * np.sqrt(252)
    
    t_stat = mean_active / (std_active / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   TREND + CARRY STRATEGY")
    print("   Simple, academically-backed approach")
    print("=" * 80)
    
    prices, vix = fetch_data()
    
    # Define strategies
    strategies = {
        'A: Equal Weight': lambda p, v: strategy_base(p),
        'B: Trend (12M)': lambda p, v: strategy_trend(p),
        'C: Trend + VolTarget': lambda p, v: strategy_trend_with_vol_target(p, v),
        'D: Dual Momentum': lambda p, v: strategy_dual_momentum(p),
    }
    
    # Define windows
    windows = {
        "Pre-2020 (2010-2019)": (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020 (2020-2026)": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period": (pd.Timestamp('2010-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    # Run all strategies in all windows
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        w_vix = vix[mask] if vix is not None else None
        
        if len(w_prices) < 400:
            continue
        
        results[window_name] = {}
        
        for strat_name, strat_func in strategies.items():
            weights = strat_func(w_prices, w_vix)
            returns = compute_returns(w_prices, weights)
            results[window_name][strat_name] = {
                'metrics': compute_metrics(returns),
                'returns': returns
            }
    
    # Print results
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        print(f"\n   {window_name}")
        print("   " + "-" * 60)
        print(f"   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
        print("   " + "-" * 60)
        
        for strat_name in strategies.keys():
            m = results[window_name][strat_name]['metrics']
            if m:
                print(f"   {strat_name:<25} {m['sharpe']:>10.2f} {m['cagr']:>9.1f}% {m['max_dd']:>9.1f}%")
    
    # Active return analysis
    print("\n" + "=" * 80)
    print("   ACTIVE RETURN vs EQUAL WEIGHT (Strategy A)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'Window':<20} {'IR':>8} {'t-stat':>8} {'p-val':>8}")
    print("   " + "-" * 75)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        r_a = results[window_name]['A: Equal Weight']['returns']
        
        for strat_name in ['B: Trend (12M)', 'C: Trend + VolTarget', 'D: Dual Momentum']:
            r_x = results[window_name][strat_name]['returns']
            ir_stats = compute_active_ir(r_x, r_a)
            
            if ir_stats:
                sig = "**" if ir_stats['p_val'] < 0.05 else "*" if ir_stats['p_val'] < 0.10 else ""
                print(f"   {strat_name:<25} {window_name:<20} {ir_stats['ir']:>8.2f} {ir_stats['t_stat']:>8.2f} {ir_stats['p_val']:>8.3f} {sig}")
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Final summary
    print("\n" + "=" * 80)
    print("   BEST PERFORMING STRATEGY")
    print("=" * 80)
    
    full = results.get("Full Period", {})
    best_sharpe = 0
    best_name = None
    
    for strat_name, data in full.items():
        m = data['metrics']
        if m and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = strat_name
    
    if best_name:
        m = full[best_name]['metrics']
        r_best = full[best_name]['returns']
        r_base = full['A: Equal Weight']['returns']
        ir_stats = compute_active_ir(r_best, r_base)
        
        print(f"""
   Winner: {best_name}
   ─────────────────────────────────────────────────────────────
   Sharpe:  {m['sharpe']:.2f}
   CAGR:    {m['cagr']:.1f}%
   MaxDD:   {m['max_dd']:.1f}%
        """)
        
        if ir_stats:
            print(f"""   vs Equal Weight:
   IR:      {ir_stats['ir']:.2f}
   t-stat:  {ir_stats['t_stat']:.2f}
   p-val:   {ir_stats['p_val']:.3f}
            """)
            
            if ir_stats['p_val'] < 0.05:
                print("   ✅ STATISTICALLY SIGNIFICANT improvement over equal weight!")
            elif ir_stats['p_val'] < 0.10:
                print("   ⚠️  Marginal improvement (p < 0.10)")
            else:
                print("   ❌ Not statistically significant vs equal weight")
    
    print("\n" + "=" * 80)
