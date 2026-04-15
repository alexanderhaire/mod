"""
Volatility Targeting & Risk Parity
===================================

Instead of trying to TIME the market, use volatility to SIZE positions.
This is a more robust, academically-backed approach.

Key ideas:
1. Inverse volatility weighting
2. Risk parity allocation
3. Volatility targeting (scale total exposure)

RUN: python vol_targeting.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'IEF', 'VNQ', 'EEM', 'IWM']
    
    data = yf.download(tickers + ['^VIX'], start='2006-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    if '^VIX' in prices.columns:
        prices = prices.drop('^VIX', axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices, vix

def strategy_equal_weight(prices):
    """Simple equal weight."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    n = len(prices.columns)
    
    for i in range(60, len(prices)):
        for col in prices.columns:
            weights.iloc[i][col] = 1.0 / n
    
    return weights.shift(1).fillna(0)

def strategy_inverse_vol(prices, lookback=60):
    """
    Inverse volatility weighting:
    - Higher vol assets get lower weight
    - Lower vol assets get higher weight
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    for i in range(lookback + 1, len(prices)):
        # Recent volatility
        vol = returns.iloc[i-lookback:i].std() * np.sqrt(252)
        vol = vol.replace(0, 0.01)  # Avoid div by zero
        
        # Inverse vol weights
        inv_vol = 1 / vol
        w = inv_vol / inv_vol.sum()
        
        for col in prices.columns:
            weights.iloc[i][col] = w.get(col, 0)
    
    return weights.shift(1).fillna(0)

def strategy_risk_parity(prices, lookback=60):
    """
    Risk parity: Equal risk contribution from each asset.
    Simplified version using inverse vol.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    for i in range(lookback + 1, len(prices)):
        # Recent volatility
        vol = returns.iloc[i-lookback:i].std() * np.sqrt(252)
        vol = vol.replace(0, 0.01)
        
        # Risk parity weights (equal vol contribution)
        inv_vol = 1 / vol
        w = inv_vol / inv_vol.sum()
        
        for col in prices.columns:
            weights.iloc[i][col] = w.get(col, 0)
    
    return weights.shift(1).fillna(0)

def strategy_vol_targeting(prices, vix, target_vol=0.10, lookback=60):
    """
    Volatility targeting:
    - Start with inverse vol weights
    - Scale total exposure to hit target volatility
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    for i in range(lookback + 1, len(prices)):
        # Recent portfolio volatility (using equal weight as proxy)
        port_ret = returns.iloc[i-lookback:i].mean(axis=1)
        realized_vol = port_ret.std() * np.sqrt(252)
        
        if realized_vol == 0:
            realized_vol = 0.10
        
        # Scale factor to hit target
        scale = min(target_vol / realized_vol, 1.5)  # Cap at 150%
        scale = max(scale, 0.5)  # Floor at 50%
        
        # Base weights (inverse vol)
        vol = returns.iloc[i-lookback:i].std() * np.sqrt(252)
        vol = vol.replace(0, 0.01)
        inv_vol = 1 / vol
        base_w = inv_vol / inv_vol.sum()
        
        # Scaled weights
        for col in prices.columns:
            weights.iloc[i][col] = base_w.get(col, 0) * scale
    
    return weights.shift(1).fillna(0)

def strategy_vol_timing(prices, vix, lookback=60):
    """
    Volatility timing:
    - When VIX is high, reduce risk
    - When VIX is low, increase risk
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    # VIX moving average
    vix_ma = vix.rolling(20).mean() if vix is not None else None
    
    for i in range(lookback + 1, len(prices)):
        # Base inverse vol weights
        vol = returns.iloc[i-lookback:i].std() * np.sqrt(252)
        vol = vol.replace(0, 0.01)
        inv_vol = 1 / vol
        base_w = inv_vol / inv_vol.sum()
        
        # VIX-based scaling
        scale = 1.0
        if vix_ma is not None and i < len(vix_ma):
            current_vix = vix.iloc[i]
            avg_vix = vix_ma.iloc[i]
            
            if current_vix > avg_vix * 1.3:
                scale = 0.6  # Reduce exposure
            elif current_vix < avg_vix * 0.8:
                scale = 1.2  # Increase exposure
        
        for col in prices.columns:
            weights.iloc[i][col] = base_w.get(col, 0) * scale
    
    return weights.shift(1).fillna(0)

def compute_returns(prices, weights, warmup=100):
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
    print("   VOLATILITY TARGETING & RISK PARITY")
    print("   Using vol to SIZE positions, not TIME the market")
    print("=" * 80)
    
    prices, vix = fetch_data()
    
    strategies = {
        'Equal Weight': strategy_equal_weight(prices),
        'Inverse Vol': strategy_inverse_vol(prices),
        'Risk Parity': strategy_risk_parity(prices),
        'Vol Targeting (10%)': strategy_vol_targeting(prices, vix, target_vol=0.10),
        'Vol Timing': strategy_vol_timing(prices, vix),
    }
    
    windows = {
        "Pre-2020": (pd.Timestamp('2008-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period": (pd.Timestamp('2008-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("   RESULTS")
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
            w_weights = weights[mask]
            returns = compute_returns(w_prices, w_weights)
            metrics = compute_metrics(returns)
            results[window_name][strat_name] = {'metrics': metrics, 'returns': returns}
            
            if metrics:
                print(f"   {strat_name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Equal Weight)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'Window':<15} {'IR':>8} {'t-stat':>8} {'p-val':>8}")
    print("   " + "-" * 70)
    
    best_pval = 1.0
    best_strat = None
    
    for window_name in ['Full Period']:
        base_ret = results[window_name]['Equal Weight']['returns']
        
        for strat_name in strategies.keys():
            if strat_name == 'Equal Weight':
                continue
            
            strat_ret = results[window_name][strat_name]['returns']
            stats_result = compute_active_stats(strat_ret, base_ret)
            
            if stats_result:
                sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
                print(f"   {strat_name:<25} {window_name:<15} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig}")
                
                if stats_result['p_val'] < best_pval:
                    best_pval = stats_result['p_val']
                    best_strat = strat_name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    print("\n" + "=" * 80)
    print("   VERDICT")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ SIGNIFICANT: {best_strat} (p={best_pval:.3f})")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best_strat} (p={best_pval:.3f})")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE (best p={best_pval:.3f})")
    
    print("\n" + "=" * 80)
