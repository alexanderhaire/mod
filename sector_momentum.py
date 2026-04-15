"""
Sector/Cross-Sectional Momentum
================================

Academic momentum effect is stronger in cross-section than time-series.
Buy winners, sell losers among sectors.

Key ideas:
1. Rank sectors by 12-month momentum
2. Long top N, short bottom N (or just long top N)
3. Rebalance monthly

RUN: python sector_momentum.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fetch_sector_data():
    print("📊 Fetching sector ETF data...")
    
    # US Sector ETFs
    sectors = [
        'XLK',  # Technology
        'XLF',  # Financials
        'XLV',  # Healthcare
        'XLE',  # Energy
        'XLI',  # Industrials
        'XLY',  # Consumer Discretionary
        'XLP',  # Consumer Staples
        'XLB',  # Materials
        'XLU',  # Utilities
        'XLRE', # Real Estate
        'XLC',  # Communication Services
    ]
    
    # Also get benchmark
    tickers = sectors + ['SPY']
    
    data = yf.download(tickers, start='2000-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Sectors: {len([c for c in prices.columns if c != 'SPY'])}")
    
    return prices

def strategy_equal_weight_sectors(prices):
    """Equal weight all sectors."""
    sectors = [c for c in prices.columns if c != 'SPY']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        for s in sectors:
            weights.iloc[i][s] = 1.0 / len(sectors)
    
    return weights.shift(1).fillna(0)

def strategy_momentum_long_only(prices, lookback=252, top_n=3):
    """
    Long-only sector momentum:
    - Rank sectors by 12-month return
    - Equal weight top N sectors
    """
    sectors = [c for c in prices.columns if c != 'SPY']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 21, len(prices)):
        # Skip most recent month (momentum reversal at short horizon)
        mom = prices[sectors].iloc[i-21] / prices[sectors].iloc[i-lookback] - 1
        
        # Rank and pick top N
        top = mom.nlargest(top_n).index.tolist()
        
        for s in top:
            weights.iloc[i][s] = 1.0 / top_n
    
    return weights.shift(1).fillna(0)

def strategy_momentum_long_short(prices, lookback=252, top_n=3, bottom_n=3):
    """
    Long-short sector momentum:
    - Long top N, short bottom N
    """
    sectors = [c for c in prices.columns if c != 'SPY']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 21, len(prices)):
        mom = prices[sectors].iloc[i-21] / prices[sectors].iloc[i-lookback] - 1
        
        top = mom.nlargest(top_n).index.tolist()
        bottom = mom.nsmallest(bottom_n).index.tolist()
        
        for s in top:
            weights.iloc[i][s] = 0.5 / top_n
        for s in bottom:
            weights.iloc[i][s] = -0.5 / bottom_n
    
    return weights.shift(1).fillna(0)

def strategy_momentum_with_skipping(prices, lookback=252, skip=21, top_n=3):
    """
    Momentum with 1-month skip (avoids short-term reversal).
    """
    sectors = [c for c in prices.columns if c != 'SPY']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + skip, len(prices)):
        # Calculate momentum excluding most recent month
        start_price = prices[sectors].iloc[i-lookback]
        end_price = prices[sectors].iloc[i-skip]  # Skip recent month
        mom = end_price / start_price - 1
        
        top = mom.nlargest(top_n).index.tolist()
        
        for s in top:
            weights.iloc[i][s] = 1.0 / top_n
    
    return weights.shift(1).fillna(0)

def strategy_relative_strength(prices, lookback=63, top_n=4):
    """
    Shorter-term relative strength (3-month momentum).
    """
    sectors = [c for c in prices.columns if c != 'SPY']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 5, len(prices)):
        mom = prices[sectors].iloc[i-5] / prices[sectors].iloc[i-lookback] - 1
        
        top = mom.nlargest(top_n).index.tolist()
        
        for s in top:
            weights.iloc[i][s] = 1.0 / top_n
    
    return weights.shift(1).fillna(0)

def compute_returns(prices, weights, warmup=252):
    returns = prices.pct_change()
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    
    # Handle long-short
    w_abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    
    port_ret = (weights[common].shift(1) * returns[common]).sum(axis=1)
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
    print("   SECTOR MOMENTUM STRATEGIES")
    print("   Cross-sectional momentum with academic backing")
    print("=" * 80)
    
    prices = fetch_sector_data()
    
    strategies = {
        'Equal Weight': strategy_equal_weight_sectors(prices),
        'Momentum Top 3 (12M)': strategy_momentum_long_only(prices, lookback=252, top_n=3),
        'Momentum Top 4 (12M)': strategy_momentum_long_only(prices, lookback=252, top_n=4),
        'Momentum w/ Skip': strategy_momentum_with_skipping(prices, lookback=252, skip=21, top_n=3),
        'Rel Strength (3M)': strategy_relative_strength(prices, lookback=63, top_n=4),
        'Long-Short': strategy_momentum_long_short(prices, lookback=252, top_n=3, bottom_n=3),
    }
    
    windows = {
        "Pre-2020": (pd.Timestamp('2005-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period": (pd.Timestamp('2005-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("   RESULTS")
    print("=" * 80)
    
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        
        if len(w_prices) < 300:
            continue
        
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
        if window_name not in results:
            continue
        base_ret = results[window_name]['Equal Weight']['returns']
        
        for strat_name in strategies.keys():
            if strat_name == 'Equal Weight':
                continue
            if strat_name not in results[window_name]:
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
    
    # vs SPY benchmark
    print("\n" + "=" * 80)
    print("   VS SPY (absolute benchmark)")
    print("=" * 80)
    
    if 'Full Period' in results:
        spy_ret = prices['SPY'].pct_change().iloc[252:].dropna()
        
        print(f"\n   {'Strategy':<25} {'Sharpe':>10} vs SPY IR")
        print("   " + "-" * 50)
        
        for strat_name, data in results['Full Period'].items():
            m = data['metrics']
            if m:
                strat_ret = data['returns']
                ir_stats = compute_active_stats(strat_ret, spy_ret)
                ir_val = ir_stats['ir'] if ir_stats else 'N/A'
                ir_str = f"{ir_val:.2f}" if isinstance(ir_val, float) else ir_val
                print(f"   {strat_name:<25} {m['sharpe']:>10.2f}     {ir_str}")
    
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
