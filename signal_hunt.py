"""
Signal Hunting: Find Something That Works
==========================================

Testing multiple documented edges with proper falsification:
1. VIX Term Structure (contango = risk-on, backwardation = risk-off)
2. Cross-Asset Momentum (bonds/gold leading equities)
3. Volatility Regime (realized vs implied)
4. Combined Multi-Signal Ensemble

RUN: python signal_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA
# =============================================================================

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = [
        'SPY', 'QQQ', 'IWM',  # Equities
        'TLT', 'IEF', 'SHY',  # Bonds
        'GLD', 'SLV',          # Precious metals
        'XLE', 'XLF', 'XLK',  # Sectors
        'EEM', 'EFA',          # International
        'HYG', 'LQD',          # Credit
    ]
    
    # VIX and VIX3M for term structure
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
    print(f"   Assets: {len(prices.columns)}")
    
    return prices, vix, vix3m

# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def vix_term_structure_signal(vix, vix3m):
    """
    VIX Term Structure:
    - Contango (VIX < VIX3M): Normal, risk-on
    - Backwardation (VIX > VIX3M): Stress, risk-off
    
    Academic backing: VIX futures roll premium is well-documented.
    """
    if vix is None or vix3m is None:
        return None
    
    # Term structure ratio
    ratio = vix / vix3m
    
    # Signal: 1 = contango (bullish), -1 = backwardation (bearish)
    signal = pd.Series(0, index=ratio.index)
    signal[ratio < 0.95] = 1   # Strong contango
    signal[ratio > 1.05] = -1  # Backwardation
    
    return signal

def cross_asset_momentum_signal(prices, lookback=20):
    """
    Cross-asset momentum:
    - Bond strength (TLT up) often precedes equity weakness
    - Gold strength often signals risk-off coming
    """
    signals = pd.DataFrame(index=prices.index)
    
    # Bond momentum
    if 'TLT' in prices.columns:
        tlt_ret = prices['TLT'].pct_change(lookback)
        signals['bond_mom'] = tlt_ret
    
    # Gold momentum
    if 'GLD' in prices.columns:
        gld_ret = prices['GLD'].pct_change(lookback)
        signals['gold_mom'] = gld_ret
    
    # Credit spread proxy (HYG vs LQD)
    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        spread = (prices['HYG'] / prices['LQD']).pct_change(lookback)
        signals['credit_mom'] = spread
    
    # Composite: positive = risk-on, negative = risk-off
    # Bond strength + gold strength = risk-off signal
    signal = pd.Series(0, index=prices.index)
    
    if 'bond_mom' in signals.columns and 'gold_mom' in signals.columns:
        # If bonds AND gold are up, it's a risk-off signal
        risk_off = (signals['bond_mom'] > 0.02) & (signals['gold_mom'] > 0.02)
        risk_on = (signals['bond_mom'] < -0.02) & (signals['gold_mom'] < -0.02)
        
        signal[risk_off] = -1
        signal[risk_on] = 1
    
    return signal

def realized_vs_implied_vol(prices, vix):
    """
    Realized vs Implied Vol:
    - VIX >> realized: Risk premium is high, buy
    - VIX << realized: Risk premium is low, sell
    """
    if vix is None or 'SPY' not in prices.columns:
        return None
    
    # Realized vol (20-day)
    spy_ret = prices['SPY'].pct_change()
    realized = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # Annualized %
    
    # VRP = Implied - Realized
    vrp = vix - realized
    
    # Signal: high VRP = bullish, low VRP = bearish
    vrp_ma = vrp.rolling(60).mean()
    vrp_std = vrp.rolling(60).std()
    vrp_zscore = (vrp - vrp_ma) / vrp_std
    
    signal = pd.Series(0, index=prices.index)
    signal[vrp_zscore > 1] = 1   # High risk premium, buy
    signal[vrp_zscore < -1] = -1  # Low risk premium, sell
    
    return signal

def multi_signal_ensemble(signals_dict):
    """
    Combine multiple signals with equal weight.
    """
    df = pd.DataFrame(signals_dict)
    df = df.fillna(0)
    
    # Simple average
    ensemble = df.mean(axis=1)
    
    # Discretize
    result = pd.Series(0, index=ensemble.index)
    result[ensemble > 0.3] = 1
    result[ensemble < -0.3] = -1
    
    return result

# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_base(prices):
    """Equal weight benchmark."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    n = len([c for c in prices.columns if c in ['SPY', 'TLT', 'GLD', 'IEF']])
    
    for i in range(252, len(prices)):
        for col in ['SPY', 'TLT', 'GLD', 'IEF']:
            if col in prices.columns:
                weights.iloc[i][col] = 1.0 / n
    
    return weights.shift(1).fillna(0)

def strategy_signal_driven(prices, signal, risk_asset='SPY', safe_asset='TLT'):
    """
    Use signal to shift between risk and safe assets.
    Signal = 1: 80% risk, 20% safe
    Signal = 0: 50/50
    Signal = -1: 20% risk, 80% safe
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0:
            w_risk, w_safe = 0.80, 0.20
        elif s < 0:
            w_risk, w_safe = 0.20, 0.80
        else:
            w_risk, w_safe = 0.50, 0.50
        
        if risk_asset in prices.columns:
            weights.iloc[i][risk_asset] = w_risk
        if safe_asset in prices.columns:
            weights.iloc[i][safe_asset] = w_safe
    
    return weights.shift(1).fillna(0)

def strategy_multi_asset_signal(prices, signal):
    """
    Multi-asset allocation based on signal.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    risk_assets = ['SPY', 'QQQ', 'XLE', 'EEM']
    safe_assets = ['TLT', 'GLD', 'IEF']
    
    for i in range(252, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0:
            # Risk-on: 70% equities, 30% safe
            for a in risk_assets:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.70 / len([x for x in risk_assets if x in prices.columns])
            for a in safe_assets:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.30 / len([x for x in safe_assets if x in prices.columns])
        elif s < 0:
            # Risk-off: 30% equities, 70% safe
            for a in risk_assets:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.30 / len([x for x in risk_assets if x in prices.columns])
            for a in safe_assets:
                if a in prices.columns:
                    weights.iloc[i][a] = 0.70 / len([x for x in safe_assets if x in prices.columns])
        else:
            # Neutral: 50/50
            all_assets = [a for a in risk_assets + safe_assets if a in prices.columns]
            for a in all_assets:
                weights.iloc[i][a] = 1.0 / len(all_assets)
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

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

def compute_active_ir(r_c, r_b):
    common_idx = r_b.index.intersection(r_c.index)
    active = r_c.loc[common_idx] - r_b.loc[common_idx]
    active = active.dropna()
    
    if len(active) < 20 or active.std() == 0:
        return None
    
    ir = active.mean() / active.std() * np.sqrt(252)
    t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   SIGNAL HUNTING: Finding What Works")
    print("=" * 80)
    
    prices, vix, vix3m = fetch_data()
    
    # Generate signals
    print("\n📈 Generating signals...")
    
    vts_signal = vix_term_structure_signal(vix, vix3m)
    cam_signal = cross_asset_momentum_signal(prices)
    vrp_signal = realized_vs_implied_vol(prices, vix)
    
    # Ensemble
    signals_dict = {}
    if vts_signal is not None:
        signals_dict['vts'] = vts_signal
        print(f"   VIX Term Structure: {(vts_signal != 0).sum()} active days")
    if cam_signal is not None:
        signals_dict['cam'] = cam_signal
        print(f"   Cross-Asset Momentum: {(cam_signal != 0).sum()} active days")
    if vrp_signal is not None:
        signals_dict['vrp'] = vrp_signal
        print(f"   Vol Risk Premium: {(vrp_signal != 0).sum()} active days")
    
    ensemble_signal = multi_signal_ensemble(signals_dict)
    print(f"   Ensemble: {(ensemble_signal != 0).sum()} active days")
    
    # Build strategies
    strategies = {
        'Base (SPY/TLT)': strategy_base(prices),
        'VIX Term Structure': strategy_signal_driven(prices, vts_signal) if vts_signal is not None else None,
        'Cross-Asset Mom': strategy_signal_driven(prices, cam_signal) if cam_signal is not None else None,
        'Vol Risk Premium': strategy_signal_driven(prices, vrp_signal) if vrp_signal is not None else None,
        'Multi-Signal Ensemble': strategy_multi_asset_signal(prices, ensemble_signal),
    }
    
    # Remove None strategies
    strategies = {k: v for k, v in strategies.items() if v is not None}
    
    # Test windows
    windows = {
        "Pre-2020": (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full": (pd.Timestamp('2010-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    # Results
    print("\n" + "=" * 80)
    print("   RESULTS")
    print("=" * 80)
    
    all_results = {}
    
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        
        print(f"\n   {window_name} ({start.date()} to {end.date()})")
        print("   " + "-" * 60)
        print(f"   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
        print("   " + "-" * 60)
        
        all_results[window_name] = {}
        
        for strat_name, weights in strategies.items():
            w_weights = weights[mask]
            returns = compute_returns(w_prices, w_weights)
            metrics = compute_metrics(returns)
            
            all_results[window_name][strat_name] = {
                'metrics': metrics,
                'returns': returns
            }
            
            if metrics:
                print(f"   {strat_name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Active return analysis
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Base)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'Window':<12} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 75)
    
    best_strategy = None
    best_pval = 1.0
    
    for window_name in ['Full']:  # Focus on full period
        base_ret = all_results[window_name]['Base (SPY/TLT)']['returns']
        
        for strat_name in strategies.keys():
            if strat_name == 'Base (SPY/TLT)':
                continue
            
            strat_ret = all_results[window_name][strat_name]['returns']
            ir_stats = compute_active_ir(strat_ret, base_ret)
            
            if ir_stats:
                sig = "**" if ir_stats['p_val'] < 0.05 else "*" if ir_stats['p_val'] < 0.10 else ""
                print(f"   {strat_name:<25} {window_name:<12} {ir_stats['ir']:>8.2f} {ir_stats['t_stat']:>8.2f} {ir_stats['p_val']:>8.3f} {sig:>5}")
                
                if ir_stats['p_val'] < best_pval:
                    best_pval = ir_stats['p_val']
                    best_strategy = strat_name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Summary
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ FOUND SIGNIFICANT EDGE: {best_strategy} (p={best_pval:.3f})")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL EDGE: {best_strategy} (p={best_pval:.3f})")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE FOUND")
        print(f"   Best candidate: {best_strategy} (p={best_pval:.3f})")
    
    print("\n" + "=" * 80)
