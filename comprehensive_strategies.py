"""
Comprehensive Strategy Testing: All Approaches
================================================

Testing ALL the approaches mentioned in the research:
1. Value (CAPE, dividend yield proxies)
2. Carry (yield differentials)
3. Sentiment (put/call ratio proxy via VIX/VIX3M)
4. Seasonality (Sell in May, monthly patterns)
5. Multi-Factor Composite (combining all signals)

Plus comparing against what we already tested.

RUN: python comprehensive_strategies.py
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
        'SPY', 'QQQ', 'IWM',  # US Equities
        'EFA', 'EEM',          # International
        'TLT', 'IEF', 'SHY',  # Bonds
        'GLD',                 # Gold
        'DBC',                 # Commodities
        'VNQ',                 # REITs
        'HYG', 'LQD',         # Credit
    ]
    
    data = yf.download(tickers + ['^VIX', '^VIX3M'], start='2007-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    for col in ['^VIX', '^VIX3M']:
        if col in prices.columns:
            prices = prices.drop(col, axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, vix, vix3m

# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def value_signal(prices, lookback=252*5):
    """
    Value signal: assets that have underperformed over long horizon
    are cheaper and expected to mean-revert.
    Use 5-year relative performance as value proxy.
    """
    if len(prices) < lookback:
        return None
    
    # Long-term return (5 years)
    long_ret = prices.pct_change(lookback)
    
    # Value = inverse of long-term performance (cheap = underperformed)
    # Rank assets: lowest long-term return = highest value score
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 21, len(prices)):
        lr = long_ret.iloc[i].dropna()
        if len(lr) > 2:
            # Rank (lowest = highest value)
            ranks = lr.rank(ascending=True)  # 1 = best value (lowest return)
            # Normalize to -1 to 1
            norm = (ranks - ranks.mean()) / ranks.std() if ranks.std() > 0 else 0
            for asset in norm.index:
                signal.iloc[i][asset] = norm[asset]
    
    return signal

def carry_signal(prices, lookback=60):
    """
    Carry signal: assets with higher yield/momentum continuation
    Proxy: recent price appreciation + vol-adjusted returns
    (Real carry would need dividend yields, but we proxy with momentum quality)
    """
    # 3-month momentum as carry proxy (stable uptrend = positive carry characteristics)
    mom = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(252)
    
    # Sharpe-like ratio as carry proxy
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 21, len(prices)):
        m = mom.iloc[i]
        v = vol.iloc[i].replace(0, 0.1)
        sharp = m / v
        
        # Normalize
        if sharp.std() > 0:
            norm = (sharp - sharp.mean()) / sharp.std()
            for asset in norm.index:
                signal.iloc[i][asset] = norm.get(asset, 0)
    
    return signal

def sentiment_signal(vix, vix3m):
    """
    Sentiment signal using VIX term structure as proxy for fear/greed.
    Backwardation (VIX > VIX3M) = fear = contrarian buy
    Strong contango (VIX << VIX3M) = complacency = cautious
    """
    if vix is None or vix3m is None:
        return None
    
    ratio = vix / vix3m
    
    # Use percentile of ratio as signal
    signal = pd.Series(0.0, index=vix.index)
    
    for i in range(252, len(vix)):
        hist = ratio.iloc[max(0, i-252):i]
        current = ratio.iloc[i]
        pct = stats.percentileofscore(hist.dropna(), current) / 100
        
        # High percentile (backwardation, fear) = contrarian bullish
        # Low percentile (contango, complacency) = cautious
        if pct > 0.8:
            signal.iloc[i] = 1  # Extreme fear = buy signal
        elif pct < 0.2:
            signal.iloc[i] = -1  # Extreme complacency = cautious
    
    return signal

def seasonality_signal(prices):
    """
    Seasonality: Sell in May and go away pattern.
    Nov-Apr = historically strong months
    May-Oct = historically weak months
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        if month in [11, 12, 1, 2, 3, 4]:  # Strong months
            signal.iloc[i] = 1
        else:  # Weak months
            signal.iloc[i] = -1
    
    return signal

def momentum_signal(prices, lookback=252, skip=21):
    """
    12-month momentum with 1-month skip.
    """
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + skip, len(prices)):
        mom = prices.iloc[i-skip] / prices.iloc[i-lookback] - 1
        
        if mom.std() > 0:
            norm = (mom - mom.mean()) / mom.std()
            for asset in norm.index:
                signal.iloc[i][asset] = norm.get(asset, 0)
    
    return signal

def vix_regime_signal(vix):
    """
    VIX regime: simple high/low VIX signal.
    """
    if vix is None:
        return None
    
    signal = pd.Series(0.0, index=vix.index)
    vix_ma = vix.rolling(60).mean()
    
    for i in range(60, len(vix)):
        if vix.iloc[i] > vix_ma.iloc[i] * 1.2:
            signal.iloc[i] = -1  # Risk-off
        elif vix.iloc[i] < vix_ma.iloc[i] * 0.8:
            signal.iloc[i] = 1   # Risk-on
    
    return signal

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_base(prices):
    """Equal weight baseline."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    core = ['SPY', 'TLT', 'GLD', 'EFA']
    n = len([c for c in core if c in prices.columns])
    
    for i in range(252, len(prices)):
        for col in core:
            if col in prices.columns:
                weights.iloc[i][col] = 1.0 / n
    
    return weights.shift(1).fillna(0)

def strategy_single_signal(prices, asset_signal, risk_assets, safe_assets):
    """Apply a single signal to tilt between risk and safe assets."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        # Aggregate signal across risk assets
        if isinstance(asset_signal, pd.DataFrame):
            s = asset_signal.iloc[i][risk_assets].mean() if len(risk_assets) > 0 else 0
        else:
            s = asset_signal.iloc[i] if i < len(asset_signal) else 0
        
        if s > 0.5:
            risk_w, safe_w = 0.75, 0.25
        elif s < -0.5:
            risk_w, safe_w = 0.25, 0.75
        else:
            risk_w, safe_w = 0.50, 0.50
        
        for a in risk_assets:
            if a in prices.columns:
                weights.iloc[i][a] = risk_w / len(risk_assets)
        for a in safe_assets:
            if a in prices.columns:
                weights.iloc[i][a] = safe_w / len(safe_assets)
    
    return weights.shift(1).fillna(0)

def strategy_multi_factor(prices, signals_dict, risk_assets, safe_assets):
    """
    Multi-factor composite: average of all signals.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        # Collect all signals
        signal_values = []
        
        for name, sig in signals_dict.items():
            if sig is None:
                continue
            if isinstance(sig, pd.DataFrame):
                s = sig.iloc[i][risk_assets].mean() if len(risk_assets) > 0 else 0
            else:
                s = sig.iloc[i] if i < len(sig) else 0
            signal_values.append(s)
        
        # Composite signal = average
        if len(signal_values) > 0:
            composite = np.mean(signal_values)
        else:
            composite = 0
        
        # Apply to weights
        if composite > 0.3:
            risk_w, safe_w = 0.70, 0.30
        elif composite < -0.3:
            risk_w, safe_w = 0.30, 0.70
        else:
            risk_w, safe_w = 0.50, 0.50
        
        for a in risk_assets:
            if a in prices.columns:
                weights.iloc[i][a] = risk_w / len(risk_assets)
        for a in safe_assets:
            if a in prices.columns:
                weights.iloc[i][a] = safe_w / len(safe_assets)
    
    return weights.shift(1).fillna(0)

def strategy_voting(prices, signals_dict, risk_assets, safe_assets, threshold=0.6):
    """
    Voting system: only shift when majority of signals agree.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(252, len(prices)):
        bullish = 0
        bearish = 0
        total = 0
        
        for name, sig in signals_dict.items():
            if sig is None:
                continue
            if isinstance(sig, pd.DataFrame):
                s = sig.iloc[i][risk_assets].mean() if len(risk_assets) > 0 else 0
            else:
                s = sig.iloc[i] if i < len(sig) else 0
            
            total += 1
            if s > 0.3:
                bullish += 1
            elif s < -0.3:
                bearish += 1
        
        # Apply voting threshold
        if total > 0:
            if bullish / total >= threshold:
                risk_w, safe_w = 0.80, 0.20
            elif bearish / total >= threshold:
                risk_w, safe_w = 0.20, 0.80
            else:
                risk_w, safe_w = 0.50, 0.50
        else:
            risk_w, safe_w = 0.50, 0.50
        
        for a in risk_assets:
            if a in prices.columns:
                weights.iloc[i][a] = risk_w / len(risk_assets)
        for a in safe_assets:
            if a in prices.columns:
                weights.iloc[i][a] = safe_w / len(safe_assets)
    
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

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   COMPREHENSIVE STRATEGY TESTING")
    print("   Testing ALL approaches from research summary")
    print("=" * 80)
    
    prices, vix, vix3m = fetch_data()
    
    risk_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    safe_assets = ['TLT', 'GLD', 'IEF']
    
    # Generate all signals
    print("\n📈 Generating signals...")
    
    value_sig = value_signal(prices)
    carry_sig = carry_signal(prices)
    mom_sig = momentum_signal(prices)
    sentiment_sig = sentiment_signal(vix, vix3m)
    season_sig = seasonality_signal(prices)
    vix_sig = vix_regime_signal(vix)
    
    signals = {
        'value': value_sig,
        'carry': carry_sig,
        'momentum': mom_sig,
        'sentiment': sentiment_sig,
        'seasonality': season_sig,
        'vix_regime': vix_sig,
    }
    
    for name, sig in signals.items():
        if sig is not None:
            if isinstance(sig, pd.DataFrame):
                active = (sig.abs().sum(axis=1) > 0.1).sum()
            else:
                active = (sig.abs() > 0.1).sum()
            print(f"   {name}: {active} active days")
    
    # Build strategies
    print("\n📊 Building strategies...")
    
    strategies = {
        'Base (Equal Weight)': strategy_base(prices),
        'Value Only': strategy_single_signal(prices, value_sig, risk_assets, safe_assets) if value_sig is not None else None,
        'Carry Only': strategy_single_signal(prices, carry_sig, risk_assets, safe_assets) if carry_sig is not None else None,
        'Momentum Only': strategy_single_signal(prices, mom_sig, risk_assets, safe_assets) if mom_sig is not None else None,
        'Sentiment Only': strategy_single_signal(prices, sentiment_sig, risk_assets, safe_assets) if sentiment_sig is not None else None,
        'Seasonality Only': strategy_single_signal(prices, season_sig, risk_assets, safe_assets),
        'VIX Regime Only': strategy_single_signal(prices, vix_sig, risk_assets, safe_assets) if vix_sig is not None else None,
        'Multi-Factor Average': strategy_multi_factor(prices, signals, risk_assets, safe_assets),
        'Voting (60% agree)': strategy_voting(prices, signals, risk_assets, safe_assets, threshold=0.6),
        'Voting (50% agree)': strategy_voting(prices, signals, risk_assets, safe_assets, threshold=0.5),
    }
    
    # Remove None strategies
    strategies = {k: v for k, v in strategies.items() if v is not None}
    
    # Test
    windows = {
        "Pre-2020": (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31')),
        "Post-2020": (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')),
        "Full": (pd.Timestamp('2010-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("   RESULTS")
    print("=" * 80)
    
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        
        print(f"\n   {window_name}")
        print("   " + "-" * 65)
        print(f"   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
        print("   " + "-" * 65)
        
        results[window_name] = {}
        
        for strat_name, weights in strategies.items():
            w_weights = weights[mask]
            returns = compute_returns(w_prices, w_weights)
            metrics = compute_metrics(returns)
            results[window_name][strat_name] = {'metrics': metrics, 'returns': returns}
            
            if metrics:
                print(f"   {strat_name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Base)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 60)
    
    best_pval = 1.0
    best_strat = None
    
    base_ret = results['Full']['Base (Equal Weight)']['returns']
    
    all_results = []
    
    for strat_name in strategies.keys():
        if strat_name == 'Base (Equal Weight)':
            continue
        
        strat_ret = results['Full'][strat_name]['returns']
        stats_result = compute_active_stats(strat_ret, base_ret)
        
        if stats_result:
            sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
            print(f"   {strat_name:<25} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig}")
            
            all_results.append({
                'strategy': strat_name,
                'ir': stats_result['ir'],
                'p_val': stats_result['p_val'],
                'sharpe': results['Full'][strat_name]['metrics']['sharpe']
            })
            
            if stats_result['p_val'] < best_pval:
                best_pval = stats_result['p_val']
                best_strat = strat_name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Summary table
    print("\n" + "=" * 80)
    print("   COMPLETE RESULTS SUMMARY (ranked by p-value)")
    print("=" * 80)
    
    all_results_sorted = sorted(all_results, key=lambda x: x['p_val'])
    
    print(f"\n   {'Rank':<5} {'Strategy':<25} {'Sharpe':>8} {'IR':>8} {'p-val':>8}")
    print("   " + "-" * 60)
    
    for i, r in enumerate(all_results_sorted):
        sig = "**" if r['p_val'] < 0.05 else "*" if r['p_val'] < 0.10 else ""
        print(f"   {i+1:<5} {r['strategy']:<25} {r['sharpe']:>8.2f} {r['ir']:>8.2f} {r['p_val']:>8.3f} {sig}")
    
    print("\n" + "=" * 80)
    print("   FINAL VERDICT")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ FOUND SIGNIFICANT EDGE: {best_strat} (p={best_pval:.3f})")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL EDGE: {best_strat} (p={best_pval:.3f})")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE")
        print(f"   Best candidate: {best_strat} (p={best_pval:.3f})")
    
    print("\n" + "=" * 80)
