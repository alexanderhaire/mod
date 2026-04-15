"""
Crypto-Based Trading Signals
==============================

Testing crypto data for alpha:
1. Bitcoin Momentum (crypto tends to trend strongly)
2. Bitcoin as Risk Indicator (BTC leads risk assets?)
3. Halving Cycle (4-year cycle of supply shocks)
4. Crypto-Equity Correlation (regime indicator)
5. Bitcoin Dominance proxy

Crypto is less efficient - maybe there's edge here!

RUN: python crypto_signals.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA
# =============================================================================

def fetch_data():
    print("📊 Fetching crypto and equity data...")
    
    # Traditional assets
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'EEM']
    
    # Crypto (BTC-USD and ETH-USD available on yfinance)
    crypto = ['BTC-USD', 'ETH-USD']
    
    data = yf.download(tickers + crypto, start='2015-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# BITCOIN HALVING CYCLES
# =============================================================================

# Bitcoin halving dates (supply shock events)
BTC_HALVINGS = [
    datetime(2012, 11, 28),  # First halving
    datetime(2016, 7, 9),    # Second halving
    datetime(2020, 5, 11),   # Third halving
    datetime(2024, 4, 20),   # Fourth halving (approximate)
]

def days_since_halving(date):
    """Calculate days since last Bitcoin halving."""
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    for i, halving in enumerate(BTC_HALVINGS):
        if date < halving:
            if i == 0:
                return 0  # Before first halving
            else:
                return (date - BTC_HALVINGS[i-1]).days
    
    # After last halving
    return (date - BTC_HALVINGS[-1]).days

def halving_cycle_position(date):
    """
    Position in 4-year halving cycle.
    Returns 0-1 where:
    - 0-0.25: First year after halving (historically bullish)
    - 0.25-0.5: Second year (peak year)
    - 0.5-0.75: Third year (correction)
    - 0.75-1.0: Fourth year (accumulation before next halving)
    """
    days = days_since_halving(date)
    cycle_length = 365 * 4  # ~4 years between halvings
    return (days % cycle_length) / cycle_length

# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def btc_momentum_signal(prices, lookback=60):
    """
    Bitcoin momentum as risk signal.
    When BTC is trending up, risk-on for all assets.
    """
    if 'BTC-USD' not in prices.columns:
        return None
    
    btc = prices['BTC-USD']
    btc_ret = btc.pct_change(lookback)
    btc_ma = btc.rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        # BTC above MA and positive momentum = bullish
        if btc.iloc[i] > btc_ma.iloc[i] and btc_ret.iloc[i] > 0:
            signal.iloc[i] = 1
        elif btc.iloc[i] < btc_ma.iloc[i] and btc_ret.iloc[i] < 0:
            signal.iloc[i] = -1
    
    return signal

def btc_eth_ratio_signal(prices, lookback=30):
    """
    BTC/ETH ratio as crypto risk indicator.
    High ratio (BTC dominance) = risk-off in crypto
    Low ratio (ETH outperforming) = risk-on in crypto
    """
    if 'BTC-USD' not in prices.columns or 'ETH-USD' not in prices.columns:
        return None
    
    ratio = prices['BTC-USD'] / prices['ETH-USD']
    ratio_ma = ratio.rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if ratio.iloc[i] > ratio_ma.iloc[i] * 1.1:
            signal.iloc[i] = -0.5  # BTC dominance = slight risk-off
        elif ratio.iloc[i] < ratio_ma.iloc[i] * 0.9:
            signal.iloc[i] = 1     # ETH outperformance = risk-on
    
    return signal

def halving_cycle_signal(prices):
    """
    Bitcoin halving cycle signal.
    Years 1-2 after halving = historically bullish for crypto/risk
    Years 3-4 = historically choppy/bearish
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        date = prices.index[i]
        cycle_pos = halving_cycle_position(date)
        
        if cycle_pos < 0.5:
            # First half of cycle (post-halving bull run)
            signal.iloc[i] = 1
        else:
            # Second half (correction/accumulation)
            signal.iloc[i] = -0.5
    
    return signal

def btc_volatility_signal(prices, lookback=30):
    """
    Bitcoin volatility as market stress indicator.
    High BTC vol often precedes/coincides with risk-off in traditional markets.
    """
    if 'BTC-USD' not in prices.columns:
        return None
    
    btc_ret = prices['BTC-USD'].pct_change()
    btc_vol = btc_ret.rolling(lookback).std() * np.sqrt(365)  # Annualized
    vol_ma = btc_vol.rolling(60).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback + 60, len(prices)):
        if btc_vol.iloc[i] > vol_ma.iloc[i] * 1.5:
            signal.iloc[i] = -1  # High BTC vol = caution
        elif btc_vol.iloc[i] < vol_ma.iloc[i] * 0.7:
            signal.iloc[i] = 0.5  # Low vol = calm
    
    return signal

def btc_equity_correlation_signal(prices, lookback=60):
    """
    BTC-SPY correlation as regime indicator.
    High correlation = crypto acting like risk asset
    Low/negative correlation = crypto as diversifier
    """
    if 'BTC-USD' not in prices.columns or 'SPY' not in prices.columns:
        return None
    
    btc_ret = prices['BTC-USD'].pct_change()
    spy_ret = prices['SPY'].pct_change()
    
    correlation = btc_ret.rolling(lookback).corr(spy_ret)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        corr = correlation.iloc[i]
        if corr > 0.5:
            # High correlation - crypto moving with stocks
            signal.iloc[i] = 0  # Neutral (no diversification benefit)
        elif corr < 0:
            # Negative correlation - crypto as hedge
            signal.iloc[i] = 1  # Bullish signal (diversification working)
    
    return signal

def crypto_risk_composite(prices):
    """Combine all crypto signals."""
    btc_mom = btc_momentum_signal(prices)
    btc_eth = btc_eth_ratio_signal(prices)
    halving = halving_cycle_signal(prices)
    btc_vol = btc_volatility_signal(prices)
    btc_corr = btc_equity_correlation_signal(prices)
    
    signals = [s for s in [btc_mom, btc_eth, halving, btc_vol, btc_corr] if s is not None]
    
    if len(signals) == 0:
        return None
    
    combined = pd.DataFrame(signals).T.mean(axis=1)
    return combined

# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_base(prices):
    """Equal weight traditional assets."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    core = ['SPY', 'TLT', 'GLD']
    
    for i in range(100, len(prices)):
        for col in core:
            if col in prices.columns:
                weights.iloc[i][col] = 1/3
    
    return weights.shift(1).fillna(0)

def strategy_with_crypto_signal(prices, signal):
    """Use crypto signal to time traditional assets."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(100, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0.3:
            # Crypto bullish = risk-on
            w = {'SPY': 0.55, 'QQQ': 0.15, 'TLT': 0.15, 'GLD': 0.15}
        elif s < -0.3:
            # Crypto bearish = risk-off
            w = {'SPY': 0.20, 'TLT': 0.50, 'GLD': 0.30}
        else:
            w = {'SPY': 0.40, 'TLT': 0.35, 'GLD': 0.25}
        
        for col, wt in w.items():
            if col in prices.columns:
                weights.iloc[i][col] = wt
    
    return weights.shift(1).fillna(0)

def strategy_with_btc_allocation(prices, signal):
    """Include small BTC allocation when signal is positive."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(100, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0.5:
            # Strong crypto signal = include BTC
            w = {'SPY': 0.45, 'TLT': 0.20, 'GLD': 0.15, 'BTC-USD': 0.20}
        elif s > 0:
            w = {'SPY': 0.45, 'TLT': 0.25, 'GLD': 0.20, 'BTC-USD': 0.10}
        elif s < -0.3:
            w = {'SPY': 0.25, 'TLT': 0.45, 'GLD': 0.30}
        else:
            w = {'SPY': 0.40, 'TLT': 0.35, 'GLD': 0.25}
        
        for col, wt in w.items():
            if col in prices.columns:
                weights.iloc[i][col] = wt
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

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
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val}

# =============================================================================
# DIRECT CRYPTO ANALYSIS
# =============================================================================

def analyze_btc_patterns(prices):
    """Analyze BTC return patterns."""
    if 'BTC-USD' not in prices.columns:
        return None
    
    btc = prices['BTC-USD']
    btc_ret = btc.pct_change().dropna()
    
    # Day of week
    dow_results = {}
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow in range(7):
        d_ret = btc_ret[btc_ret.index.dayofweek == dow]
        dow_results[dow_names[dow]] = {
            'mean': d_ret.mean() * 365 * 100,
            'sharpe': d_ret.mean() / d_ret.std() * np.sqrt(365) if d_ret.std() > 0 else 0,
            'n': len(d_ret)
        }
    
    # Halving cycle
    cycle_results = {}
    for i in range(len(btc_ret)):
        date = btc_ret.index[i]
        pos = halving_cycle_position(date)
        year = int(pos * 4) + 1  # Year 1-4
        if year not in cycle_results:
            cycle_results[year] = []
        cycle_results[year].append(btc_ret.iloc[i])
    
    cycle_stats = {}
    for year, rets in cycle_results.items():
        rets = np.array(rets)
        cycle_stats[f'Year {year}'] = {
            'mean': rets.mean() * 365 * 100,
            'sharpe': rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0,
            'n': len(rets)
        }
    
    return dow_results, cycle_stats

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   CRYPTO-BASED TRADING SIGNALS")
    print("   Testing BTC/ETH data for alpha")
    print("=" * 80)
    
    prices = fetch_data()
    
    # Analyze BTC patterns
    print("\n" + "=" * 80)
    print("   BITCOIN DAY-OF-WEEK RETURNS")
    print("=" * 80)
    
    dow_results, cycle_results = analyze_btc_patterns(prices)
    
    if dow_results:
        print(f"\n   {'Day':<8} {'Ann. Return':>12} {'Sharpe':>10} {'N':>8}")
        print("   " + "-" * 45)
        for day, data in dow_results.items():
            print(f"   {day:<8} {data['mean']:>11.1f}% {data['sharpe']:>10.2f} {data['n']:>8}")
    
    print("\n" + "=" * 80)
    print("   BITCOIN HALVING CYCLE RETURNS")
    print("=" * 80)
    
    if cycle_results:
        print(f"\n   {'Cycle Year':<12} {'Ann. Return':>12} {'Sharpe':>10} {'N':>8}")
        print("   " + "-" * 50)
        for year, data in sorted(cycle_results.items()):
            print(f"   {year:<12} {data['mean']:>11.1f}% {data['sharpe']:>10.2f} {data['n']:>8}")
    
    # Generate signals
    print("\n📈 Generating crypto signals...")
    
    signals = {
        'BTC Momentum': btc_momentum_signal(prices),
        'BTC/ETH Ratio': btc_eth_ratio_signal(prices),
        'Halving Cycle': halving_cycle_signal(prices),
        'BTC Volatility': btc_volatility_signal(prices),
        'BTC-Equity Correlation': btc_equity_correlation_signal(prices),
        'Crypto Composite': crypto_risk_composite(prices),
    }
    
    for name, sig in signals.items():
        if sig is not None:
            active = (sig.abs() > 0.1).sum()
            print(f"   {name}: {active} active days")
    
    # Build strategies
    strategies = {
        'Base (No Crypto)': strategy_base(prices),
    }
    
    for name, sig in signals.items():
        if sig is not None:
            strategies[f'{name} Signal'] = strategy_with_crypto_signal(prices, sig)
    
    # Also with BTC allocation
    composite = signals['Crypto Composite']
    if composite is not None:
        strategies['Composite + BTC Alloc'] = strategy_with_btc_allocation(prices, composite)
    
    # Test
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    results = {}
    
    print(f"\n   {'Strategy':<30} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("   " + "-" * 65)
    
    for name, weights in strategies.items():
        returns = compute_returns(prices, weights)
        metrics = compute_metrics(returns)
        results[name] = {'metrics': metrics, 'returns': returns}
        
        if metrics:
            print(f"   {name:<30} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Base)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<30} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 65)
    
    base_ret = results['Base (No Crypto)']['returns']
    
    best_pval = 1.0
    best_strat = None
    
    for name in strategies.keys():
        if name == 'Base (No Crypto)':
            continue
        
        strat_ret = results[name]['returns']
        stats_result = compute_active_stats(strat_ret, base_ret)
        
        if stats_result:
            sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
            print(f"   {name:<30} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig}")
            
            if stats_result['p_val'] < best_pval:
                best_pval = stats_result['p_val']
                best_strat = name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Verdict
    print("\n" + "=" * 80)
    print("   VERDICT")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ SIGNIFICANT: {best_strat} (p={best_pval:.3f})")
        print("   Crypto signals provide real edge! 🚀")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best_strat} (p={best_pval:.3f})")
        print("   Crypto might be onto something... 🪙")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE")
        print(f"   Best: {best_strat} (p={best_pval:.3f})")
        print("   Crypto signals don't help traditional assets. 📉")
    
    print("\n" + "=" * 80)
