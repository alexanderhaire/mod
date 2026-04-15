"""
Exotic Calendar & Lunar Factors
================================

Testing unconventional timing signals:
1. Day of Week (Monday effect, Friday effect - academically documented!)
2. Moon Phases (Full moon vs New moon - some studies show weak effects)
3. Month of Year (already tested, but included for completeness)
4. Chinese Calendar New Year effect
5. Mercury Retrograde (yes, we're going there)

Some of these are documented anomalies, others are fun to test.

RUN: python exotic_factors.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LUNAR CALCULATIONS
# =============================================================================

def get_moon_phase(date):
    """
    Calculate moon phase for a date.
    Returns: 0 = New Moon, 0.5 = Full Moon
    """
    # Known new moon date for reference
    known_new_moon = datetime(2000, 1, 6, 18, 14)
    
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    # Synodic month = 29.53 days
    synodic_month = 29.530588853
    
    days_since = (date - known_new_moon).total_seconds() / 86400
    phase = (days_since % synodic_month) / synodic_month
    
    return phase

def is_full_moon(date, tolerance=0.1):
    """Check if date is near full moon (phase ~0.5)."""
    phase = get_moon_phase(date)
    return abs(phase - 0.5) < tolerance

def is_new_moon(date, tolerance=0.1):
    """Check if date is near new moon (phase ~0 or ~1)."""
    phase = get_moon_phase(date)
    return phase < tolerance or phase > (1 - tolerance)

# =============================================================================
# MERCURY RETROGRADE (Yes, really)
# =============================================================================

# Mercury retrograde periods 2010-2026 (approximate dates)
MERCURY_RETROGRADE = [
    ("2010-12-10", "2010-12-30"),
    ("2011-03-30", "2011-04-23"),
    ("2011-08-02", "2011-08-26"),
    ("2011-11-24", "2011-12-13"),
    ("2012-03-12", "2012-04-04"),
    ("2012-07-14", "2012-08-08"),
    ("2012-11-06", "2012-11-26"),
    ("2013-02-23", "2013-03-17"),
    ("2013-06-26", "2013-07-20"),
    ("2013-10-21", "2013-11-10"),
    ("2014-02-06", "2014-02-28"),
    ("2014-06-07", "2014-07-01"),
    ("2014-10-04", "2014-10-25"),
    ("2015-01-21", "2015-02-11"),
    ("2015-05-18", "2015-06-11"),
    ("2015-09-17", "2015-10-09"),
    ("2016-01-05", "2016-01-25"),
    ("2016-04-28", "2016-05-22"),
    ("2016-08-30", "2016-09-22"),
    ("2016-12-19", "2017-01-08"),
    ("2017-04-09", "2017-05-03"),
    ("2017-08-12", "2017-09-05"),
    ("2017-12-03", "2017-12-22"),
    ("2018-03-22", "2018-04-15"),
    ("2018-07-26", "2018-08-19"),
    ("2018-11-16", "2018-12-06"),
    ("2019-03-05", "2019-03-28"),
    ("2019-07-07", "2019-07-31"),
    ("2019-10-31", "2019-11-20"),
    ("2020-02-16", "2020-03-09"),
    ("2020-06-18", "2020-07-12"),
    ("2020-10-13", "2020-11-03"),
    ("2021-01-30", "2021-02-20"),
    ("2021-05-29", "2021-06-22"),
    ("2021-09-27", "2021-10-18"),
    ("2022-01-14", "2022-02-03"),
    ("2022-05-10", "2022-06-03"),
    ("2022-09-09", "2022-10-02"),
    ("2022-12-29", "2023-01-18"),
    ("2023-04-21", "2023-05-14"),
    ("2023-08-23", "2023-09-15"),
    ("2023-12-13", "2024-01-01"),
    ("2024-04-01", "2024-04-25"),
    ("2024-08-04", "2024-08-28"),
    ("2024-11-25", "2024-12-15"),
    ("2025-03-14", "2025-04-07"),
    ("2025-07-17", "2025-08-11"),
    ("2025-11-09", "2025-11-29"),
]

def is_mercury_retrograde(date):
    """Check if date falls during Mercury retrograde."""
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    for start_str, end_str in MERCURY_RETROGRADE:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        if start <= date <= end:
            return True
    return False

# =============================================================================
# CHINESE NEW YEAR
# =============================================================================

# Chinese New Year dates (approximate market effects around these)
CHINESE_NEW_YEAR = [
    "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
    "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
    "2025-01-29", "2026-02-17"
]

def is_near_chinese_new_year(date, days_before=5, days_after=10):
    """Check if date is near Chinese New Year (often bullish for EM)."""
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    for cny_str in CHINESE_NEW_YEAR:
        cny = datetime.strptime(cny_str, "%Y-%m-%d")
        if (cny - timedelta(days=days_before)) <= date <= (cny + timedelta(days=days_after)):
            return True
    return False

# =============================================================================
# DATA
# =============================================================================

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'EEM']
    
    data = yf.download(tickers, start='2010-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices

# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def day_of_week_signal(prices):
    """
    Day of week effect:
    - Monday effect: historically weak (negative returns)
    - Friday: historically strong
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        dow = prices.index[i].dayofweek
        if dow == 0:  # Monday
            signal.iloc[i] = -1  # Historical Monday weakness
        elif dow == 4:  # Friday
            signal.iloc[i] = 1   # Historical Friday strength
        elif dow == 3:  # Thursday (often second strongest)
            signal.iloc[i] = 0.5
    
    return signal

def moon_phase_signal(prices):
    """
    Moon phase trading:
    - Full moon: historically slightly negative
    - New moon: historically slightly positive
    Some studies show ~0.1% difference per period.
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        date = prices.index[i]
        phase = get_moon_phase(date)
        
        # Distance from full moon (0.5)
        dist_from_full = abs(phase - 0.5)
        
        # More positive near new moon, more negative near full moon
        # Signal ranges from -1 (full) to +1 (new)
        signal.iloc[i] = (dist_from_full - 0.25) * 4  # Scale to roughly -1 to +1
    
    return signal

def mercury_retrograde_signal(prices):
    """
    Mercury retrograde signal:
    - During retrograde: be cautious (some claim higher volatility)
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        date = prices.index[i]
        if is_mercury_retrograde(date):
            signal.iloc[i] = -1  # Cautious during retrograde
        else:
            signal.iloc[i] = 0.5  # Normal otherwise
    
    return signal

def chinese_new_year_signal(prices):
    """
    Chinese New Year effect:
    - EM tends to rally around CNY
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        date = prices.index[i]
        if is_near_chinese_new_year(date):
            signal.iloc[i] = 1  # Bullish around CNY
    
    return signal

def all_exotic_ensemble(prices):
    """Combine all exotic signals."""
    dow = day_of_week_signal(prices)
    moon = moon_phase_signal(prices)
    mercury = mercury_retrograde_signal(prices)
    cny = chinese_new_year_signal(prices)
    
    # Equal weight average
    ensemble = (dow + moon + mercury + cny) / 4
    
    return ensemble

# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_base(prices):
    """Equal weight baseline."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        for col in ['SPY', 'TLT', 'GLD']:
            if col in prices.columns:
                weights.iloc[i][col] = 1/3
    
    return weights.shift(1).fillna(0)

def strategy_with_signal(prices, signal):
    """Apply signal to shift risk allocation."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0.3:
            # Risk-on
            w = {'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}
        elif s < -0.3:
            # Risk-off
            w = {'SPY': 0.2, 'TLT': 0.5, 'GLD': 0.3}
        else:
            # Neutral
            w = {'SPY': 0.4, 'TLT': 0.35, 'GLD': 0.25}
        
        for col, wt in w.items():
            if col in prices.columns:
                weights.iloc[i][col] = wt
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

def compute_returns(prices, weights, warmup=60):
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
# DAY-OF-WEEK ANALYSIS (Direct test)
# =============================================================================

def analyze_day_of_week_returns(prices):
    """Direct analysis of returns by day of week."""
    spy_ret = prices['SPY'].pct_change().dropna()
    
    results = {}
    for dow in range(5):
        dow_rets = spy_ret[spy_ret.index.dayofweek == dow]
        results[dow] = {
            'mean': dow_rets.mean() * 252 * 100,  # Annualized %
            'std': dow_rets.std() * np.sqrt(252) * 100,
            'sharpe': dow_rets.mean() / dow_rets.std() * np.sqrt(252) if dow_rets.std() > 0 else 0,
            'n': len(dow_rets)
        }
    
    return results

def analyze_moon_phase_returns(prices):
    """Direct analysis of returns by moon phase."""
    spy_ret = prices['SPY'].pct_change().dropna()
    
    full_moon_rets = []
    new_moon_rets = []
    other_rets = []
    
    for date, ret in spy_ret.items():
        if is_full_moon(date):
            full_moon_rets.append(ret)
        elif is_new_moon(date):
            new_moon_rets.append(ret)
        else:
            other_rets.append(ret)
    
    return {
        'full_moon': {'mean': np.mean(full_moon_rets) * 252 * 100, 'n': len(full_moon_rets)},
        'new_moon': {'mean': np.mean(new_moon_rets) * 252 * 100, 'n': len(new_moon_rets)},
        'other': {'mean': np.mean(other_rets) * 252 * 100, 'n': len(other_rets)}
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   EXOTIC CALENDAR & LUNAR FACTORS")
    print("   Testing: Moon, Mercury, Day-of-Week, Chinese New Year")
    print("=" * 80)
    
    prices = fetch_data()
    
    # Direct day-of-week analysis
    print("\n" + "=" * 80)
    print("   DAY-OF-WEEK RETURNS (SPY)")
    print("=" * 80)
    
    dow_results = analyze_day_of_week_returns(prices)
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    print(f"\n   {'Day':<12} {'Ann. Return':>12} {'Volatility':>12} {'Sharpe':>10} {'N':>8}")
    print("   " + "-" * 60)
    
    for dow, name in enumerate(dow_names):
        r = dow_results[dow]
        print(f"   {name:<12} {r['mean']:>11.1f}% {r['std']:>11.1f}% {r['sharpe']:>10.2f} {r['n']:>8}")
    
    # Moon phase analysis
    print("\n" + "=" * 80)
    print("   MOON PHASE RETURNS (SPY)")
    print("=" * 80)
    
    moon_results = analyze_moon_phase_returns(prices)
    
    print(f"\n   {'Phase':<15} {'Ann. Return':>12} {'N':>8}")
    print("   " + "-" * 40)
    for phase, data in moon_results.items():
        print(f"   {phase:<15} {data['mean']:>11.1f}% {data['n']:>8}")
    
    # Strategy tests
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    signals = {
        'Base': None,
        'Day-of-Week': day_of_week_signal(prices),
        'Moon Phase': moon_phase_signal(prices),
        'Mercury Retrograde': mercury_retrograde_signal(prices),
        'Chinese New Year': chinese_new_year_signal(prices),
        'All Exotic Combined': all_exotic_ensemble(prices),
    }
    
    strategies = {}
    for name, sig in signals.items():
        if sig is None:
            strategies[name] = strategy_base(prices)
        else:
            strategies[name] = strategy_with_signal(prices, sig)
    
    # Backtest
    results = {}
    
    print(f"\n   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("   " + "-" * 60)
    
    for name, weights in strategies.items():
        returns = compute_returns(prices, weights)
        metrics = compute_metrics(returns)
        results[name] = {'metrics': metrics, 'returns': returns}
        
        if metrics:
            print(f"   {name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Base)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 60)
    
    base_ret = results['Base']['returns']
    
    best_pval = 1.0
    best_strat = None
    
    for name in strategies.keys():
        if name == 'Base':
            continue
        
        strat_ret = results[name]['returns']
        stats_result = compute_active_stats(strat_ret, base_ret)
        
        if stats_result:
            sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
            print(f"   {name:<25} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig}")
            
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
        print("   The stars have aligned! 🌙✨")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best_strat} (p={best_pval:.3f})")
        print("   Mercury might be onto something... 🪐")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE")
        print(f"   Best: {best_strat} (p={best_pval:.3f})")
        print("   The universe is efficient after all. 🌌")
    
    print("\n" + "=" * 80)
