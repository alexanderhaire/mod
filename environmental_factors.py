"""
Environmental & Health Factors
================================

Testing unconventional environmental signals:
1. Weather/Sunshine Effect (academically documented! Hirshleifer & Shumway 2003)
2. Earthquake/Seismic Activity
3. Flu Season (seasonal health patterns)
4. SAD (Seasonal Affective Disorder) - winter depression effect

We'll use proxies since we don't have real-time data:
- Sunshine: Use seasonal patterns (summer = more sun)
- Earthquakes: Proxy via volatility spikes in specific sectors
- Health: Flu season (Oct-Mar historically)

RUN: python environmental_factors.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SIGNAL BUILDERS
# =============================================================================

def sunshine_effect_signal(prices):
    """
    Sunshine Effect (Hirshleifer & Shumway 2003):
    - Sunny weather in NYC correlates with positive market returns
    - Proxy: Use seasonal patterns (more daylight hours = proxy for sunshine)
    - Summer months have more sun, winter has less
    
    Academic finding: ~0.5% monthly difference between sunny and cloudy days
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        
        # Daylight hours proxy (Northern Hemisphere)
        # Peak sunshine: June (month 6)
        # Minimum: December (month 12)
        
        # Calculate distance from peak summer (June 21)
        if month <= 6:
            sun_factor = month / 6  # Increasing towards summer
        else:
            sun_factor = (12 - month + 1) / 6  # Decreasing after summer
        
        # Scale to -1 to +1 (winter = -1, summer = +1)
        signal.iloc[i] = (sun_factor - 0.5) * 2
    
    return signal

def sad_effect_signal(prices):
    """
    Seasonal Affective Disorder (SAD) Effect:
    - Studies show markets underperform in fall/winter due to trader mood
    - Kamstra, Kramer, Levi (2003): "Winter Blues"
    - Effect strongest in northern latitudes
    
    Signal: Negative in SAD months (Oct-Mar), positive otherwise
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        
        # SAD months: October through March
        if month in [10, 11, 12, 1, 2, 3]:
            signal.iloc[i] = -1  # Cautious during SAD season
        else:
            signal.iloc[i] = 1   # Risk-on during sunny months
    
    return signal

def flu_season_signal(prices):
    """
    Flu Season Effect:
    - Flu season (Oct-Apr) historically sees different market behavior
    - Healthcare sector often outperforms
    - Consumer discretionary may underperform (sick people shop less)
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        
        if month in [12, 1, 2]:  # Peak flu
            signal.iloc[i] = -1  # Defensive
        elif month in [10, 11, 3, 4]:  # Shoulder flu season
            signal.iloc[i] = -0.5
        else:
            signal.iloc[i] = 0.5  # Non-flu season
    
    return signal

def disaster_proxy_signal(prices):
    """
    Natural disaster/high uncertainty proxy:
    - Use realized volatility as proxy for disaster/stress periods
    - High vol = potential disaster/uncertainty period
    - This captures earthquakes, storms, pandemics, etc. indirectly
    """
    if 'SPY' not in prices.columns:
        return None
    
    spy_ret = prices['SPY'].pct_change()
    realized_vol = spy_ret.rolling(20).std() * np.sqrt(252)
    vol_ma = realized_vol.rolling(60).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(60, len(prices)):
        if realized_vol.iloc[i] > vol_ma.iloc[i] * 1.5:
            signal.iloc[i] = -1  # High stress/disaster period
        elif realized_vol.iloc[i] < vol_ma.iloc[i] * 0.7:
            signal.iloc[i] = 1   # Calm period
    
    return signal

def temperature_anomaly_signal(prices):
    """
    Temperature anomaly proxy:
    - Extreme temperatures (hot summers, cold winters) can affect
      energy sector, agriculture, retail
    - Proxy: deviation from typical seasonal pattern
    
    We simulate this with calendar-based expectations
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        day = prices.index[i].day
        
        # Expected temperature pattern (sin wave peaking in July)
        day_of_year = prices.index[i].dayofyear
        expected_temp = np.sin((day_of_year - 80) * 2 * np.pi / 365)
        
        # Random anomaly simulation (in reality would use actual temp data)
        # For now, use year's position in decade as pseudo-random
        year = prices.index[i].year
        anomaly = np.sin(year * 1.5 + day_of_year * 0.01)
        
        # Signal based on anomaly
        if abs(anomaly) > 0.7:
            signal.iloc[i] = -0.5 if anomaly > 0 else 0.5
    
    return signal

def hurricane_season_signal(prices):
    """
    Hurricane Season (June-November):
    - Atlantic hurricane season affects energy, insurance, retail
    - Peak: August-October
    """
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        
        if month in [8, 9, 10]:  # Peak hurricane
            signal.iloc[i] = -0.5  # Cautious
        elif month in [6, 7, 11]:  # Shoulder season
            signal.iloc[i] = -0.25
        else:
            signal.iloc[i] = 0.3
    
    return signal

def combined_environmental(prices):
    """Combine all environmental signals."""
    sunshine = sunshine_effect_signal(prices)
    sad = sad_effect_signal(prices)
    flu = flu_season_signal(prices)
    disaster = disaster_proxy_signal(prices)
    hurricane = hurricane_season_signal(prices)
    
    # Average (handle None)
    signals = [sunshine, sad, flu, hurricane]
    if disaster is not None:
        signals.append(disaster)
    
    combined = pd.DataFrame(signals).T.mean(axis=1)
    return combined

# =============================================================================
# STRATEGIES
# =============================================================================

def fetch_data():
    print("📊 Fetching data...")
    
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'XLE', 'XLV', 'XLU']
    
    data = yf.download(tickers, start='2010-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices

def strategy_base(prices):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        for col in ['SPY', 'TLT', 'GLD']:
            if col in prices.columns:
                weights.iloc[i][col] = 1/3
    
    return weights.shift(1).fillna(0)

def strategy_with_signal(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0.3:
            w = {'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}
        elif s < -0.3:
            w = {'SPY': 0.2, 'TLT': 0.5, 'GLD': 0.3}
        else:
            w = {'SPY': 0.4, 'TLT': 0.35, 'GLD': 0.25}
        
        for col, wt in w.items():
            if col in prices.columns:
                weights.iloc[i][col] = wt
    
    return weights.shift(1).fillna(0)

def strategy_sector_rotation(prices, signal):
    """Rotate between sectors based on environmental signal."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0.3:
            # Fair weather: growth
            w = {'SPY': 0.5, 'XLE': 0.2, 'TLT': 0.15, 'GLD': 0.15}
        elif s < -0.3:
            # Stress: defensive
            w = {'XLV': 0.3, 'XLU': 0.25, 'TLT': 0.30, 'GLD': 0.15}
        else:
            w = {'SPY': 0.35, 'XLV': 0.15, 'TLT': 0.30, 'GLD': 0.20}
        
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   ENVIRONMENTAL & HEALTH FACTORS")
    print("   Testing: Weather, Disasters, Flu Season, SAD Effect")
    print("=" * 80)
    
    prices = fetch_data()
    
    # Generate signals
    print("\n📈 Generating environmental signals...")
    
    signals = {
        'Sunshine Effect': sunshine_effect_signal(prices),
        'SAD Effect': sad_effect_signal(prices),
        'Flu Season': flu_season_signal(prices),
        'Hurricane Season': hurricane_season_signal(prices),
        'Disaster Proxy': disaster_proxy_signal(prices),
        'Combined Environmental': combined_environmental(prices),
    }
    
    for name, sig in signals.items():
        if sig is not None:
            active = (sig.abs() > 0.1).sum()
            print(f"   {name}: {active} active days")
    
    # Build strategies
    strategies = {
        'Base': strategy_base(prices),
    }
    
    for name, sig in signals.items():
        if sig is not None:
            strategies[f'{name}'] = strategy_with_signal(prices, sig)
    
    # Also test sector rotation with combined
    combined_sig = signals['Combined Environmental']
    strategies['Sector Rotation (Env)'] = strategy_sector_rotation(prices, combined_sig)
    
    # Test
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    results = {}
    
    print(f"\n   {'Strategy':<28} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("   " + "-" * 65)
    
    for name, weights in strategies.items():
        returns = compute_returns(prices, weights)
        metrics = compute_metrics(returns)
        results[name] = {'metrics': metrics, 'returns': returns}
        
        if metrics:
            print(f"   {name:<28} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE (vs Base)")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<28} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 65)
    
    base_ret = results['Base']['returns']
    
    all_results = []
    best_pval = 1.0
    best_strat = None
    
    for name in strategies.keys():
        if name == 'Base':
            continue
        
        strat_ret = results[name]['returns']
        stats_result = compute_active_stats(strat_ret, base_ret)
        
        if stats_result:
            sig_mark = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
            print(f"   {name:<28} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig_mark}")
            
            all_results.append({
                'strategy': name,
                'ir': stats_result['ir'],
                'p_val': stats_result['p_val'],
                'sharpe': results[name]['metrics']['sharpe']
            })
            
            if stats_result['p_val'] < best_pval:
                best_pval = stats_result['p_val']
                best_strat = name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Direct seasonal analysis
    print("\n" + "=" * 80)
    print("   MONTHLY RETURNS (SPY)")
    print("=" * 80)
    
    spy_ret = prices['SPY'].pct_change().dropna()
    
    print(f"\n   {'Month':<12} {'Ann. Return':>12} {'Sharpe':>10} {'N':>8}")
    print("   " + "-" * 50)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for m in range(1, 13):
        m_rets = spy_ret[spy_ret.index.month == m]
        ann_ret = m_rets.mean() * 252 * 100
        sharpe = m_rets.mean() / m_rets.std() * np.sqrt(252) if m_rets.std() > 0 else 0
        print(f"   {month_names[m-1]:<12} {ann_ret:>11.1f}% {sharpe:>10.2f} {len(m_rets):>8}")
    
    # Verdict
    print("\n" + "=" * 80)
    print("   VERDICT")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ SIGNIFICANT: {best_strat} (p={best_pval:.3f})")
        print("   The weather gods favor this strategy! ☀️")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best_strat} (p={best_pval:.3f})")
        print("   There might be something in the air... 🌤️")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE")
        print(f"   Best: {best_strat} (p={best_pval:.3f})")
        print("   Markets don't care about the weather. 🌧️")
    
    print("\n" + "=" * 80)
