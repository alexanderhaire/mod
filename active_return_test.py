"""
Active Return IR Test + Proxy Falsification
============================================

Tests:
1. Information Ratio of (C - B) active returns per window
2. Netflix substitution with alternative proxies (no re-optimization)
3. Negative controls: linear trend, random smooth

RUN: python active_return_test.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# WEIRD DATA - Original + Alternative Proxies
# =============================================================================

# Original Netflix (the "winner")
NETFLIX = {
    2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
    2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
    2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0, 
    2025: 320.0, 2026: 340.0
}

CHEESE = {
    2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
    2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
    2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
    2025: 43.0, 2026: 43.5
}

COFFEE = {
    2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
    2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
    2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
    2025: 6.50, 2026: 6.70
}

# =============================================================================
# ALTERNATIVE PROXIES (Tech adoption / activity proxies)
# =============================================================================

# Spotify Premium subscribers (millions) - similar tech adoption curve
SPOTIFY = {
    2010: 0.5, 2011: 1.5, 2012: 3.0, 2013: 6.0, 2014: 12.5,
    2015: 28.0, 2016: 48.0, 2017: 71.0, 2018: 96.0, 2019: 124.0,
    2020: 155.0, 2021: 180.0, 2022: 195.0, 2023: 220.0, 2024: 250.0,
    2025: 270.0, 2026: 290.0
}

# US Smartphone penetration (% of population)
SMARTPHONE = {
    2010: 20.2, 2011: 35.0, 2012: 45.0, 2013: 55.0, 2014: 64.0,
    2015: 72.0, 2016: 77.0, 2017: 80.0, 2018: 83.0, 2019: 85.0,
    2020: 87.0, 2021: 88.5, 2022: 89.5, 2023: 90.5, 2024: 91.5,
    2025: 92.0, 2026: 92.5
}

# US E-commerce as % of retail
ECOMMERCE = {
    2010: 4.2, 2011: 4.9, 2012: 5.4, 2013: 5.8, 2014: 6.4,
    2015: 7.3, 2016: 8.0, 2017: 9.0, 2018: 9.9, 2019: 11.0,
    2020: 14.0, 2021: 13.2, 2022: 14.5, 2023: 15.4, 2024: 16.0,
    2025: 17.0, 2026: 18.0
}

# Cloud computing revenue (billions)
CLOUD = {
    2010: 24.6, 2011: 37.0, 2012: 58.0, 2013: 82.0, 2014: 110.0,
    2015: 145.0, 2016: 196.0, 2017: 260.0, 2018: 330.0, 2019: 380.0,
    2020: 480.0, 2021: 590.0, 2022: 700.0, 2023: 800.0, 2024: 920.0,
    2025: 1050.0, 2026: 1200.0
}

# =============================================================================
# NEGATIVE CONTROLS
# =============================================================================

def generate_linear_trend():
    """Linear trend control - pure time index."""
    return {yr: yr - 2009 for yr in range(2010, 2027)}

def generate_random_smooth(seed=42):
    """Random smooth control - smoothed random walk."""
    np.random.seed(seed)
    years = list(range(2010, 2027))
    values = [100.0]
    for _ in range(len(years) - 1):
        change = np.random.normal(0.08, 0.15)  # ~8% growth with noise
        values.append(values[-1] * (1 + change))
    return dict(zip(years, values))

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def get_signal(date, proxy_data, weight=-0.5):
    """Get signal using a given proxy instead of Netflix."""
    year = date.year
    proxy_yoy = cheese_yoy = coffee_yoy = 0
    
    if year in proxy_data and year-1 in proxy_data:
        proxy_yoy = (proxy_data[year] - proxy_data[year-1]) / proxy_data[year-1]
    if year in CHEESE and year-1 in CHEESE:
        cheese_yoy = (CHEESE[year] - CHEESE[year-1]) / CHEESE[year-1]
    if year in COFFEE and year-1 in COFFEE:
        coffee_yoy = (COFFEE[year] - COFFEE[year-1]) / COFFEE[year-1]
    
    # Same coefficients as original - NO re-optimization
    signal = proxy_yoy * weight + cheese_yoy * 0.3 + coffee_yoy * 0.2
    return signal

def strategy_B(prices, vix):
    """Base + VIX only (no weird-data)."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(60, len(prices)), len(prices)):
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
            if 'SPY' in w: w['SPY'] *= 0.8
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

def strategy_C_with_proxy(prices, vix, proxy_data):
    """Full ERP with substituted proxy (no re-optimization)."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(60, len(prices)), len(prices)):
        date = prices.index[i]
        signal = get_signal(date, proxy_data)
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if signal > 0.02: w['XLE'], w['SPY'] = 0.35, 0.20
            elif signal < -0.02: w['XLE'], w['GLD'] = 0.10, 0.35
        
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

def compute_returns(prices, weights, warmup=60):
    """Compute portfolio returns."""
    returns = prices.pct_change()
    
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm = weights[common].div(abs_sum, axis=0)
    
    port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
    return port_ret.dropna()

def compute_active_ir(r_c, r_b):
    """Compute Information Ratio and t-stat of active returns."""
    active = r_c - r_b
    active = active.dropna()
    
    if len(active) < 20 or active.std() == 0:
        return None
    
    mean_active = active.mean()
    std_active = active.std()
    ir = mean_active / std_active * np.sqrt(252)
    
    # T-stat (simple, can add Newey-West later)
    t_stat = mean_active / (std_active / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {
        'mean_active_bps': mean_active * 10000,
        'std_active_bps': std_active * 10000,
        'ir': ir,
        't_stat': t_stat,
        'p_val': p_val,
        'n_days': len(active)
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   ACTIVE RETURN IR TEST + PROXY FALSIFICATION")
    print("=" * 80)
    
    # Fetch data
    print("\n📊 Fetching data...")
    tickers = ['SPY', 'XLE', 'GLD', 'TLT']
    data = yf.download(tickers + ['^VIX'], start='2010-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    if vix is not None:
        prices = prices.drop('^VIX', axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # ==========================================================================
    # PART 1: Active Return IR for Original Netflix (C - B)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   PART 1: ACTIVE RETURN IR (C - B) BY WINDOW")
    print("=" * 80)
    print("\n   Testing if weird-data increment is statistically significant")
    
    windows = {
        "Pre-2020 (2014-2019)": (pd.Timestamp('2014-01-01'), pd.Timestamp('2019-12-31')),
        "2020 Only": (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')),
        "Post-2020 (2021-2026)": (pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period (2014-2026)": (pd.Timestamp('2014-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    print(f"\n   {'Window':<25} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Mean(bps)':>10} {'Days':>6}")
    print("   " + "-" * 70)
    
    for window_name, (start, end) in windows.items():
        window_prices = prices[(prices.index >= start) & (prices.index <= end)]
        window_vix = vix[(vix.index >= start) & (vix.index <= end)] if vix is not None else None
        
        if len(window_prices) < 100:
            continue
        
        w_b = strategy_B(window_prices, window_vix)
        w_c = strategy_C_with_proxy(window_prices, window_vix, NETFLIX)
        
        r_b = compute_returns(window_prices, w_b)
        r_c = compute_returns(window_prices, w_c)
        
        # Align
        common_idx = r_b.index.intersection(r_c.index)
        r_b = r_b.loc[common_idx]
        r_c = r_c.loc[common_idx]
        
        ir_stats = compute_active_ir(r_c, r_b)
        
        if ir_stats:
            sig = "**" if ir_stats['p_val'] < 0.05 else "*" if ir_stats['p_val'] < 0.10 else ""
            print(f"   {window_name:<25} {ir_stats['ir']:>8.2f} {ir_stats['t_stat']:>8.2f} {ir_stats['p_val']:>8.3f} {ir_stats['mean_active_bps']:>10.1f} {ir_stats['n_days']:>6} {sig}")
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # ==========================================================================
    # PART 2: PROXY SUBSTITUTION (No Re-optimization)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   PART 2: PROXY SUBSTITUTION (No Re-optimization)")
    print("=" * 80)
    print("\n   Testing if Netflix is special or any smooth proxy works similarly")
    
    # Use full period for comparison
    full_prices = prices[(prices.index >= pd.Timestamp('2014-01-01'))]
    full_vix = vix[(vix.index >= pd.Timestamp('2014-01-01'))] if vix is not None else None
    
    # Compute B returns once
    w_b = strategy_B(full_prices, full_vix)
    r_b = compute_returns(full_prices, w_b)
    
    proxies = {
        "Netflix (Original)": NETFLIX,
        "Spotify": SPOTIFY,
        "Smartphone": SMARTPHONE,
        "E-commerce": ECOMMERCE,
        "Cloud": CLOUD,
        "Linear Trend (Control)": generate_linear_trend(),
        "Random Smooth #1": generate_random_smooth(seed=42),
        "Random Smooth #2": generate_random_smooth(seed=123),
        "Random Smooth #3": generate_random_smooth(seed=999),
    }
    
    print(f"\n   {'Proxy':<25} {'Sharpe':>8} {'Δ vs B':>8} {'IR':>8} {'t-stat':>8} {'p-val':>8}")
    print("   " + "-" * 75)
    
    proxy_results = []
    
    for proxy_name, proxy_data in proxies.items():
        w_c = strategy_C_with_proxy(full_prices, full_vix, proxy_data)
        r_c = compute_returns(full_prices, w_c)
        
        common_idx = r_b.index.intersection(r_c.index)
        r_b_aligned = r_b.loc[common_idx]
        r_c_aligned = r_c.loc[common_idx]
        
        sharpe_c = r_c_aligned.mean() / r_c_aligned.std() * np.sqrt(252)
        sharpe_b = r_b_aligned.mean() / r_b_aligned.std() * np.sqrt(252)
        delta = sharpe_c - sharpe_b
        
        ir_stats = compute_active_ir(r_c_aligned, r_b_aligned)
        
        if ir_stats:
            sig = "**" if ir_stats['p_val'] < 0.05 else "*" if ir_stats['p_val'] < 0.10 else ""
            print(f"   {proxy_name:<25} {sharpe_c:>8.2f} {delta:>+8.2f} {ir_stats['ir']:>8.2f} {ir_stats['t_stat']:>8.2f} {ir_stats['p_val']:>8.3f} {sig}")
            
            proxy_results.append({
                'name': proxy_name,
                'sharpe': sharpe_c,
                'delta': delta,
                'ir': ir_stats['ir'],
                't_stat': ir_stats['t_stat'],
                'p_val': ir_stats['p_val']
            })
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # ==========================================================================
    # PART 3: RANKING & INTERPRETATION
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   PART 3: PROXY RANKING BY INFORMATION RATIO")
    print("=" * 80)
    
    proxy_results_sorted = sorted(proxy_results, key=lambda x: x['ir'], reverse=True)
    
    print(f"\n   Rank  {'Proxy':<25} {'IR':>8} {'Δ Sharpe':>10} {'p-val':>8}")
    print("   " + "-" * 55)
    
    netflix_rank = None
    for i, r in enumerate(proxy_results_sorted):
        print(f"   {i+1:<5} {r['name']:<25} {r['ir']:>8.2f} {r['delta']:>+10.2f} {r['p_val']:>8.3f}")
        if r['name'] == "Netflix (Original)":
            netflix_rank = i + 1
    
    # Calculate control performance
    controls = [r for r in proxy_results if 'Smooth' in r['name'] or 'Trend' in r['name']]
    control_mean_ir = np.mean([r['ir'] for r in controls])
    control_mean_delta = np.mean([r['delta'] for r in controls])
    
    netflix_result = [r for r in proxy_results if 'Netflix' in r['name']][0]
    
    print("\n" + "=" * 80)
    print("   FINAL INTERPRETATION")
    print("=" * 80)
    
    print(f"""
   Netflix vs Controls:
   ────────────────────────────────────────────────────────────
   Netflix IR:       {netflix_result['ir']:.2f}
   Netflix Δ Sharpe: {netflix_result['delta']:+.2f}
   Netflix p-value:  {netflix_result['p_val']:.3f}
   Netflix Rank:     {netflix_rank} of {len(proxy_results)}
   
   Control Mean IR:       {control_mean_ir:.2f}
   Control Mean Δ Sharpe: {control_mean_delta:+.2f}
    """)
    
    if netflix_result['ir'] > control_mean_ir + 0.1 and netflix_result['p_val'] < 0.10:
        print("""   ✅ NETFLIX IS SPECIAL: 
   Netflix outperforms random smooth controls with marginal significance.
   The weird-data timing captures something beyond smooth trend exposure.""")
    elif abs(netflix_result['ir'] - control_mean_ir) < 0.1:
        print("""   ⚠️  NETFLIX = SMOOTH TREND:
   Netflix performs similarly to random smooth controls.
   The signal is likely acting as generic momentum/trend timing,
   not a Netflix-specific mechanism.""")
    else:
        print("""   ❌ NETFLIX UNDERPERFORMS CONTROLS:
   Netflix is worse than some random controls.
   The signal may be overfitted to the Netflix narrative.""")
    
    # Statistical significance summary
    print(f"""
   Statistical Significance:
   ────────────────────────────────────────────────────────────
   Active return t-stat: {netflix_result['t_stat']:.2f}
   p-value: {netflix_result['p_val']:.3f}
    """)
    
    if netflix_result['p_val'] < 0.05:
        print("   ✅ SIGNIFICANT at 5%: The +0.2 Sharpe increment is statistically real.")
    elif netflix_result['p_val'] < 0.10:
        print("   ⚠️  MARGINAL at 10%: Some evidence, but not conclusive.")
    else:
        print("   ❌ NOT SIGNIFICANT: The +0.2 Sharpe increment could be noise.")
    
    print("\n" + "=" * 80)
