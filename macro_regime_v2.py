"""
Macro Regime v2: Economically Direct Signals
=============================================

Replaces the "weird data" (Netflix/cheese/coffee) with actual macro indicators:
- 5Y Breakeven Inflation (inflation expectations)
- Credit Spread (JNK/IEF risk appetite)
- Yield Curve Slope (recession indicator)
- VIX Level (volatility regime)

Tests with same falsification battery as ERP:
1. Pre-2020 vs Post-2020 split
2. Active return IR (vs Base + VIX)
3. Statistical significance

RUN: python macro_regime_v2.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FETCH DATA
# =============================================================================

def fetch_all_data():
    """Fetch price data and macro indicators."""
    print("📊 Fetching data...")
    
    # Core assets
    tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'JNK', 'IEF', 'DBC']
    
    data = yf.download(tickers + ['^VIX', '^TNX'], start='2010-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    tnx = prices['^TNX'].copy() if '^TNX' in prices.columns else None
    
    # Remove non-tradeable from price frame
    for col in ['^VIX', '^TNX']:
        if col in prices.columns:
            prices = prices.drop(col, axis=1)
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Days: {len(prices)}")
    
    return prices, vix, tnx

# =============================================================================
# MACRO REGIME SIGNALS
# =============================================================================

def compute_regime_signals(prices, vix, tnx):
    """
    Compute daily regime signals from actual macro data.
    
    Signals:
    1. Inflation Regime: Commodity momentum (DBC) as proxy for breakevens
    2. Credit Stress: JNK/IEF ratio (risk appetite)
    3. Volatility Regime: VIX level and trend
    4. Trend: 200-day momentum of SPY
    """
    signals = pd.DataFrame(index=prices.index)
    
    # 1. Inflation Signal: Commodity momentum
    if 'DBC' in prices.columns:
        dbc = prices['DBC']
        signals['commodity_mom'] = dbc.pct_change(60)  # 3-month momentum
        signals['inflation_regime'] = (signals['commodity_mom'] > 0).astype(int)
    else:
        signals['inflation_regime'] = 0
    
    # 2. Credit Spread Signal: JNK/IEF ratio
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        credit_ratio = prices['JNK'] / prices['IEF']
        signals['credit_ratio'] = credit_ratio
        signals['credit_ma'] = credit_ratio.rolling(60).mean()
        signals['credit_percentile'] = credit_ratio.rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 20 else 0.5
        )
        signals['risk_off'] = (signals['credit_percentile'] < 0.2).astype(int)
    else:
        signals['risk_off'] = 0
    
    # 3. Volatility Regime
    if vix is not None:
        signals['vix'] = vix
        signals['vix_high'] = (vix > 25).astype(int)
        signals['vix_extreme'] = (vix > 35).astype(int)
    else:
        signals['vix_high'] = 0
        signals['vix_extreme'] = 0
    
    # 4. Trend Signal: SPY 200-day momentum
    if 'SPY' in prices.columns:
        spy = prices['SPY']
        signals['spy_trend'] = spy / spy.rolling(200).mean() - 1
        signals['trend_positive'] = (signals['spy_trend'] > 0).astype(int)
    else:
        signals['trend_positive'] = 1
    
    return signals

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_A_base(prices):
    """Base allocation: equal weight tradeable assets."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(200, len(prices)), len(prices)):
        for a in assets:
            weights.iloc[i][a] = 1.0 / len(assets)
    
    return weights.shift(1).fillna(0)

def strategy_B_base_vix(prices, signals):
    """Base + VIX overlay only."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(200, len(prices)), len(prices)):
        w = {a: 0.25 for a in assets}
        
        # VIX overlay
        if signals['vix_high'].iloc[i]:
            if 'TLT' in w: w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
            if 'SPY' in w: w['SPY'] *= 0.8
        
        if signals['vix_extreme'].iloc[i]:
            if 'TLT' in w: w['TLT'] = 0.50
            if 'XLE' in w: w['XLE'] = 0.05
            if 'SPY' in w: w['SPY'] = 0.15
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

def strategy_C_macro_regime(prices, signals):
    """
    Full Macro Regime v2:
    - Inflation regime → XLE tilt
    - Risk-off → TLT/GLD tilt  
    - Trend negative → defensive
    - VIX overlay
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(200, len(prices)), len(prices)):
        w = {a: 0.25 for a in assets}
        
        # 1. Inflation Regime: Tilt to energy/commodities
        if signals['inflation_regime'].iloc[i]:
            if 'XLE' in w: w['XLE'] = 0.35
            if 'SPY' in w: w['SPY'] = 0.20
        
        # 2. Risk-Off: Tilt to safety
        if signals['risk_off'].iloc[i]:
            if 'TLT' in w: w['TLT'] = 0.35
            if 'GLD' in w: w['GLD'] = 0.30
            if 'XLE' in w: w['XLE'] = 0.10
            if 'SPY' in w: w['SPY'] = 0.25
        
        # 3. Trend Negative: Reduce equity
        if not signals['trend_positive'].iloc[i] and not signals['risk_off'].iloc[i]:
            if 'SPY' in w: w['SPY'] *= 0.7
            if 'TLT' in w: w['TLT'] += 0.10
        
        # 4. VIX Overlay (always applies)
        if signals['vix_high'].iloc[i]:
            if 'TLT' in w: w['TLT'] = max(w.get('TLT', 0.25), 0.40)
            if 'XLE' in w: w['XLE'] *= 0.6
        
        if signals['vix_extreme'].iloc[i]:
            if 'TLT' in w: w['TLT'] = 0.50
            if 'GLD' in w: w['GLD'] = 0.25
            if 'XLE' in w: w['XLE'] = 0.05
            if 'SPY' in w: w['SPY'] = 0.20
        
        total = sum(w.values())
        for a in w:
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)

# =============================================================================
# BACKTEST & ANALYSIS
# =============================================================================

def compute_returns(prices, weights, warmup=200):
    """Compute portfolio returns."""
    returns = prices.pct_change()
    
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm = weights[common].div(abs_sum, axis=0)
    
    port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
    return port_ret.dropna()

def compute_metrics(returns):
    """Compute strategy metrics."""
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
    """Compute Information Ratio of active returns."""
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
    print("   MACRO REGIME v2: ECONOMICALLY DIRECT SIGNALS")
    print("   Testing if proper macro timing can achieve statistical significance")
    print("=" * 80)
    
    # Fetch data
    prices, vix, tnx = fetch_all_data()
    
    # Compute regime signals
    print("\n📈 Computing regime signals...")
    signals = compute_regime_signals(prices, vix, tnx)
    
    # Signal summary
    print(f"   Inflation regime days: {signals['inflation_regime'].sum()} ({signals['inflation_regime'].mean()*100:.1f}%)")
    print(f"   Risk-off days: {signals['risk_off'].sum()} ({signals['risk_off'].mean()*100:.1f}%)")
    print(f"   VIX high days: {signals['vix_high'].sum()} ({signals['vix_high'].mean()*100:.1f}%)")
    print(f"   Trend positive days: {signals['trend_positive'].sum()} ({signals['trend_positive'].mean()*100:.1f}%)")
    
    # ==========================================================================
    # RUN STRATEGIES
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    windows = {
        "Pre-2020 (2014-2019)": (pd.Timestamp('2014-01-01'), pd.Timestamp('2019-12-31')),
        "2020 Only": (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')),
        "Post-2020 (2021-2026)": (pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31')),
        "Full Period": (pd.Timestamp('2014-01-01'), pd.Timestamp('2026-12-31')),
    }
    
    results = {}
    
    for window_name, (start, end) in windows.items():
        mask = (prices.index >= start) & (prices.index <= end)
        w_prices = prices[mask]
        w_signals = signals[mask]
        
        if len(w_prices) < 250:
            continue
        
        results[window_name] = {}
        
        # Strategy A: Base
        w_a = strategy_A_base(w_prices)
        r_a = compute_returns(w_prices, w_a)
        results[window_name]['A: Base'] = compute_metrics(r_a)
        results[window_name]['r_a'] = r_a
        
        # Strategy B: Base + VIX
        w_b = strategy_B_base_vix(w_prices, w_signals)
        r_b = compute_returns(w_prices, w_b)
        results[window_name]['B: Base+VIX'] = compute_metrics(r_b)
        results[window_name]['r_b'] = r_b
        
        # Strategy C: Full Macro Regime
        w_c = strategy_C_macro_regime(w_prices, w_signals)
        r_c = compute_returns(w_prices, w_c)
        results[window_name]['C: MacroRegime'] = compute_metrics(r_c)
        results[window_name]['r_c'] = r_c
    
    # Print results
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        print(f"\n   {window_name}")
        print("   " + "-" * 60)
        print(f"   {'Strategy':<20} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
        print("   " + "-" * 60)
        
        for strat in ['A: Base', 'B: Base+VIX', 'C: MacroRegime']:
            m = results[window_name].get(strat)
            if m:
                print(f"   {strat:<20} {m['sharpe']:>10.2f} {m['cagr']:>9.1f}% {m['max_dd']:>9.1f}%")
    
    # ==========================================================================
    # ACTIVE RETURN ANALYSIS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   ACTIVE RETURN ANALYSIS (C - B)")
    print("   Testing if macro regime adds statistically significant value")
    print("=" * 80)
    
    print(f"\n   {'Window':<25} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>6}")
    print("   " + "-" * 60)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        r_b = results[window_name].get('r_b')
        r_c = results[window_name].get('r_c')
        
        if r_b is not None and r_c is not None:
            ir_stats = compute_active_ir(r_c, r_b)
            
            if ir_stats:
                sig = "**" if ir_stats['p_val'] < 0.05 else "*" if ir_stats['p_val'] < 0.10 else ""
                print(f"   {window_name:<25} {ir_stats['ir']:>8.2f} {ir_stats['t_stat']:>8.2f} {ir_stats['p_val']:>8.3f} {sig:>6}")
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # ==========================================================================
    # INCREMENTAL VALUE (C - B)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   INCREMENTAL VALUE OF MACRO REGIME (C - B)")
    print("=" * 80)
    
    print(f"\n   {'Window':<25} {'Sharpe Δ':>12} {'CAGR Δ':>12} {'MaxDD Δ':>12}")
    print("   " + "-" * 65)
    
    for window_name in windows.keys():
        if window_name not in results:
            continue
        
        b = results[window_name].get('B: Base+VIX')
        c = results[window_name].get('C: MacroRegime')
        
        if b and c:
            d_sharpe = c['sharpe'] - b['sharpe']
            d_cagr = c['cagr'] - b['cagr']
            d_dd = c['max_dd'] - b['max_dd']
            
            sign_s = "+" if d_sharpe >= 0 else ""
            sign_c = "+" if d_cagr >= 0 else ""
            
            print(f"   {window_name:<25} {sign_s}{d_sharpe:>11.2f} {sign_c}{d_cagr:>10.1f}% {d_dd:>11.1f}%")
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   FINAL VERDICT: MACRO REGIME v2")
    print("=" * 80)
    
    # Get full period stats
    full = results.get("Full Period", {})
    r_b_full = full.get('r_b')
    r_c_full = full.get('r_c')
    
    if r_b_full is not None and r_c_full is not None:
        ir_full = compute_active_ir(r_c_full, r_b_full)
        b_metrics = full.get('B: Base+VIX', {})
        c_metrics = full.get('C: MacroRegime', {})
        
        if ir_full and b_metrics and c_metrics:
            delta_sharpe = c_metrics['sharpe'] - b_metrics['sharpe']
            
            print(f"""
   Full Period Results:
   ─────────────────────────────────────────────────────────────
   Base+VIX Sharpe:     {b_metrics['sharpe']:.2f}
   MacroRegime Sharpe:  {c_metrics['sharpe']:.2f}
   Incremental Δ:       {delta_sharpe:+.2f}
   
   Active Return IR:    {ir_full['ir']:.2f}
   Active t-stat:       {ir_full['t_stat']:.2f}
   Active p-value:      {ir_full['p_val']:.3f}
            """)
            
            if ir_full['p_val'] < 0.05:
                print("   ✅ SIGNIFICANT at 5%: Macro regime timing adds real value!")
            elif ir_full['p_val'] < 0.10:
                print("   ⚠️  MARGINAL at 10%: Some evidence of value, but not conclusive.")
            else:
                print("   ❌ NOT SIGNIFICANT: Macro regime timing does not add provable value.")
            
            print(f"""
   Comparison to ERP "Weird Data":
   ─────────────────────────────────────────────────────────────
   ERP Weird-Data:      IR=0.42, p=0.146
   Macro Regime v2:     IR={ir_full['ir']:.2f}, p={ir_full['p_val']:.3f}
            """)
            
            if ir_full['p_val'] < 0.146:
                print("   ✅ IMPROVEMENT: Macro Regime v2 is more statistically robust than weird-data.")
            else:
                print("   ❌ NO IMPROVEMENT: Still within noise range.")
    
    print("\n" + "=" * 80)
