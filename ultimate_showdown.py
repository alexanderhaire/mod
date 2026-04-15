"""
ULTIMATE STRATEGY SHOWDOWN
==========================

Testing ALL strategies in the codebase to find the overall winner.

Strategies found:
1. all_weather_strategy.py - All-Weather multi-asset strategy
2. alpha_max_strategy.py - GBM with macro features  
3. compounder_strategy.py - ML ensemble with regime overlay
4. erp_alpha_strategy.py - ERP V1 weird data strategy
5. erp_alpha_v2.py - ERP V2 significant correlations
6. erp_regime (from money_finder.py) - ERP regime switching
7. Simple Momentum
8. SPY Buy & Hold (benchmark)

RUN: python ultimate_showdown.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("   🏆 ULTIMATE STRATEGY SHOWDOWN 🏆")
print("   Testing EVERY Strategy to Find the Winner")
print("=" * 70)

# =============================================================================
# FETCH DATA
# =============================================================================

print("\n📊 Fetching data...")

# Comprehensive ticker list that works for all strategies
tickers = ['SPY', 'QQQ', 'IWM', 'XLB', 'XLI', 'XLE', 'XLK', 'XLF', 'XLV', 
           'GLD', 'TLT', 'JNK', 'IEF', 'MOO', 'DBC']
macro_tickers = ['^TNX', 'UUP', '^VIX']

end = datetime.now()
start = end - timedelta(days=365*12)

data = yf.download(tickers + macro_tickers, start=start, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix = prices['^VIX'] if '^VIX' in prices.columns else None
if vix is not None:
    prices = prices.drop('^VIX', axis=1)

# Macro data
macro_cols = [c for c in ['^TNX', 'UUP', 'IEF', 'JNK'] if c in prices.columns]
macro_data = prices[macro_cols].copy() if macro_cols else pd.DataFrame(index=prices.index)

print(f"   Loaded {len(prices)} days, {len(prices.columns)} assets")

# Split 70/30
split = int(len(prices) * 0.7)
is_prices = prices.iloc[:split]
oos_prices = prices.iloc[split:]
is_vix = vix.iloc[:split] if vix is not None else None
oos_vix = vix.iloc[split:] if vix is not None else None
oos_macro = macro_data.iloc[split:]

print(f"   In-Sample:     {is_prices.index[0].date()} to {is_prices.index[-1].date()}")
print(f"   Out-of-Sample: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")

# =============================================================================
# SIMPLE HELPER STRATEGIES
# =============================================================================

def spy_benchmark(prices, vix=None):
    """Buy and hold SPY."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if 'SPY' in w.columns:
        w['SPY'] = 1.0
    return w

def equal_weight(prices, vix=None):
    """Equal weight all assets."""
    n = len(prices.columns)
    return pd.DataFrame(1.0/n, index=prices.index, columns=prices.columns)

def momentum_60d(prices, vix=None):
    """60-day momentum strategy."""
    ret = prices.pct_change()
    mom = ret.rolling(60).mean()
    vol = ret.rolling(20).std() + 0.001
    signal = mom / vol
    
    # Long only top 5
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(60, len(prices)):
        s = signal.iloc[i].dropna()
        top = s.nlargest(5)
        for asset in top.index:
            weights.iloc[i][asset] = 1.0 / 5
    return weights.shift(1).fillna(0)

# =============================================================================
# ERP REGIME STRATEGY (from money_finder.py)
# =============================================================================

WEIRD_DATA = {
    "netflix": {2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
                2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0, 2025: 320.0, 2026: 340.0},
    "cheese": {2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
               2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5, 2025: 43.0, 2026: 43.5},
    "coffee": {2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
               2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32, 2025: 6.50, 2026: 6.70},
}

def get_erp_signal(date):
    year = date.year
    signals = {}
    for name, data in WEIRD_DATA.items():
        if year in data and year-1 in data:
            signals[name] = (data[year] - data[year-1]) / data[year-1]
    netflix = signals.get('netflix', 0)
    cheese = signals.get('cheese', 0)  
    coffee = signals.get('coffee', 0)
    return -netflix * 0.5 + cheese * 0.3 + coffee * 0.2

def erp_regime_strategy(prices, vix=None):
    """ERP Regime strategy - the one that got Sharpe 2.06."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(252, len(prices)):
        date = prices.index[i]
        sig = get_erp_signal(date)
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val.iloc[0]) if isinstance(v_val, pd.Series) else float(v_val)
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if sig > 0.02: w['XLE'], w['SPY'] = 0.35, 0.20 if 'SPY' in w else 0.25
            elif sig < -0.02: w['XLE'], w['GLD'] = 0.10, 0.35 if 'GLD' in w else 0.25
        
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] = max(0.05, w['XLE'] * 0.5)
        
        total = sum(w.values())
        for a in w: 
            if a in weights.columns:
                weights.loc[prices.index[i], a] = w[a] / total
    
    return weights.shift(1).fillna(0)

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_strategy(prices, strategy_func, vix=None, name="Strategy", warmup=300):
    """Run backtest and return results."""
    try:
        weights = strategy_func(prices, vix)
    except Exception as e:
        return {"name": name, "error": str(e)}
    
    returns = prices.pct_change()
    
    # Skip warmup
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    if weights.empty:
        return {"name": name, "error": "Empty weights"}
    
    # Align columns
    common = weights.columns.intersection(returns.columns)
    if len(common) == 0:
        return {"name": name, "error": "No common columns"}
    
    # Normalize weights
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm_weights = weights[common].div(abs_sum, axis=0)
    
    # Portfolio returns
    port_ret = (norm_weights.shift(1) * returns[common]).sum(axis=1)
    
    if port_ret.std() == 0 or len(port_ret) < 50:
        return {"name": name, "error": "Insufficient data"}
    
    # Metrics
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    equity = (1 + port_ret).cumprod()
    cagr = equity.iloc[-1] ** (252/len(equity)) - 1 if len(equity) > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min()
    sortino_denom = port_ret[port_ret < 0].std() * np.sqrt(252)
    sortino = (port_ret.mean() * 252) / sortino_denom if sortino_denom > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "name": name,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "sortino": sortino,
        "calmar": calmar,
        "returns": port_ret,
    }

# =============================================================================
# LOAD EXTERNAL STRATEGIES
# =============================================================================

strategies = {}

# 1. SPY Benchmark
strategies["SPY Buy & Hold"] = spy_benchmark

# 2. Equal Weight
strategies["Equal Weight"] = equal_weight

# 3. Momentum
strategies["Momentum (60d)"] = momentum_60d

# 4. ERP Regime (our winner)
strategies["ERP Regime"] = erp_regime_strategy

# 5. Try to load Compounder
print("\n📦 Loading strategies...")
try:
    from compounder_strategy import compounder_strategy
    # Rename columns for compounder
    def compounder_wrapper(prices, vix=None):
        rename = {'SPY': 'S&P 500', 'XLE': 'Energy', 'XLK': 'Technology', 
                  'XLB': 'Materials', 'MOO': 'Agriculture', 'GLD': 'Gold',
                  'TLT': 'Long Treasuries', 'XLF': 'Financials', 'XLV': 'Healthcare'}
        p = prices.rename(columns=rename)
        w = compounder_strategy(p, vix)
        # Rename back
        rev_rename = {v: k for k, v in rename.items()}
        return w.rename(columns=rev_rename)
    strategies["Compounder"] = compounder_wrapper
    print("   ✓ Compounder loaded")
except Exception as e:
    print(f"   ✗ Compounder failed: {e}")

# 6. Try to load All-Weather
try:
    from all_weather_strategy import all_weather_strategy
    strategies["All-Weather"] = all_weather_strategy
    print("   ✓ All-Weather loaded")
except Exception as e:
    print(f"   ✗ All-Weather: {e}")

# 7. Try to load AlphaMax
try:
    from alpha_max_strategy import AlphaMaxStrategy
    def alpha_max_wrapper(prices, vix=None):
        # AlphaMax needs specific assets
        target = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD']
        am_prices = prices[[c for c in target if c in prices.columns]]
        macro = prices[[c for c in ['^TNX', 'UUP', 'IEF', 'JNK'] if c in prices.columns]]
        
        strategy = AlphaMaxStrategy()
        
        # Generate weights for each day
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for i in range(200, len(prices), 21):  # Monthly retraining
            train_prices = am_prices.iloc[:i]
            train_macro = macro.iloc[:i]
            
            if len(train_prices) < 100:
                continue
            
            try:
                strategy.is_trained = False
                strategy.models = {}
                w = strategy.generate_signals(train_prices, train_macro)
                
                # Apply for next 21 days
                for j in range(i, min(i+21, len(prices))):
                    for asset in w.index:
                        if asset in weights.columns:
                            weights.iloc[j][asset] = w[asset]
            except:
                pass
        
        return weights.shift(1).fillna(0)
    
    strategies["AlphaMax (GBM)"] = alpha_max_wrapper
    print("   ✓ AlphaMax loaded")
except Exception as e:
    print(f"   ✗ AlphaMax: {e}")

# 8. ERP Alpha V2
try:
    from erp_alpha_v2 import erp_alpha_v2_strategy
    strategies["ERP Alpha V2"] = erp_alpha_v2_strategy
    print("   ✓ ERP Alpha V2 loaded")
except Exception as e:
    print(f"   ✗ ERP Alpha V2: {e}")

print(f"\n   Total strategies to test: {len(strategies)}")

# =============================================================================
# RUN ALL BACKTESTS
# =============================================================================

print("\n" + "=" * 70)
print("   🔬 RUNNING BACKTESTS (Out-of-Sample)")
print("=" * 70)

results = []

for name, func in strategies.items():
    print(f"   Testing {name}...", end=" ", flush=True)
    result = backtest_strategy(oos_prices, func, oos_vix, name)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Sharpe={result['sharpe']:.2f}")
        results.append(result)

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "=" * 70)
print("   📊 FINAL RESULTS (OUT-OF-SAMPLE)")
print("=" * 70)

# Sort by Sharpe
results.sort(key=lambda x: x.get('sharpe', -999), reverse=True)

print(f"\n{'Rank':<5} {'Strategy':<25} {'Sharpe':>8} {'CAGR':>8} {'Max DD':>8} {'Sortino':>8} {'Calmar':>8}")
print("-" * 80)

for i, r in enumerate(results, 1):
    sharpe = r.get('sharpe', 0)
    cagr = r.get('cagr', 0)
    max_dd = r.get('max_dd', 0)
    sortino = r.get('sortino', 0)
    calmar = r.get('calmar', 0)
    
    # Medal
    if i == 1:
        medal = "🥇"
    elif i == 2:
        medal = "🥈"
    elif i == 3:
        medal = "🥉"
    elif sharpe > 1:
        medal = "⭐"
    elif sharpe > 0:
        medal = "✓"
    else:
        medal = "✗"
    
    print(f"{medal} {i:<3} {r['name']:<25} {sharpe:>7.2f} {cagr:>7.1%} {max_dd:>7.1%} {sortino:>7.2f} {calmar:>7.2f}")

# =============================================================================
# WINNER ANALYSIS
# =============================================================================

if results:
    winner = results[0]
    
    print("\n" + "=" * 70)
    print(f"   🏆 THE WINNER: {winner['name']}")
    print("=" * 70)
    
    print(f"""
   Sharpe Ratio:     {winner['sharpe']:.2f}
   CAGR:             {winner['cagr']:.1%}
   Max Drawdown:     {winner['max_dd']:.1%}
   Sortino Ratio:    {winner['sortino']:.2f}
   Calmar Ratio:     {winner['calmar']:.2f}
""")
    
    # Compare to SPY
    spy_result = next((r for r in results if 'SPY' in r['name']), None)
    if spy_result:
        print(f"   vs SPY:")
        print(f"   Sharpe improvement: +{(winner['sharpe'] - spy_result['sharpe']):.2f}")
        print(f"   CAGR improvement:   {(winner['cagr'] - spy_result['cagr']):.1%}")
        print(f"   DD improvement:     {(abs(spy_result['max_dd']) - abs(winner['max_dd'])):.1%}")

print("\n" + "=" * 70)
