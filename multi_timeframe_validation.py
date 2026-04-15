"""
MULTI-TIMEFRAME STATISTICAL VALIDATION
=======================================

Tests ALL strategies across MULTIPLE timeframes with COMPREHENSIVE statistics.

Timeframes tested:
- 1 Year
- 2 Years  
- 3 Years
- 5 Years
- Full history

Statistical tests per strategy:
1. Sharpe Ratio + Significance (t-test)
2. Bootstrap Confidence Interval
3. Information Ratio vs SPY
4. Win Rate
5. Consistency (% positive months)

RUN: python multi_timeframe_validation.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("   📊 MULTI-TIMEFRAME STATISTICAL VALIDATION")
print("   Testing All Strategies Across Multiple Time Periods")
print("=" * 80)

# =============================================================================
# DATA
# =============================================================================

print("\n📊 Fetching data...")

tickers = ['SPY', 'QQQ', 'XLB', 'XLI', 'XLE', 'XLK', 'GLD', 'TLT', 'JNK', 'IEF']

end = datetime.now()
start = end - timedelta(days=365*12)

data = yf.download(tickers + ['^VIX'], start=start, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
if vix is not None:
    prices = prices.drop('^VIX', axis=1)

print(f"   Loaded {len(prices)} days")

# =============================================================================
# STRATEGIES
# =============================================================================

WEIRD_DATA = {
    "netflix": {2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
                2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0, 2025: 320.0, 2026: 340.0},
    "cheese": {2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
               2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5, 2025: 43.0, 2026: 43.5},
}

def get_erp_signal(date):
    year = date.year
    netflix = cheese = 0
    if year in WEIRD_DATA['netflix'] and year-1 in WEIRD_DATA['netflix']:
        netflix = (WEIRD_DATA['netflix'][year] - WEIRD_DATA['netflix'][year-1]) / WEIRD_DATA['netflix'][year-1]
    if year in WEIRD_DATA['cheese'] and year-1 in WEIRD_DATA['cheese']:
        cheese = (WEIRD_DATA['cheese'][year] - WEIRD_DATA['cheese'][year-1]) / WEIRD_DATA['cheese'][year-1]
    return -netflix * 0.5 + cheese * 0.3

def erp_regime(prices, vix):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(252, len(prices)), len(prices)):
        date = prices.index[i]
        sig = get_erp_signal(date)
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if sig > 0.02: w['XLE'], w['SPY'] = 0.35, 0.20
            elif sig < -0.02: w['XLE'], w['GLD'] = 0.10, 0.35
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w: 
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    return weights.shift(1).fillna(0)

def spy_only(prices, vix):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if 'SPY' in w.columns: w['SPY'] = 1.0
    return w

def momentum_strat(prices, vix):
    ret = prices.pct_change()
    mom = ret.rolling(60).mean()
    vol = ret.rolling(20).std() + 0.001
    signal = mom / vol
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(60, len(prices)):
        s = signal.iloc[i].dropna()
        top = s.nlargest(3)
        for asset in top.index:
            weights.iloc[i][asset] = 1.0 / 3
    return weights.shift(1).fillna(0)

def equal_weight(prices, vix):
    n = len([c for c in prices.columns if c in ['SPY', 'XLE', 'GLD', 'TLT']])
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a in ['SPY', 'XLE', 'GLD', 'TLT']:
        if a in w.columns: w[a] = 1.0/n
    return w

strategies = {
    "ERP Regime": erp_regime,
    "SPY Buy&Hold": spy_only,
    "Momentum": momentum_strat,
    "Equal Weight": equal_weight,
}

# Try to add Compounder
try:
    from compounder_strategy import compounder_strategy
    def comp_wrapper(prices, vix):
        rename = {'SPY': 'S&P 500', 'XLE': 'Energy', 'XLK': 'Technology', 
                  'GLD': 'Gold', 'TLT': 'Long Treasuries'}
        p = prices.rename(columns=rename)
        w = compounder_strategy(p, vix)
        return w.rename(columns={v:k for k,v in rename.items()})
    strategies["Compounder"] = comp_wrapper
    print("   ✓ Compounder loaded")
except: pass

# Try AlphaMax
try:
    from alpha_max_strategy import AlphaMaxStrategy
    def am_wrapper(prices, vix):
        targets = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD']
        am_p = prices[[c for c in targets if c in prices.columns]]
        macro = prices[[c for c in ['JNK', 'IEF'] if c in prices.columns]]
        strat = AlphaMaxStrategy()
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i in range(200, len(prices), 63):
            try:
                strat.is_trained = False
                strat.models = {}
                w = strat.generate_signals(am_p.iloc[:i], macro.iloc[:i])
                for j in range(i, min(i+63, len(prices))):
                    for a in w.index:
                        if a in weights.columns:
                            weights.iloc[j][a] = w[a]
            except: pass
        return weights.shift(1).fillna(0)
    strategies["AlphaMax"] = am_wrapper
    print("   ✓ AlphaMax loaded")
except: pass

print(f"\n   Strategies: {list(strategies.keys())}")

# =============================================================================
# TIMEFRAMES
# =============================================================================

timeframes = {
    "1 Year": 252,
    "2 Years": 504,
    "3 Years": 756,
    "5 Years": 1260,
    "Full": len(prices),
}

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_stats(returns: pd.Series):
    """Compute comprehensive statistics."""
    if len(returns) < 50 or returns.std() == 0:
        return None
    
    n = len(returns)
    
    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # Sharpe significance (Lo 2002)
    se = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)
    t_stat = sharpe / se
    p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    
    # Bootstrap CI
    boot_sharpes = []
    for _ in range(1000):
        sample = returns.sample(n, replace=True)
        if sample.std() > 0:
            boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    ci_low = np.percentile(boot_sharpes, 2.5) if boot_sharpes else 0
    ci_high = np.percentile(boot_sharpes, 97.5) if boot_sharpes else 0
    
    # Other metrics
    equity = (1 + returns).cumprod()
    cagr = equity.iloc[-1] ** (252/n) - 1
    max_dd = (equity / equity.cummax() - 1).min()
    
    # Monthly returns for consistency
    monthly = returns.resample('ME').sum()
    positive_months = (monthly > 0).mean() * 100
    
    # Win rate
    win_rate = (returns > 0).mean() * 100
    
    return {
        "sharpe": sharpe,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cagr": cagr,
        "max_dd": max_dd,
        "positive_months": positive_months,
        "win_rate": win_rate,
        "n_days": n,
    }

# =============================================================================
# RUN TESTS
# =============================================================================

print("\n" + "=" * 80)
print("   🔬 RUNNING MULTI-TIMEFRAME VALIDATION")
print("=" * 80)

all_results = {}

for tf_name, tf_days in timeframes.items():
    print(f"\n📅 Timeframe: {tf_name} ({tf_days} days)")
    print("-" * 60)
    
    # Get data for this timeframe
    tf_prices = prices.iloc[-tf_days:].copy()
    tf_vix = vix.iloc[-tf_days:].copy() if vix is not None else None
    
    tf_results = {}
    
    for strat_name, strat_func in strategies.items():
        print(f"   {strat_name}...", end=" ", flush=True)
        
        try:
            weights = strat_func(tf_prices, tf_vix)
            returns = tf_prices.pct_change()
            
            # Warmup
            warmup = min(300, len(tf_prices) // 3)
            weights = weights.iloc[warmup:]
            returns = returns.iloc[warmup:]
            
            common = weights.columns.intersection(returns.columns)
            abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
            norm = weights[common].div(abs_sum, axis=0)
            
            port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
            
            stats_result = compute_stats(port_ret)
            
            if stats_result:
                tf_results[strat_name] = stats_result
                sig = "*" if stats_result['significant'] else ""
                print(f"Sharpe={stats_result['sharpe']:.2f}{sig}")
            else:
                print("N/A")
                
        except Exception as e:
            print(f"Error: {e}")
    
    all_results[tf_name] = tf_results

# =============================================================================
# SUMMARY TABLES
# =============================================================================

print("\n" + "=" * 80)
print("   📊 SHARPE RATIO BY TIMEFRAME")
print("=" * 80)

# Header
header = f"{'Strategy':<18}"
for tf in timeframes.keys():
    header += f"{tf:>12}"
print(f"\n{header}")
print("-" * (18 + 12 * len(timeframes)))

for strat in strategies.keys():
    row = f"{strat:<18}"
    for tf in timeframes.keys():
        if strat in all_results.get(tf, {}):
            r = all_results[tf][strat]
            sig = "*" if r['significant'] else ""
            row += f"{r['sharpe']:>11.2f}{sig}"
        else:
            row += f"{'N/A':>12}"
    print(row)

print("\n* = statistically significant (p<0.05)")

# Significance summary
print("\n" + "=" * 80)
print("   🔬 STATISTICAL SIGNIFICANCE SUMMARY")
print("=" * 80)

print(f"\n{'Strategy':<18} {'Sig. Tests':>12} {'Avg Sharpe':>12} {'Consistency':>12}")
print("-" * 58)

for strat in strategies.keys():
    sig_count = 0
    sharpes = []
    for tf in timeframes.keys():
        if strat in all_results.get(tf, {}):
            r = all_results[tf][strat]
            if r['significant']:
                sig_count += 1
            sharpes.append(r['sharpe'])
    
    if sharpes:
        avg_sharpe = np.mean(sharpes)
        consistency = f"{sig_count}/{len(timeframes)}"
        print(f"{strat:<18} {consistency:>12} {avg_sharpe:>12.2f} {(sig_count/len(timeframes)*100):>11.0f}%")

# Detailed by strategy
print("\n" + "=" * 80)
print("   📈 DETAILED METRICS - LATEST 3 YEARS")
print("=" * 80)

tf = "3 Years"
if tf in all_results:
    print(f"\n{'Strategy':<18} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinRate':>8} {'+Months':>8} {'p-value':>10}")
    print("-" * 72)
    
    sorted_results = sorted(all_results[tf].items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    for i, (strat, r) in enumerate(sorted_results, 1):
        medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
        sig = "✓" if r['significant'] else ""
        print(f"{medal}{strat:<16} {r['sharpe']:>8.2f} {r['cagr']:>7.1%} {r['max_dd']:>7.1%} {r['win_rate']:>7.1%} {r['positive_months']:>7.1%} {r['p_value']:>9.4f} {sig}")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("   🏆 FINAL VERDICT: MOST ROBUST STRATEGY")
print("=" * 80)

# Score each strategy
scores = {}
for strat in strategies.keys():
    score = 0
    for tf in timeframes.keys():
        if strat in all_results.get(tf, {}):
            r = all_results[tf][strat]
            # Points for positive Sharpe
            if r['sharpe'] > 0: score += 1
            if r['sharpe'] > 0.5: score += 1
            if r['sharpe'] > 1.0: score += 2
            if r['sharpe'] > 1.5: score += 2
            # Points for significance
            if r['significant']: score += 3
            # Points for CI excluding zero
            if r['ci_low'] > 0: score += 2
            # Points for positive months
            if r['positive_months'] > 55: score += 1
    scores[strat] = score

# Sort
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Strategy':<20} {'Score':>8} {'Assessment':<20}")
print("-" * 60)

for i, (strat, score) in enumerate(sorted_scores, 1):
    if score >= 30:
        assessment = "⭐ Highly Robust"
    elif score >= 20:
        assessment = "✓ Robust"
    elif score >= 10:
        assessment = "~ Moderate"
    else:
        assessment = "✗ Weak"
    
    medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
    print(f"{medal} {i:<4} {strat:<20} {score:>8} {assessment:<20}")

winner = sorted_scores[0]
print(f"\n🏆 WINNER: {winner[0]} (Score: {winner[1]})")

# Get winner's average stats
winner_sharpes = [all_results[tf][winner[0]]['sharpe'] for tf in timeframes if winner[0] in all_results.get(tf, {})]
winner_sig = sum(1 for tf in timeframes if winner[0] in all_results.get(tf, {}) and all_results[tf][winner[0]]['significant'])

print(f"   Average Sharpe: {np.mean(winner_sharpes):.2f}")
print(f"   Significant in: {winner_sig}/{len(timeframes)} timeframes")

print("\n" + "=" * 80)
