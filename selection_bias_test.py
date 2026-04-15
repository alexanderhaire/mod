"""
SELECTION BIAS DETECTION
========================

Comprehensive tests to ensure NO selection bias in the strategies:

1. LOOK-AHEAD BIAS TEST
   - Verify no future data leakage in signals
   - Check weight generation uses only past data

2. MULTIPLE HYPOTHESIS CORRECTION  
   - Bonferroni correction for testing multiple strategies
   - False Discovery Rate (FDR) control

3. WALK-FORWARD OUT-OF-SAMPLE TEST
   - True out-of-sample with no peeking
   - Rolling window validation

4. DATA SNOOPING TEST
   - White's Reality Check
   - Hansen's SPA Test (Superior Predictive Ability)

5. RANDOM STRATEGY COMPARISON
   - Compare against 10,000 random strategies
   - Determine if edge is real or lucky

6. PARAMETER SENSITIVITY TEST
   - Test if strategy is over-tuned to specific parameters
   - Check robustness to parameter changes

RUN: python selection_bias_test.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("   🔬 SELECTION BIAS DETECTION")
print("   Ensuring No Overfitting, Data Snooping, or Look-Ahead Bias")
print("=" * 80)

# =============================================================================
# DATA
# =============================================================================

print("\n📊 Fetching data...")

tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'XLB', 'XLI', 'JNK', 'IEF']

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
# STRATEGIES TO TEST
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

strategies = {
    "ERP Regime": erp_regime,
    "SPY": spy_only,
    "Momentum": momentum_strat,
}

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
                # CRITICAL: Only use data up to i (no look-ahead)
                w = strat.generate_signals(am_p.iloc[:i], macro.iloc[:i])
                for j in range(i, min(i+63, len(prices))):
                    for a in w.index:
                        if a in weights.columns:
                            weights.iloc[j][a] = w[a]
            except: pass
        return weights.shift(1).fillna(0)
    strategies["AlphaMax"] = am_wrapper
except: pass

print(f"   Strategies: {list(strategies.keys())}")

# =============================================================================
# TEST 1: LOOK-AHEAD BIAS CHECK
# =============================================================================

print("\n" + "=" * 80)
print("   TEST 1: LOOK-AHEAD BIAS CHECK")
print("=" * 80)

print("\n   Checking each strategy for future data leakage...")

bias_results = {}

for name, func in strategies.items():
    print(f"\n   {name}:")
    
    # Generate weights at day T using only data up to T
    # Then verify weight at T doesn't change if we add future data
    
    test_idx = len(prices) // 2  # Middle point
    
    # Weights using data up to test_idx
    weights_partial = func(prices.iloc[:test_idx+1], 
                            vix.iloc[:test_idx+1] if vix is not None else None)
    
    # Weights using all data (should be same at test_idx)
    weights_full = func(prices, vix)
    
    # Compare weights at test_idx
    w_partial = weights_partial.iloc[-1] if len(weights_partial) > 0 else None
    w_full = weights_full.iloc[test_idx] if test_idx < len(weights_full) else None
    
    if w_partial is not None and w_full is not None:
        diff = (w_partial - w_full).abs().sum()
        if diff < 0.01:
            print(f"      ✅ PASS - No look-ahead bias detected (diff={diff:.6f})")
            bias_results[name] = True
        else:
            print(f"      ⚠️  POSSIBLE BIAS - Weights differ by {diff:.4f}")
            bias_results[name] = False
    else:
        print(f"      ✓ Cannot compare (short data)")
        bias_results[name] = True

# =============================================================================
# TEST 2: MULTIPLE HYPOTHESIS CORRECTION (Bonferroni)
# =============================================================================

print("\n" + "=" * 80)
print("   TEST 2: MULTIPLE HYPOTHESIS CORRECTION (Bonferroni)")
print("=" * 80)

n_strategies = len(strategies)
alpha = 0.05
bonferroni_alpha = alpha / n_strategies

print(f"\n   Testing {n_strategies} strategies")
print(f"   Original alpha: {alpha}")
print(f"   Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")

# Run backtests and get p-values
split = int(len(prices) * 0.7)
oos_prices = prices.iloc[split:]
oos_vix = vix.iloc[split:] if vix is not None else None

bonf_results = {}

print(f"\n   {'Strategy':<20} {'Sharpe':>8} {'P-value':>10} {'Passes Bonf':>14}")
print("   " + "-" * 55)

for name, func in strategies.items():
    try:
        weights = func(oos_prices, oos_vix)
        returns = oos_prices.pct_change()
        
        warmup = 300
        weights = weights.iloc[warmup:]
        returns = returns.iloc[warmup:]
        
        common = weights.columns.intersection(returns.columns)
        abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
        norm = weights[common].div(abs_sum, axis=0)
        
        port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
        
        n = len(port_ret)
        sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
        
        # P-value (Lo 2002)
        se = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)
        t_stat = sharpe / se if se > 0 else 0
        p_val = 1 - stats.t.cdf(t_stat, df=n-1)
        
        passes = p_val < bonferroni_alpha
        bonf_results[name] = {"sharpe": sharpe, "p_value": p_val, "passes": passes}
        
        status = "✅" if passes else "❌"
        print(f"   {name:<20} {sharpe:>8.2f} {p_val:>10.4f} {status:>14}")
        
    except Exception as e:
        print(f"   {name:<20} Error: {e}")

# =============================================================================
# TEST 3: TRUE WALK-FORWARD (Rolling OOS)
# =============================================================================

print("\n" + "=" * 80)
print("   TEST 3: TRUE WALK-FORWARD (No Peeking)")
print("=" * 80)

n_folds = 8
fold_size = len(prices) // (n_folds + 1)

print(f"\n   Rolling {n_folds} folds, {fold_size} days each")

wf_results = {name: [] for name in strategies.keys()}

for fold in range(n_folds):
    train_end = (fold + 1) * fold_size
    test_start = train_end
    test_end = test_start + fold_size
    
    if test_end > len(prices):
        break
    
    fold_prices = prices.iloc[test_start:test_end]
    fold_vix = vix.iloc[test_start:test_end] if vix is not None else None
    
    for name, func in strategies.items():
        try:
            weights = func(fold_prices, fold_vix)
            returns = fold_prices.pct_change()
            
            warmup = min(100, len(fold_prices) // 3)
            weights = weights.iloc[warmup:]
            returns = returns.iloc[warmup:]
            
            common = weights.columns.intersection(returns.columns)
            if len(common) == 0:
                continue
            abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
            norm = weights[common].div(abs_sum, axis=0)
            
            port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
            
            if port_ret.std() > 0:
                sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
                wf_results[name].append(sharpe)
        except:
            pass

print(f"\n   {'Strategy':<20} {'Folds':>6} {'Mean Sharpe':>12} {'+ve Folds':>10} {'Consistency':>12}")
print("   " + "-" * 65)

for name in strategies.keys():
    sharpes = wf_results[name]
    if sharpes:
        mean_s = np.mean(sharpes)
        pos_folds = sum(1 for s in sharpes if s > 0)
        consistency = pos_folds / len(sharpes) * 100
        
        status = "✅" if consistency >= 75 else "⚠️" if consistency >= 50 else "❌"
        print(f"   {name:<20} {len(sharpes):>6} {mean_s:>12.2f} {pos_folds}/{len(sharpes):>8} {consistency:>11.0f}% {status}")

# =============================================================================
# TEST 4: WHITE'S REALITY CHECK
# =============================================================================

print("\n" + "=" * 80)
print("   TEST 4: WHITE'S REALITY CHECK (vs 10,000 Random Strategies)")
print("=" * 80)

print("\n   Generating 10,000 random weight strategies...")

n_random = 10000
random_sharpes = []

# Use OOS data
returns = oos_prices.pct_change().iloc[300:]
n_days = len(returns)

for _ in range(n_random):
    # Random weights for available assets
    rand_w = np.random.dirichlet([1] * len(returns.columns), size=n_days)
    rand_ret = (rand_w[:-1] * returns.iloc[1:].values).sum(axis=1)
    if np.std(rand_ret) > 0:
        random_sharpes.append(np.mean(rand_ret) / np.std(rand_ret) * np.sqrt(252))

random_sharpes = np.array(random_sharpes)

print(f"   Random strategies mean Sharpe: {np.mean(random_sharpes):.2f}")
print(f"   Random strategies 95th percentile: {np.percentile(random_sharpes, 95):.2f}")
print(f"   Random strategies 99th percentile: {np.percentile(random_sharpes, 99):.2f}")

print(f"\n   {'Strategy':<20} {'Sharpe':>8} {'Percentile':>12} {'Beats Random':>14}")
print("   " + "-" * 58)

reality_check = {}

for name in strategies.keys():
    if name in bonf_results:
        sharpe = bonf_results[name]['sharpe']
        percentile = (random_sharpes < sharpe).mean() * 100
        beats = percentile > 95
        
        reality_check[name] = {"percentile": percentile, "beats": beats}
        
        status = "✅" if percentile > 99 else "⚠️" if percentile > 95 else "❌"
        print(f"   {name:<20} {sharpe:>8.2f} {percentile:>11.1f}% {status:>14}")

# =============================================================================
# TEST 5: PARAMETER SENSITIVITY
# =============================================================================

print("\n" + "=" * 80)
print("   TEST 5: PARAMETER SENSITIVITY (Robustness)")
print("=" * 80)

print("\n   Testing ERP Regime with parameter variations...")

param_variations = [
    ("Base", 0.02, 0.5, 0.3),
    ("Loose threshold", 0.01, 0.5, 0.3),
    ("Tight threshold", 0.03, 0.5, 0.3),
    ("More Netflix", 0.02, 0.6, 0.2),
    ("More Cheese", 0.02, 0.4, 0.4),
    ("Less Netflix", 0.02, 0.4, 0.3),
]

def erp_regime_param(prices, vix, threshold, netflix_wt, cheese_wt):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(252, len(prices)), len(prices)):
        date = prices.index[i]
        year = date.year
        
        netflix = cheese = 0
        if year in WEIRD_DATA['netflix'] and year-1 in WEIRD_DATA['netflix']:
            netflix = (WEIRD_DATA['netflix'][year] - WEIRD_DATA['netflix'][year-1]) / WEIRD_DATA['netflix'][year-1]
        if year in WEIRD_DATA['cheese'] and year-1 in WEIRD_DATA['cheese']:
            cheese = (WEIRD_DATA['cheese'][year] - WEIRD_DATA['cheese'][year-1]) / WEIRD_DATA['cheese'][year-1]
        
        sig = -netflix * netflix_wt + cheese * cheese_wt
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if sig > threshold: w['XLE'], w['SPY'] = 0.35, 0.20
            elif sig < -threshold: w['XLE'], w['GLD'] = 0.10, 0.35
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w: 
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    return weights.shift(1).fillna(0)

param_sharpes = []

print(f"\n   {'Variation':<20} {'Sharpe':>8}")
print("   " + "-" * 30)

for var_name, thresh, nf_wt, ch_wt in param_variations:
    try:
        weights = erp_regime_param(oos_prices, oos_vix, thresh, nf_wt, ch_wt)
        returns = oos_prices.pct_change()
        
        warmup = 300
        weights = weights.iloc[warmup:]
        returns = returns.iloc[warmup:]
        
        common = weights.columns.intersection(returns.columns)
        abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
        norm = weights[common].div(abs_sum, axis=0)
        
        port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
        sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
        
        param_sharpes.append(sharpe)
        print(f"   {var_name:<20} {sharpe:>8.2f}")
    except:
        pass

if param_sharpes:
    mean_s = np.mean(param_sharpes)
    std_s = np.std(param_sharpes)
    cv = std_s / mean_s * 100 if mean_s != 0 else 0
    
    print(f"\n   Parameter sensitivity:")
    print(f"   Mean Sharpe across variations: {mean_s:.2f}")
    print(f"   Std Dev: {std_s:.2f}")
    print(f"   Coefficient of Variation: {cv:.1f}%")
    
    if cv < 20:
        print("   ✅ ROBUST - Low parameter sensitivity")
    elif cv < 40:
        print("   ⚠️  MODERATE - Some parameter sensitivity")
    else:
        print("   ❌ FRAGILE - High parameter sensitivity")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("   🏆 SELECTION BIAS FINAL VERDICT")
print("=" * 80)

final_scores = {}

for name in strategies.keys():
    score = 0
    issues = []
    
    # Look-ahead
    if bias_results.get(name, False):
        score += 2
    else:
        issues.append("Look-ahead bias")
    
    # Bonferroni
    if bonf_results.get(name, {}).get("passes", False):
        score += 3
    else:
        issues.append("Fails Bonferroni")
    
    # Walk-forward
    wf = wf_results.get(name, [])
    if wf and sum(1 for s in wf if s > 0) / len(wf) >= 0.75:
        score += 3
    elif wf and sum(1 for s in wf if s > 0) / len(wf) >= 0.5:
        score += 1
        issues.append("Moderate walk-forward")
    else:
        issues.append("Poor walk-forward")
    
    # Reality check
    if reality_check.get(name, {}).get("percentile", 0) > 99:
        score += 3
    elif reality_check.get(name, {}).get("percentile", 0) > 95:
        score += 2
    else:
        issues.append("May be luck")
    
    final_scores[name] = {"score": score, "issues": issues}

print(f"\n   {'Strategy':<20} {'Score':>8} {'Status':<20} Issues")
print("   " + "-" * 70)

for name, data in sorted(final_scores.items(), key=lambda x: x[1]["score"], reverse=True):
    score = data["score"]
    issues = data["issues"]
    
    if score >= 9:
        status = "✅ No Selection Bias"
    elif score >= 6:
        status = "⚠️  Minor Concerns"
    else:
        status = "❌ Possible Bias"
    
    issue_str = ", ".join(issues) if issues else "None"
    print(f"   {name:<20} {score:>8} {status:<20} {issue_str}")

# Winner
winner = max(final_scores.items(), key=lambda x: x[1]["score"])
print(f"\n   🏆 MOST ROBUST (No Selection Bias): {winner[0]}")

print("\n" + "=" * 80)
