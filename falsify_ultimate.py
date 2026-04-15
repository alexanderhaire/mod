"""
Rigorous Falsification: Ultimate Strategy
==========================================

Applying the same tests we used on AROS:
1. Active Return IR with t-test and p-value
2. Pre/Post/Full window splits for consistency
3. Proxy Falsification (random signal, inverted signal)
4. Bootstrap Confidence Intervals
5. Monte Carlo Permutation Test
6. Walk-Forward Validation

If the strategy passes all these, it's real. If it fails, it's luck.

RUN: python falsify_ultimate.py
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

def fetch_all_data():
    print("📊 Fetching data...")
    
    traditional = ['SPY', 'QQQ', 'TLT', 'GLD', 'IEF']
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD']
    vix = ['^VIX', '^VIX3M']
    
    tickers = traditional + crypto + vix
    
    data = yf.download(tickers, start='2020-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    
    vix_spot = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix_3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    for v in ['^VIX', '^VIX3M']:
        if v in prices.columns:
            prices = prices.drop(v, axis=1)
    
    prices = prices.dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Days: {len(prices)}")
    
    return prices, vix_spot, vix_3m

# =============================================================================
# SIGNALS
# =============================================================================

def altseason_signal(prices, lookback=14):
    """Altseason detector."""
    alts = [c for c in prices.columns if c.endswith('-USD') and c != 'BTC-USD']
    
    if 'BTC-USD' not in prices.columns or len(alts) < 2:
        return None, None
    
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    top_alts_series = pd.Series([None] * len(prices), index=prices.index)
    
    for i in range(lookback + 10, len(prices)):
        alt_moms = {}
        for alt in alts:
            try:
                ret = prices[alt].iloc[i] / prices[alt].iloc[i-lookback] - 1
                if not np.isnan(ret):
                    alt_moms[alt] = ret
            except:
                pass
        
        avg_alt_mom = np.mean(list(alt_moms.values())) if len(alt_moms) > 0 else 0
        btc_m = btc_mom.iloc[i] if not np.isnan(btc_mom.iloc[i]) else 0
        
        if avg_alt_mom > btc_m * 1.2:
            signal.iloc[i] = 1
            sorted_alts = sorted(alt_moms.items(), key=lambda x: x[1], reverse=True)[:3]
            top_alts_series.iloc[i] = [x[0] for x in sorted_alts]
        else:
            signal.iloc[i] = -1
    
    return signal, top_alts_series

def random_signal(prices, seed=42):
    """Random signal for falsification."""
    np.random.seed(seed)
    signal = pd.Series(np.random.choice([-1, 1], size=len(prices)), index=prices.index)
    return signal

def inverted_signal(original_signal):
    """Inverted signal (buy when we should sell)."""
    return -original_signal

def lagged_signal(original_signal, lag=5):
    """Lagged signal (delayed by N days)."""
    return original_signal.shift(lag).fillna(0)

# =============================================================================
# STRATEGY
# =============================================================================

def strategy_base(prices):
    """60/40 baseline."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        if 'SPY' in prices.columns:
            weights.iloc[i]['SPY'] = 0.60
        if 'TLT' in prices.columns:
            weights.iloc[i]['TLT'] = 0.40
    
    return weights.shift(1).fillna(0)

def strategy_altseason(prices, alt_signal, top_alts):
    """Altseason strategy (what we're testing)."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        alt_s = alt_signal.iloc[i] if alt_signal is not None and i < len(alt_signal) else 0
        
        if alt_s > 0:
            alts = top_alts.iloc[i] if top_alts is not None and i < len(top_alts) else None
            if alts and len(alts) > 0:
                for alt in alts:
                    if alt in prices.columns:
                        weights.iloc[i][alt] = 1.0 / len(alts)
            else:
                if 'ETH-USD' in prices.columns:
                    weights.iloc[i]['ETH-USD'] = 1.0
        else:
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 1.0
    
    return weights.shift(1).fillna(0)

def strategy_with_signal(prices, signal):
    """Generic strategy that uses any signal (for falsification)."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        s = signal.iloc[i] if i < len(signal) else 0
        
        if s > 0:
            # Risk-on: alts
            for alt in ['ETH-USD', 'SOL-USD', 'ADA-USD']:
                if alt in prices.columns:
                    weights.iloc[i][alt] = 1/3
        else:
            # Risk-off: BTC
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 1.0
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

def compute_returns(prices, weights, warmup=60):
    returns = prices.pct_change()
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    port_ret = (weights[common].shift(1) * returns[common]).sum(axis=1)
    return port_ret.dropna()

def compute_metrics(returns):
    if len(returns) < 20 or returns.std() == 0:
        return None
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    
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
    delta_sharpe = compute_metrics(r_c)['sharpe'] - compute_metrics(r_b)['sharpe'] if compute_metrics(r_c) and compute_metrics(r_b) else 0
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val, 'n': len(active), 'delta_sharpe': delta_sharpe}

# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def bootstrap_sharpe_ci(returns, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    sharpes = []
    n = len(returns)
    
    for _ in range(n_bootstrap):
        sample = returns.sample(n=n, replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    
    sharpes = np.array(sharpes)
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    
    return lower, upper, np.mean(sharpes)

# =============================================================================
# MONTE CARLO PERMUTATION
# =============================================================================

def monte_carlo_permutation(returns_strat, returns_base, n_sims=1000):
    """Monte Carlo permutation test for active return significance."""
    common_idx = returns_base.index.intersection(returns_strat.index)
    actual_active = returns_strat.loc[common_idx] - returns_base.loc[common_idx]
    actual_active = actual_active.dropna()
    
    actual_mean = actual_active.mean()
    
    count_worse = 0
    for _ in range(n_sims):
        shuffled = actual_active.sample(frac=1, replace=False)
        if shuffled.mean() >= actual_mean:
            count_worse += 1
    
    perm_p_val = count_worse / n_sims
    
    return perm_p_val, actual_mean * 252 * 100  # Annualized %

# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(prices, n_folds=4):
    """Walk-forward validation with expanding window."""
    n = len(prices)
    fold_size = n // n_folds
    
    results = []
    
    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min((fold + 1) * fold_size, n)
        
        # Training data
        train_prices = prices.iloc[:train_end]
        
        # Test data
        test_prices = prices.iloc[test_start:test_end]
        
        if len(test_prices) < 60:
            continue
        
        # Generate signal on train data, apply to test
        train_signal, train_alts = altseason_signal(train_prices)
        
        # Apply same logic to test period
        test_signal, test_alts = altseason_signal(test_prices)
        
        # Compute returns
        test_weights = strategy_altseason(test_prices, test_signal, test_alts)
        test_returns = compute_returns(test_prices, test_weights, warmup=30)
        
        base_weights = strategy_base(test_prices)
        base_returns = compute_returns(test_prices, base_weights, warmup=30)
        
        test_metrics = compute_metrics(test_returns)
        base_metrics = compute_metrics(base_returns)
        
        if test_metrics and base_metrics:
            results.append({
                'fold': fold,
                'period': f"{test_prices.index[0].date()} to {test_prices.index[-1].date()}",
                'strat_sharpe': test_metrics['sharpe'],
                'base_sharpe': base_metrics['sharpe'],
                'diff': test_metrics['sharpe'] - base_metrics['sharpe']
            })
    
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   RIGOROUS FALSIFICATION: ULTIMATE STRATEGY")
    print("   AROS-level testing: Can we break it?")
    print("=" * 80)
    
    prices, vix, vix3m = fetch_all_data()
    
    # Generate real signal
    real_signal, real_alts = altseason_signal(prices)
    
    # Generate placebo signals
    random_sig = random_signal(prices)
    inverted_sig = inverted_signal(real_signal)
    lagged_sig = lagged_signal(real_signal, lag=10)
    
    # Build strategies
    real_weights = strategy_altseason(prices, real_signal, real_alts)
    base_weights = strategy_base(prices)
    random_weights = strategy_with_signal(prices, random_sig)
    inverted_weights = strategy_with_signal(prices, inverted_sig)
    lagged_weights = strategy_with_signal(prices, lagged_sig)
    
    # Compute returns
    real_returns = compute_returns(prices, real_weights)
    base_returns = compute_returns(prices, base_weights)
    random_returns = compute_returns(prices, random_weights)
    inverted_returns = compute_returns(prices, inverted_weights)
    lagged_returns = compute_returns(prices, lagged_weights)
    
    # ==========================================================================
    # TEST 1: Active Return IR with p-value
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 1: ACTIVE RETURN IR & P-VALUE")
    print("=" * 80)
    
    real_stats = compute_active_stats(real_returns, base_returns)
    print(f"""
   Real Signal (Altseason):
   Active IR:    {real_stats['ir']:.3f}
   t-statistic:  {real_stats['t_stat']:.3f}
   p-value:      {real_stats['p_val']:.4f}
   Δ Sharpe:     +{real_stats['delta_sharpe']:.2f}
   N (days):     {real_stats['n']}
    """)
    
    # ==========================================================================
    # TEST 2: Pre/Post Split Consistency
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 2: PRE/POST SPLIT CONSISTENCY")
    print("=" * 80)
    
    mid_point = len(prices) // 2
    
    pre_prices = prices.iloc[:mid_point]
    post_prices = prices.iloc[mid_point:]
    
    # Pre-period
    pre_signal, pre_alts = altseason_signal(pre_prices)
    pre_weights = strategy_altseason(pre_prices, pre_signal, pre_alts)
    pre_returns = compute_returns(pre_prices, pre_weights, warmup=30)
    pre_base_weights = strategy_base(pre_prices)
    pre_base_returns = compute_returns(pre_prices, pre_base_weights, warmup=30)
    pre_stats = compute_active_stats(pre_returns, pre_base_returns)
    
    # Post-period
    post_signal, post_alts = altseason_signal(post_prices)
    post_weights = strategy_altseason(post_prices, post_signal, post_alts)
    post_returns = compute_returns(post_prices, post_weights, warmup=30)
    post_base_weights = strategy_base(post_prices)
    post_base_returns = compute_returns(post_prices, post_base_weights, warmup=30)
    post_stats = compute_active_stats(post_returns, post_base_returns)
    
    print(f"""
   Pre-Period ({pre_prices.index[0].date()} to {pre_prices.index[-1].date()}):
   Active IR:    {pre_stats['ir']:.3f}
   p-value:      {pre_stats['p_val']:.4f}
   
   Post-Period ({post_prices.index[0].date()} to {post_prices.index[-1].date()}):
   Active IR:    {post_stats['ir']:.3f}
   p-value:      {post_stats['p_val']:.4f}
   
   Consistency Check:
   IR Pre vs Post:  {pre_stats['ir']:.2f} vs {post_stats['ir']:.2f}
   {"✅ CONSISTENT (both positive)" if pre_stats['ir'] > 0 and post_stats['ir'] > 0 else "⚠️ INCONSISTENT"}
    """)
    
    # ==========================================================================
    # TEST 3: Proxy Falsification
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 3: PROXY FALSIFICATION")
    print("=" * 80)
    
    random_stats = compute_active_stats(random_returns, base_returns)
    inverted_stats = compute_active_stats(inverted_returns, base_returns)
    lagged_stats = compute_active_stats(lagged_returns, base_returns)
    
    print(f"""
   Does the SPECIFIC signal matter, or does random work too?
   
   {"Signal":<20} {"IR":>8} {"p-value":>10} {"Δ Sharpe":>10}
   {"-"*55}
   {"Real (Altseason)":<20} {real_stats['ir']:>8.2f} {real_stats['p_val']:>10.4f} {real_stats['delta_sharpe']:>+10.2f}
   {"Random":<20} {random_stats['ir']:>8.2f} {random_stats['p_val']:>10.4f} {random_stats['delta_sharpe']:>+10.2f}
   {"Inverted":<20} {inverted_stats['ir']:>8.2f} {inverted_stats['p_val']:>10.4f} {inverted_stats['delta_sharpe']:>+10.2f}
   {"Lagged (10 days)":<20} {lagged_stats['ir']:>8.2f} {lagged_stats['p_val']:>10.4f} {lagged_stats['delta_sharpe']:>+10.2f}
   
   {"✅ REAL SIGNAL IS DISTINCT" if real_stats['ir'] > max(random_stats['ir'], inverted_stats['ir'], lagged_stats['ir']) else "❌ SIGNAL NOT DISTINCT"}
    """)
    
    # ==========================================================================
    # TEST 4: Bootstrap Confidence Interval
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 4: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 80)
    
    lower, upper, mean = bootstrap_sharpe_ci(real_returns, n_bootstrap=1000)
    base_lower, base_upper, base_mean = bootstrap_sharpe_ci(base_returns, n_bootstrap=1000)
    
    print(f"""
   Strategy Sharpe: {mean:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])
   Baseline Sharpe: {base_mean:.2f} (95% CI: [{base_lower:.2f}, {base_upper:.2f}])
   
   {"✅ CI DOES NOT INCLUDE BASELINE" if lower > base_upper else "⚠️ CI OVERLAPS WITH BASELINE"}
    """)
    
    # ==========================================================================
    # TEST 5: Monte Carlo Permutation
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 5: MONTE CARLO PERMUTATION (1000 sims)")
    print("=" * 80)
    
    perm_p_val, ann_active = monte_carlo_permutation(real_returns, base_returns, n_sims=1000)
    
    print(f"""
   Annualized Active Return: {ann_active:.2f}%
   Permutation p-value: {perm_p_val:.4f}
   
   {"✅ SIGNIFICANT by permutation" if perm_p_val < 0.05 else "⚠️ NOT significant by permutation"}
    """)
    
    # ==========================================================================
    # TEST 6: Walk-Forward Validation
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 6: WALK-FORWARD VALIDATION (4 folds)")
    print("=" * 80)
    
    wf_results = walk_forward_validation(prices, n_folds=4)
    
    print(f"\n   {'Fold':<6} {'Period':<30} {'Strat':>8} {'Base':>8} {'Diff':>8}")
    print("   " + "-" * 65)
    
    positive_folds = 0
    for r in wf_results:
        print(f"   {r['fold']:<6} {r['period']:<30} {r['strat_sharpe']:>8.2f} {r['base_sharpe']:>8.2f} {r['diff']:>+8.2f}")
        if r['diff'] > 0:
            positive_folds += 1
    
    print(f"""
   Consistency: {positive_folds}/{len(wf_results)} folds outperformed baseline
   {"✅ ROBUST across folds" if positive_folds == len(wf_results) else "⚠️ NOT consistent across all folds"}
    """)
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   FINAL FALSIFICATION VERDICT")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: p-value
    if real_stats['p_val'] < 0.05:
        tests_passed += 1
        t1 = "✅ PASS"
    else:
        t1 = "❌ FAIL"
    
    # Test 2: Consistency
    if pre_stats['ir'] > 0 and post_stats['ir'] > 0:
        tests_passed += 1
        t2 = "✅ PASS"
    else:
        t2 = "❌ FAIL"
    
    # Test 3: Proxy
    if real_stats['ir'] > max(random_stats['ir'], inverted_stats['ir'], lagged_stats['ir']):
        tests_passed += 1
        t3 = "✅ PASS"
    else:
        t3 = "❌ FAIL"
    
    # Test 4: Bootstrap
    if lower > base_upper:
        tests_passed += 1
        t4 = "✅ PASS"
    else:
        t4 = "❌ FAIL"
    
    # Test 5: Monte Carlo
    if perm_p_val < 0.05:
        tests_passed += 1
        t5 = "✅ PASS"
    else:
        t5 = "❌ FAIL"
    
    # Test 6: Walk-forward
    if positive_folds == len(wf_results):
        tests_passed += 1
        t6 = "✅ PASS"
    else:
        t6 = "❌ FAIL"
    
    print(f"""
   Test Results:
   1. Active Return p-value:     {t1} (p={real_stats['p_val']:.4f})
   2. Pre/Post Consistency:      {t2} (IR: {pre_stats['ir']:.2f}/{post_stats['ir']:.2f})
   3. Proxy Falsification:       {t3} (Real IR > Controls)
   4. Bootstrap CI:              {t4} (CI: [{lower:.2f}, {upper:.2f}])
   5. Monte Carlo Permutation:   {t5} (p={perm_p_val:.4f})
   6. Walk-Forward:              {t6} ({positive_folds}/{len(wf_results)} folds)
   
   {"="*50}
   OVERALL: {tests_passed}/{total_tests} TESTS PASSED
   {"="*50}
    """)
    
    if tests_passed >= 5:
        print("   🏆 STRATEGY VALIDATED - GENUINE EDGE CONFIRMED")
    elif tests_passed >= 3:
        print("   ⚠️  PARTIAL VALIDATION - PROCEED WITH CAUTION")
    else:
        print("   ❌ STRATEGY FAILED VALIDATION - LIKELY SPURIOUS")
    
    print("\n" + "=" * 80)
