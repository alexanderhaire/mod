"""
Rigorous Falsification: ULTIMATE COMBINED Strategy
===================================================

Testing the full 60% Traditional + 40% Crypto strategy
that achieved Sharpe 1.42 and p=0.000

Same 6 tests as AROS falsification.

RUN: python falsify_ultimate_combined.py
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

def vix_term_structure_signal(vix, vix3m):
    if vix is None or vix3m is None:
        return None
    
    ratio = vix / vix3m
    ratio_smooth = ratio.rolling(5).mean()
    
    signal = pd.Series(0.0, index=vix.index)
    
    for i in range(60, len(vix)):
        if ratio_smooth.iloc[i] < 0.90:
            signal.iloc[i] = 1
        elif ratio_smooth.iloc[i] > 1.05:
            signal.iloc[i] = -1
    
    return signal

def altseason_signal(prices, lookback=14):
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

def crypto_momentum_signal(prices, lookback=30):
    if 'BTC-USD' not in prices.columns:
        return None
    
    btc = prices['BTC-USD']
    btc_ret = btc.pct_change(lookback)
    btc_ma = btc.rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if btc.iloc[i] > btc_ma.iloc[i] and btc_ret.iloc[i] > 0:
            signal.iloc[i] = 1
        elif btc.iloc[i] < btc_ma.iloc[i] and btc_ret.iloc[i] < 0:
            signal.iloc[i] = -1
    
    return signal

# =============================================================================
# STRATEGIES
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

def strategy_ultimate_combined(prices, vix_signal, altseason_sig, top_alts, crypto_mom):
    """
    ULTIMATE COMBINED STRATEGY:
    - 60% traditional (SPY/TLT/GLD) - timed by VIX
    - 40% crypto - timed by Altseason + Crypto Momentum
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        # === TRADITIONAL SLEEVE (60%) ===
        vix_s = vix_signal.iloc[i] if vix_signal is not None and i < len(vix_signal) else 0
        
        if vix_s > 0:
            trad_w = {'SPY': 0.45, 'TLT': 0.10, 'GLD': 0.05}
        elif vix_s < 0:
            trad_w = {'SPY': 0.15, 'TLT': 0.35, 'GLD': 0.10}
        else:
            trad_w = {'SPY': 0.30, 'TLT': 0.22, 'GLD': 0.08}
        
        for asset, w in trad_w.items():
            if asset in prices.columns:
                weights.iloc[i][asset] = w
        
        # === CRYPTO SLEEVE (40%) ===
        alt_s = altseason_sig.iloc[i] if altseason_sig is not None and i < len(altseason_sig) else 0
        crypto_s = crypto_mom.iloc[i] if crypto_mom is not None and i < len(crypto_mom) else 0
        
        crypto_combined = (alt_s + crypto_s) / 2
        
        if crypto_combined > 0.5:
            alts = top_alts.iloc[i] if top_alts is not None and i < len(top_alts) else None
            if alts and len(alts) > 0:
                for alt in alts:
                    if alt in prices.columns:
                        weights.iloc[i][alt] = 0.30 / len(alts)
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 0.10
        elif crypto_combined > 0:
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 0.25
            if 'ETH-USD' in prices.columns:
                weights.iloc[i]['ETH-USD'] = 0.15
        elif crypto_combined > -0.5:
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 0.20
        else:
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 0.05
            if 'TLT' in prices.columns:
                weights.iloc[i]['TLT'] += 0.15
    
    return weights.shift(1).fillna(0)

def strategy_random_combined(prices, seed=42):
    """Random version for falsification."""
    np.random.seed(seed)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        r = np.random.random()
        if r > 0.5:
            weights.iloc[i]['SPY'] = 0.45
            weights.iloc[i]['TLT'] = 0.10
            weights.iloc[i]['BTC-USD'] = 0.30 if 'BTC-USD' in prices.columns else 0
        else:
            weights.iloc[i]['SPY'] = 0.20
            weights.iloc[i]['TLT'] = 0.40
            weights.iloc[i]['BTC-USD'] = 0.10 if 'BTC-USD' in prices.columns else 0
    
    return weights.shift(1).fillna(0)

def strategy_no_crypto(prices, vix_signal):
    """Traditional only (no crypto) - to isolate crypto contribution."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        vix_s = vix_signal.iloc[i] if vix_signal is not None and i < len(vix_signal) else 0
        
        if vix_s > 0:
            weights.iloc[i]['SPY'] = 0.70
            weights.iloc[i]['TLT'] = 0.20
            weights.iloc[i]['GLD'] = 0.10
        elif vix_s < 0:
            weights.iloc[i]['SPY'] = 0.30
            weights.iloc[i]['TLT'] = 0.50
            weights.iloc[i]['GLD'] = 0.20
        else:
            weights.iloc[i]['SPY'] = 0.50
            weights.iloc[i]['TLT'] = 0.35
            weights.iloc[i]['GLD'] = 0.15
    
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
    
    m_c = compute_metrics(r_c)
    m_b = compute_metrics(r_b)
    delta_sharpe = m_c['sharpe'] - m_b['sharpe'] if m_c and m_b else 0
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val, 'n': len(active), 'delta_sharpe': delta_sharpe}

def bootstrap_sharpe_ci(returns, n_bootstrap=1000, ci=0.95):
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

def monte_carlo_permutation(returns_strat, returns_base, n_sims=1000):
    common_idx = returns_base.index.intersection(returns_strat.index)
    strat = returns_strat.loc[common_idx].values
    base = returns_base.loc[common_idx].values
    
    actual_diff = np.mean(strat) - np.mean(base)
    
    combined = np.concatenate([strat, base])
    n_strat = len(strat)
    
    count_greater = 0
    for _ in range(n_sims):
        np.random.shuffle(combined)
        perm_strat = combined[:n_strat]
        perm_base = combined[n_strat:]
        perm_diff = np.mean(perm_strat) - np.mean(perm_base)
        if perm_diff >= actual_diff:
            count_greater += 1
    
    return count_greater / n_sims, actual_diff * 252 * 100

def walk_forward_validation(prices, vix, vix3m, n_folds=4):
    n = len(prices)
    fold_size = n // n_folds
    
    results = []
    
    for fold in range(1, n_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        
        test_prices = prices.iloc[test_start:test_end]
        test_vix = vix.iloc[test_start:test_end] if vix is not None else None
        test_vix3m = vix3m.iloc[test_start:test_end] if vix3m is not None else None
        
        if len(test_prices) < 60:
            continue
        
        # Generate signals for test period
        vix_sig = vix_term_structure_signal(test_vix, test_vix3m)
        alt_sig, top_alts = altseason_signal(test_prices)
        crypto_mom = crypto_momentum_signal(test_prices)
        
        # Build strategies
        test_weights = strategy_ultimate_combined(test_prices, vix_sig, alt_sig, top_alts, crypto_mom)
        base_weights = strategy_base(test_prices)
        
        test_returns = compute_returns(test_prices, test_weights, warmup=30)
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
    print("   RIGOROUS FALSIFICATION: ULTIMATE COMBINED STRATEGY")
    print("   60% Traditional + 40% Crypto")
    print("=" * 80)
    
    prices, vix, vix3m = fetch_all_data()
    
    # Generate signals
    vix_signal = vix_term_structure_signal(vix, vix3m)
    altseason_sig, top_alts = altseason_signal(prices)
    crypto_mom = crypto_momentum_signal(prices)
    
    # Build strategies
    ultimate_weights = strategy_ultimate_combined(prices, vix_signal, altseason_sig, top_alts, crypto_mom)
    base_weights = strategy_base(prices)
    random_weights = strategy_random_combined(prices)
    no_crypto_weights = strategy_no_crypto(prices, vix_signal)
    
    # Compute returns
    ultimate_returns = compute_returns(prices, ultimate_weights)
    base_returns = compute_returns(prices, base_weights)
    random_returns = compute_returns(prices, random_weights)
    no_crypto_returns = compute_returns(prices, no_crypto_weights)
    
    # ==========================================================================
    # TEST 1: Active Return IR
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 1: ACTIVE RETURN IR & P-VALUE")
    print("=" * 80)
    
    real_stats = compute_active_stats(ultimate_returns, base_returns)
    print(f"""
   Ultimate Combined vs 60/40:
   Active IR:    {real_stats['ir']:.3f}
   t-statistic:  {real_stats['t_stat']:.3f}
   p-value:      {real_stats['p_val']:.6f}
   Δ Sharpe:     +{real_stats['delta_sharpe']:.2f}
   N (days):     {real_stats['n']}
    """)
    
    # ==========================================================================
    # TEST 2: Pre/Post Split
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 2: PRE/POST SPLIT CONSISTENCY")
    print("=" * 80)
    
    mid = len(prices) // 2
    
    pre_prices = prices.iloc[:mid]
    post_prices = prices.iloc[mid:]
    pre_vix = vix.iloc[:mid] if vix is not None else None
    post_vix = vix.iloc[mid:] if vix is not None else None
    pre_vix3m = vix3m.iloc[:mid] if vix3m is not None else None
    post_vix3m = vix3m.iloc[mid:] if vix3m is not None else None
    
    # Pre
    pre_vix_sig = vix_term_structure_signal(pre_vix, pre_vix3m)
    pre_alt_sig, pre_top_alts = altseason_signal(pre_prices)
    pre_crypto_mom = crypto_momentum_signal(pre_prices)
    pre_ult_w = strategy_ultimate_combined(pre_prices, pre_vix_sig, pre_alt_sig, pre_top_alts, pre_crypto_mom)
    pre_base_w = strategy_base(pre_prices)
    pre_ult_ret = compute_returns(pre_prices, pre_ult_w, warmup=30)
    pre_base_ret = compute_returns(pre_prices, pre_base_w, warmup=30)
    pre_stats = compute_active_stats(pre_ult_ret, pre_base_ret)
    
    # Post
    post_vix_sig = vix_term_structure_signal(post_vix, post_vix3m)
    post_alt_sig, post_top_alts = altseason_signal(post_prices)
    post_crypto_mom = crypto_momentum_signal(post_prices)
    post_ult_w = strategy_ultimate_combined(post_prices, post_vix_sig, post_alt_sig, post_top_alts, post_crypto_mom)
    post_base_w = strategy_base(post_prices)
    post_ult_ret = compute_returns(post_prices, post_ult_w, warmup=30)
    post_base_ret = compute_returns(post_prices, post_base_w, warmup=30)
    post_stats = compute_active_stats(post_ult_ret, post_base_ret)
    
    print(f"""
   Pre-Period ({pre_prices.index[0].date()} to {pre_prices.index[-1].date()}):
   Active IR: {pre_stats['ir']:.3f}, p-value: {pre_stats['p_val']:.4f}
   
   Post-Period ({post_prices.index[0].date()} to {post_prices.index[-1].date()}):
   Active IR: {post_stats['ir']:.3f}, p-value: {post_stats['p_val']:.4f}
   
   {"✅ CONSISTENT (both positive IR)" if pre_stats['ir'] > 0 and post_stats['ir'] > 0 else "⚠️ INCONSISTENT"}
    """)
    
    # ==========================================================================
    # TEST 3: Proxy Falsification
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 3: PROXY FALSIFICATION")
    print("=" * 80)
    
    random_stats = compute_active_stats(random_returns, base_returns)
    no_crypto_stats = compute_active_stats(no_crypto_returns, base_returns)
    
    print(f"""
   {"Strategy":<25} {"IR":>8} {"p-value":>10} {"Δ Sharpe":>10}
   {"-"*55}
   {"Ultimate Combined":<25} {real_stats['ir']:>8.2f} {real_stats['p_val']:>10.6f} {real_stats['delta_sharpe']:>+10.2f}
   {"Random Allocation":<25} {random_stats['ir']:>8.2f} {random_stats['p_val']:>10.4f} {random_stats['delta_sharpe']:>+10.2f}
   {"No Crypto (VIX only)":<25} {no_crypto_stats['ir']:>8.2f} {no_crypto_stats['p_val']:>10.4f} {no_crypto_stats['delta_sharpe']:>+10.2f}
   
   {"✅ REAL STRATEGY IS DISTINCT" if real_stats['ir'] > max(random_stats['ir'], no_crypto_stats['ir']) else "⚠️ NOT DISTINCT"}
    """)
    
    # ==========================================================================
    # TEST 4: Bootstrap CI
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 4: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 80)
    
    lower, upper, mean = bootstrap_sharpe_ci(ultimate_returns, n_bootstrap=1000)
    base_lower, base_upper, base_mean = bootstrap_sharpe_ci(base_returns, n_bootstrap=1000)
    
    print(f"""
   Ultimate Sharpe: {mean:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])
   Baseline Sharpe: {base_mean:.2f} (95% CI: [{base_lower:.2f}, {base_upper:.2f}])
   
   {"✅ CIs DO NOT OVERLAP" if lower > base_upper else "⚠️ CIs OVERLAP"}
    """)
    
    # ==========================================================================
    # TEST 5: Monte Carlo
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 5: MONTE CARLO PERMUTATION (1000 sims)")
    print("=" * 80)
    
    perm_p_val, ann_active = monte_carlo_permutation(ultimate_returns, base_returns, n_sims=1000)
    
    print(f"""
   Annualized Active Return: {ann_active:.2f}%
   Permutation p-value: {perm_p_val:.4f}
   
   {"✅ SIGNIFICANT" if perm_p_val < 0.05 else "⚠️ NOT SIGNIFICANT"}
    """)
    
    # ==========================================================================
    # TEST 6: Walk-Forward
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   TEST 6: WALK-FORWARD VALIDATION (4 folds)")
    print("=" * 80)
    
    wf_results = walk_forward_validation(prices, vix, vix3m, n_folds=4)
    
    print(f"\n   {'Fold':<6} {'Period':<30} {'Ultimate':>10} {'Base':>10} {'Diff':>8}")
    print("   " + "-" * 70)
    
    positive_folds = 0
    for r in wf_results:
        print(f"   {r['fold']:<6} {r['period']:<30} {r['strat_sharpe']:>10.2f} {r['base_sharpe']:>10.2f} {r['diff']:>+8.2f}")
        if r['diff'] > 0:
            positive_folds += 1
    
    print(f"\n   Consistency: {positive_folds}/{len(wf_results)} folds outperformed")
    print(f"   {'✅ ROBUST' if positive_folds == len(wf_results) else '⚠️ NOT CONSISTENT'}")
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("   FINAL FALSIFICATION VERDICT: ULTIMATE COMBINED")
    print("=" * 80)
    
    tests_passed = 0
    
    t1 = "✅ PASS" if real_stats['p_val'] < 0.05 else "❌ FAIL"
    if real_stats['p_val'] < 0.05: tests_passed += 1
    
    t2 = "✅ PASS" if pre_stats['ir'] > 0 and post_stats['ir'] > 0 else "❌ FAIL"
    if pre_stats['ir'] > 0 and post_stats['ir'] > 0: tests_passed += 1
    
    t3 = "✅ PASS" if real_stats['ir'] > max(random_stats['ir'], no_crypto_stats['ir']) else "❌ FAIL"
    if real_stats['ir'] > max(random_stats['ir'], no_crypto_stats['ir']): tests_passed += 1
    
    t4 = "✅ PASS" if lower > base_upper else "❌ FAIL"
    if lower > base_upper: tests_passed += 1
    
    t5 = "✅ PASS" if perm_p_val < 0.05 else "❌ FAIL"
    if perm_p_val < 0.05: tests_passed += 1
    
    t6 = "✅ PASS" if positive_folds == len(wf_results) else "❌ FAIL"
    if positive_folds == len(wf_results): tests_passed += 1
    
    print(f"""
   Test Results:
   1. Active Return p-value:     {t1} (p={real_stats['p_val']:.6f})
   2. Pre/Post Consistency:      {t2} (IR: {pre_stats['ir']:.2f}/{post_stats['ir']:.2f})
   3. Proxy Falsification:       {t3}
   4. Bootstrap CI:              {t4}
   5. Monte Carlo Permutation:   {t5} (p={perm_p_val:.4f})
   6. Walk-Forward:              {t6} ({positive_folds}/{len(wf_results)} folds)
   
   {"="*55}
   OVERALL: {tests_passed}/6 TESTS PASSED
   {"="*55}
    """)
    
    if tests_passed >= 5:
        print("   🏆 STRATEGY VALIDATED - GENUINE EDGE CONFIRMED")
    elif tests_passed >= 3:
        print("   ⚠️  PARTIAL VALIDATION - PROCEED WITH CAUTION")
    else:
        print("   ❌ STRATEGY FAILED VALIDATION")
    
    print("\n" + "=" * 80)
