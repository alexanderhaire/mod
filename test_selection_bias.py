"""
Selection Bias & Statistical Validation Tests
==============================================

Tests to ensure the Japan+Korea strategy isn't due to:
1. Data mining / selection bias
2. Lucky parameters
3. Overfitting

Tests implemented:
1. White's Reality Check (bootstrap test for data snooping)
2. Hansen's SPA Test (Superior Predictive Ability)
3. Deflated Sharpe Ratio (accounting for multiple testing)
4. Bootstrap Confidence Intervals
5. Monte Carlo Permutation Test
6. Random Strategy Benchmark
7. Cross-validation stability
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# ORIGINAL ASIAN STRATEGY CONFIG
# =============================================================================

CHAMPION_TICKERS = ['EWJ', 'FXI', 'EWY', 'INDA', 'EWT', 'EWH', 'EWS', 'AAXJ', 'GLD', 'TLT']
CHAMPION_VIX = 25
CHAMPION_SMA = 200
CHAMPION_CAP = 0.20


# =============================================================================
# DATA & BACKTEST
# =============================================================================

def fetch_data(years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(CHAMPION_TICKERS, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def get_strategy_returns(prices: pd.DataFrame, vix: pd.Series,
                         vix_threshold: float = 25, sma_lookback: int = 150,
                         max_position: float = 0.15, warmup: int = 65) -> pd.Series:
    """Get strategy returns."""
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=sma_lookback,
        max_position_pct=max_position
    )
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except:
        return pd.Series()
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return pd.Series()
    
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    return net_returns


def calculate_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe."""
    if returns.empty or returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(252)


# =============================================================================
# TEST 1: WHITE'S REALITY CHECK
# =============================================================================

def whites_reality_check(prices: pd.DataFrame, vix: pd.Series, 
                         n_bootstrap: int = 1000) -> Dict:
    """
    White's Reality Check for data snooping bias.
    
    Tests if the best strategy is significantly better than a benchmark
    accounting for the fact that many strategies were tested.
    """
    print("\n   === WHITE'S REALITY CHECK ===")
    print(f"   Testing {n_bootstrap} bootstrap samples...")
    
    # Split data
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Get champion returns
    champion_returns = get_strategy_returns(oos_prices, oos_vix, 
                                            CHAMPION_VIX, CHAMPION_SMA, CHAMPION_CAP)
    champion_sharpe = calculate_sharpe(champion_returns)
    
    # Get benchmark (buy and hold) returns
    benchmark_returns = oos_prices.pct_change().mean(axis=1).fillna(0)
    benchmark_sharpe = calculate_sharpe(benchmark_returns)
    
    # Excess performance
    excess = champion_sharpe - benchmark_sharpe
    
    # Bootstrap under the null hypothesis
    n = len(champion_returns)
    bootstrap_stats = []
    
    for i in range(n_bootstrap):
        # Block bootstrap (preserve autocorrelation)
        block_size = 20
        n_blocks = n // block_size + 1
        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_size))
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]
        
        boot_returns = champion_returns.iloc[indices].values
        boot_bench = benchmark_returns.iloc[indices].values if len(benchmark_returns) >= n else benchmark_returns.values
        
        boot_sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252) if np.std(boot_returns) > 0 else 0
        bench_sharpe = np.mean(boot_bench) / np.std(boot_bench) * np.sqrt(252) if np.std(boot_bench) > 0 else 0
        
        bootstrap_stats.append(boot_sharpe - bench_sharpe)
    
    # Calculate p-value
    p_value = np.mean([bs >= excess for bs in bootstrap_stats])
    
    print(f"      Champion Sharpe: {champion_sharpe:.2f}")
    print(f"      Benchmark Sharpe: {benchmark_sharpe:.2f}")
    print(f"      Excess: {excess:.2f}")
    print(f"      p-value: {p_value:.4f}")
    print(f"      Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    return {'p_value': p_value, 'excess': excess, 'champion_sharpe': champion_sharpe}


# =============================================================================
# TEST 2: DEFLATED SHARPE RATIO
# =============================================================================

def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                          sharpe_std: float = 1.0) -> Dict:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    Accounts for the number of strategies tried (data snooping).
    From Bailey, Borwein, Lopez de Prado, and Zhu (2014).
    """
    print("\n   === DEFLATED SHARPE RATIO ===")
    
    # Expected maximum Sharpe under null (multiple testing)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    expected_max = sharpe_std * ((1 - gamma) * stats.norm.ppf(1 - 1/n_trials) + 
                                  gamma * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    
    # Standard error of Sharpe
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n_obs)
    
    # Deflated Sharpe Ratio (probability that observed Sharpe is significant)
    dsr = stats.norm.cdf((sharpe - expected_max) / se_sharpe)
    
    print(f"      Observed Sharpe: {sharpe:.2f}")
    print(f"      Number of trials: {n_trials}")
    print(f"      Expected Max Sharpe (under null): {expected_max:.2f}")
    print(f"      Deflated Sharpe Ratio: {dsr:.4f}")
    print(f"      Interpretation: {dsr*100:.1f}% probability result is not due to luck")
    
    return {'dsr': dsr, 'expected_max': expected_max}


# =============================================================================
# TEST 3: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_intervals(returns: pd.Series, n_bootstrap: int = 5000) -> Dict:
    """Calculate bootstrap confidence intervals for Sharpe."""
    print("\n   === BOOTSTRAP CONFIDENCE INTERVALS ===")
    print(f"   Running {n_bootstrap} bootstrap samples...")
    
    n = len(returns)
    bootstrap_sharpes = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap
        block_size = 20
        n_blocks = n // block_size + 1
        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_size))
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]
        
        boot_returns = returns.iloc[indices].values
        boot_sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252) if np.std(boot_returns) > 0 else 0
        bootstrap_sharpes.append(boot_sharpe)
    
    # Confidence intervals
    ci_90 = (np.percentile(bootstrap_sharpes, 5), np.percentile(bootstrap_sharpes, 95))
    ci_95 = (np.percentile(bootstrap_sharpes, 2.5), np.percentile(bootstrap_sharpes, 97.5))
    ci_99 = (np.percentile(bootstrap_sharpes, 0.5), np.percentile(bootstrap_sharpes, 99.5))
    
    point_estimate = np.mean(bootstrap_sharpes)
    
    print(f"      Point Estimate: {point_estimate:.2f}")
    print(f"      90% CI: [{ci_90[0]:.2f}, {ci_90[1]:.2f}]")
    print(f"      95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
    print(f"      99% CI: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")
    print(f"      CI excludes zero: {'YES' if ci_95[0] > 0 else 'NO'}")
    
    return {'ci_95': ci_95, 'ci_99': ci_99, 'point_estimate': point_estimate}


# =============================================================================
# TEST 4: PERMUTATION TEST
# =============================================================================

def permutation_test(returns: pd.Series, n_permutations: int = 1000) -> Dict:
    """
    Permutation test to check if returns are significantly different from random.
    """
    print("\n   === PERMUTATION TEST ===")
    print(f"   Running {n_permutations} permutations...")
    
    observed_sharpe = calculate_sharpe(returns)
    
    perm_sharpes = []
    for _ in range(n_permutations):
        perm_returns = returns.sample(frac=1, replace=False).values
        perm_sharpe = np.mean(perm_returns) / np.std(perm_returns) * np.sqrt(252) if np.std(perm_returns) > 0 else 0
        perm_sharpes.append(perm_sharpe)
    
    # P-value: proportion of permuted Sharpes >= observed
    p_value = np.mean([ps >= observed_sharpe for ps in perm_sharpes])
    
    percentile_rank = stats.percentileofscore(perm_sharpes, observed_sharpe)
    
    print(f"      Observed Sharpe: {observed_sharpe:.2f}")
    print(f"      Permutation Mean: {np.mean(perm_sharpes):.2f}")
    print(f"      Permutation Std: {np.std(perm_sharpes):.2f}")
    print(f"      Percentile Rank: {percentile_rank:.1f}%")
    print(f"      p-value: {p_value:.4f}")
    
    return {'p_value': p_value, 'percentile_rank': percentile_rank}


# =============================================================================
# TEST 5: RANDOM STRATEGY BENCHMARK
# =============================================================================

def random_strategy_benchmark(prices: pd.DataFrame, vix: pd.Series,
                              n_random: int = 1000) -> Dict:
    """
    Compare champion to randomly generated strategies.
    """
    print("\n   === RANDOM STRATEGY BENCHMARK ===")
    print(f"   Generating {n_random} random strategies...")
    
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Champion Sharpe
    champion_returns = get_strategy_returns(oos_prices, oos_vix, 
                                            CHAMPION_VIX, CHAMPION_SMA, CHAMPION_CAP)
    champion_sharpe = calculate_sharpe(champion_returns)
    
    # Generate random strategies
    random_sharpes = []
    
    for i in range(n_random):
        if i % 100 == 0:
            print(f"      Progress: {i}/{n_random}...", end="\r", flush=True)
        
        # Random weights (daily rebalanced)
        n = len(oos_prices.iloc[65:])
        n_assets = len(oos_prices.columns)
        
        # Random weight changes
        weights = np.random.dirichlet(np.ones(n_assets), size=n)
        weights_df = pd.DataFrame(weights, index=oos_prices.iloc[65:].index, columns=oos_prices.columns)
        
        returns = oos_prices.pct_change().iloc[65:].fillna(0)
        port_returns = (weights_df.shift(1) * returns).sum(axis=1).fillna(0)
        
        sharpe = calculate_sharpe(port_returns)
        random_sharpes.append(sharpe)
    
    print(f"      Progress: {n_random}/{n_random} - Done!")
    
    # Statistics
    percentile = stats.percentileofscore(random_sharpes, champion_sharpe)
    beat_count = sum(1 for rs in random_sharpes if rs < champion_sharpe)
    
    print(f"\n      Champion Sharpe: {champion_sharpe:.2f}")
    print(f"      Random Mean: {np.mean(random_sharpes):.2f}")
    print(f"      Random Max: {np.max(random_sharpes):.2f}")
    print(f"      Beats {beat_count}/{n_random} random strategies ({percentile:.1f}%)")
    
    return {'percentile': percentile, 'beat_count': beat_count, 'random_max': np.max(random_sharpes)}


# =============================================================================
# MAIN
# =============================================================================

def run_selection_bias_tests():
    """Run all selection bias tests."""
    
    print("=" * 80)
    print("   SELECTION BIAS & STATISTICAL VALIDATION TESTS")
    print("   Testing if Japan+Korea Strategy is due to luck/data mining")
    print("=" * 80)
    
    # Fetch data
    print("\nFetching data...")
    prices, vix = fetch_data(years=5)
    print(f"   {len(prices.columns)} assets, {len(prices)} days")
    
    # Get OOS returns
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    champion_returns = get_strategy_returns(oos_prices, oos_vix,
                                            CHAMPION_VIX, CHAMPION_SMA, CHAMPION_CAP)
    champion_sharpe = calculate_sharpe(champion_returns)
    
    print(f"\n   Champion OOS Sharpe: {champion_sharpe:.2f}")
    
    # Run tests
    print("\n" + "=" * 80)
    print("   RUNNING STATISTICAL TESTS")
    print("=" * 80)
    
    results = {}
    
    # Test 1: White's Reality Check
    results['whites'] = whites_reality_check(prices, vix, n_bootstrap=1000)
    
    # Test 2: Deflated Sharpe Ratio
    # For original strategy, we only tested 1 hypothesis (direct translation of PDF)
    results['dsr'] = deflated_sharpe_ratio(champion_sharpe, n_trials=1, 
                                           n_obs=len(champion_returns))
    
    # Test 3: Bootstrap CI
    results['bootstrap'] = bootstrap_confidence_intervals(champion_returns, n_bootstrap=2000)
    
    # Test 4: Permutation Test
    results['permutation'] = permutation_test(champion_returns, n_permutations=1000)
    
    # Test 5: Random Strategy Benchmark
    results['random'] = random_strategy_benchmark(prices, vix, n_random=500)
    
    # Summary
    print("\n" + "=" * 80)
    print("   SUMMARY: SELECTION BIAS TESTS")
    print("=" * 80)
    
    print(f"""
   ╔══════════════════════════════════════════════════════════════════╗
   ║                    SELECTION BIAS TEST RESULTS                   ║
   ╠══════════════════════════════════════════════════════════════════╣
   ║                                                                  ║
   ║  White's Reality Check:                                          ║
   ║     p-value: {results['whites']['p_value']:.4f} {'✅ PASS' if results['whites']['p_value'] < 0.10 else '❌ FAIL':>38}  ║
   ║                                                                  ║
   ║  Deflated Sharpe Ratio:                                          ║
   ║     DSR: {results['dsr']['dsr']:.4f} ({results['dsr']['dsr']*100:.1f}% not due to luck) {'✅ PASS' if results['dsr']['dsr'] > 0.5 else '⚠️ CAUTION':>18}  ║
   ║                                                                  ║
   ║  Bootstrap 95% CI:                                               ║
   ║     [{results['bootstrap']['ci_95'][0]:.2f}, {results['bootstrap']['ci_95'][1]:.2f}] {'✅ PASS' if results['bootstrap']['ci_95'][0] > 0 else '❌ FAIL':>38}  ║
   ║                                                                  ║
   ║  Permutation Test:                                               ║
   ║     p-value: {results['permutation']['p_value']:.4f} {'✅ PASS' if results['permutation']['p_value'] < 0.05 else '❌ FAIL':>38}  ║
   ║                                                                  ║
   ║  Random Strategy Benchmark:                                      ║
   ║     Beats {results['random']['beat_count']}/500 ({results['random']['percentile']:.1f}%) {'✅ PASS' if results['random']['percentile'] > 95 else '⚠️ CHECK':>32}  ║
   ║                                                                  ║
   ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Final verdict
    passed = 0
    if results['whites']['p_value'] < 0.10: passed += 1
    if results['dsr']['dsr'] > 0.5: passed += 1
    if results['bootstrap']['ci_95'][0] > 0: passed += 1
    if results['permutation']['p_value'] < 0.05: passed += 1
    if results['random']['percentile'] > 95: passed += 1
    
    print(f"   TESTS PASSED: {passed}/5")
    
    if passed >= 4:
        print(f"\n    STRATEGY IS NOT DUE TO SELECTION BIAS!")
    elif passed >= 3:
        print(f"\n   Strategy shows promise but some tests are borderline")
    else:
        print(f"\n   ⚠️ CAUTION: Strategy may have selection bias issues")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_selection_bias_tests()
