"""
Rigorous Statistical Edge Validation - ASIAN MARKETS
=====================================================

This script performs comprehensive statistical tests to determine if the
Compounder Detection Strategy has a REAL, statistically significant edge
when applied to ASIAN MARKET ETFs.

Uses the same strategy logic as the US version but with:
- Asian country/region ETFs (Japan, China, Korea, India, Taiwan, etc.)
- AAXJ (Asia ex-Japan) as the regime benchmark instead of SPY
- VIX for global volatility regime detection

Author: Statistical Validation Module (Asia Adaptation)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import (
    compounder_strategy,
    compounder_strategy_levered,
    compounder_strategy_no_overlay,
    CompounderStrategy
)


# =============================================================================
# DATA FETCHING - ASIAN MARKET DATA
# =============================================================================

def fetch_asian_data(years: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch real Asian market data from Yahoo Finance.
    
    Returns price DataFrame and VIX series.
    """
    print("Fetching ASIAN MARKET data from Yahoo Finance...")
    
    # Define Asian market tickers
    tickers = {
        'EWJ': 'Japan',
        'FXI': 'China Large-Cap',
        'EWY': 'South Korea',
        'INDA': 'India',
        'EWT': 'Taiwan',
        'EWH': 'Hong Kong',
        'EWS': 'Singapore',
        'AAXJ': 'Asia ex-Japan',  # Benchmark
        'GLD': 'Gold',            # Global hedge
        'TLT': 'Long Treasuries'  # Global hedge
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        # Fetch price data
        print(f"   Downloading {len(tickers)} Asian market assets from {start_date.date()} to {end_date.date()}...")
        data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
        
        # Rename columns to friendly names
        prices.columns = [tickers.get(c, c) for c in prices.columns]
        
        # Fetch VIX (global volatility indicator)
        print("   Downloading VIX (global volatility)...")
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if 'Close' in vix_data.columns:
            vix = vix_data['Close']
        else:
            vix = vix_data.iloc[:, 0] if len(vix_data.columns) > 0 else pd.Series()
        
        # Clean data
        prices = prices.dropna(how='all').ffill().dropna()
        vix = vix.reindex(prices.index).ffill().fillna(15)
        
        print(f"Loaded {len(prices)} days of real data for {len(prices.columns)} Asian market assets")
        return prices, vix
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return pd.DataFrame(), pd.Series()


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest_strategy(prices: pd.DataFrame, 
                      signal_func, 
                      vix: pd.Series = None,
                      transaction_cost: float = 0.001,
                      warmup: int = 65) -> Dict:
    """
    Run backtest and return detailed results including daily returns.
    """
    try:
        if 'vix' in signal_func.__code__.co_varnames:
            signals = signal_func(prices, vix)
        else:
            signals = signal_func(prices)
    except:
        signals = signal_func(prices)
    
    returns = prices.pct_change().fillna(0)
    signals = signals.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if signals.empty:
        return {'daily_returns': pd.Series(), 'sharpe': 0, 'equity_curve': pd.Series()}
    
    # Normalize signals
    abs_sum = signals.abs().sum(axis=1).replace(0, 1)
    normalized = signals.div(abs_sum, axis=0)
    
    # Smooth positions
    smoothed = normalized.ewm(span=5).mean()
    
    # Portfolio returns
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * transaction_cost
    
    # Metrics
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    return {
        'daily_returns': net_returns,
        'sharpe': sharpe,
        'equity_curve': equity,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'volatility': net_returns.std() * np.sqrt(252),
        'max_drawdown': ((equity - equity.cummax()) / equity.cummax()).min()
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def test_sharpe_significance(returns: pd.Series, benchmark_sharpe: float = 0) -> Dict:
    """
    Test if Sharpe ratio is statistically significant.
    """
    n = len(returns)
    if n < 30:
        return {'significant': False, 'p_value': 1.0, 'error': 'Insufficient data'}
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # Standard error of Sharpe (Lo 2002)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)
    
    # T-statistic
    t_stat = (sharpe - benchmark_sharpe) / se_sharpe
    
    # P-value (one-tailed: is Sharpe > benchmark?)
    p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    
    return {
        'sharpe': sharpe,
        'standard_error': se_sharpe,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01
    }


def bootstrap_sharpe_confidence_interval(returns: pd.Series, 
                                          n_bootstrap: int = 10000,
                                          confidence: float = 0.95) -> Dict:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.
    """
    n = len(returns)
    sharpes = []
    
    for _ in range(n_bootstrap):
        sample = returns.sample(n, replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    
    sharpes = np.array(sharpes)
    alpha = (1 - confidence) / 2
    
    ci_low = np.percentile(sharpes, alpha * 100)
    ci_high = np.percentile(sharpes, (1 - alpha) * 100)
    
    return {
        'sharpe_mean': np.mean(sharpes),
        'sharpe_median': np.median(sharpes),
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'confidence': confidence,
        'excludes_zero': ci_low > 0
    }


def probabilistic_sharpe_ratio(returns: pd.Series, 
                                benchmark_sharpe: float = 0) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR) per Bailey & Lopez de Prado.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # Skewness and kurtosis of returns
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis
    
    # Standard error including higher moments
    se = np.sqrt((1 - skew * sharpe + ((kurt - 1) / 4) * sharpe**2) / (n - 1))
    
    if se <= 0:
        return 0.5
    
    z = (sharpe - benchmark_sharpe) / (se * np.sqrt(252))
    psr = stats.norm.cdf(z)
    
    return psr


def monte_carlo_test(returns: pd.Series, n_simulations: int = 10000) -> Dict:
    """
    Monte Carlo simulation to test if returns are due to luck.
    """
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    n = len(returns)
    
    random_sharpes = []
    for _ in range(n_simulations):
        random_returns = np.random.normal(0, returns.std(), n)
        if np.std(random_returns) > 0:
            random_sharpes.append(np.mean(random_returns) / np.std(random_returns) * np.sqrt(252))
    
    random_sharpes = np.array(random_sharpes)
    percentile = (random_sharpes < actual_sharpe).mean() * 100
    
    return {
        'actual_sharpe': actual_sharpe,
        'random_mean_sharpe': np.mean(random_sharpes),
        'random_std_sharpe': np.std(random_sharpes),
        'percentile': percentile,
        'beats_random': percentile > 95,
        'highly_significant': percentile > 99
    }


def regime_analysis(returns: pd.Series, vix: pd.Series) -> Dict:
    """
    Analyze performance across different market regimes.
    """
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    common_idx = returns.index.intersection(vix.index)
    returns = returns.loc[common_idx]
    vix_aligned = vix.loc[common_idx]
    
    # Define regimes based on VIX
    low_vol = vix_aligned < 15
    normal_vol = (vix_aligned >= 15) & (vix_aligned < 25)
    high_vol = vix_aligned >= 25
    
    results = {}
    
    for regime, mask in [('Low VIX (<15)', low_vol), 
                          ('Normal VIX (15-25)', normal_vol),
                          ('High VIX (>25)', high_vol)]:
        regime_returns = returns[mask.values]
        if len(regime_returns) > 20:
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
            results[regime] = {
                'n_days': len(regime_returns),
                'sharpe': sharpe,
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252)
            }
        else:
            results[regime] = {'n_days': len(regime_returns), 'sharpe': 'N/A'}
    
    return results


def walk_forward_validation(prices: pd.DataFrame, 
                            signal_func,
                            vix: pd.Series,
                            n_splits: int = 5) -> Dict:
    """
    Walk-forward validation: train on past, test on future.
    """
    n = len(prices)
    split_size = n // (n_splits + 1)
    
    oos_sharpes = []
    is_sharpes = []
    
    for i in range(n_splits):
        is_end = (i + 1) * split_size
        oos_start = is_end
        oos_end = oos_start + split_size
        
        if oos_end > n:
            break
        
        is_prices = prices.iloc[:is_end]
        oos_prices = prices.iloc[oos_start:oos_end]
        
        is_vix = vix.iloc[:is_end]
        oos_vix = vix.iloc[oos_start:oos_end]
        
        is_result = backtest_strategy(is_prices, signal_func, is_vix)
        oos_result = backtest_strategy(oos_prices, signal_func, oos_vix)
        
        is_sharpes.append(is_result['sharpe'])
        oos_sharpes.append(oos_result['sharpe'])
    
    return {
        'in_sample_sharpes': is_sharpes,
        'out_of_sample_sharpes': oos_sharpes,
        'is_mean': np.mean(is_sharpes),
        'oos_mean': np.mean(oos_sharpes),
        'degradation': 1 - (np.mean(oos_sharpes) / np.mean(is_sharpes)) if np.mean(is_sharpes) != 0 else 0,
        'oos_consistent': all(s > 0 for s in oos_sharpes)
    }


def calculate_alpha_beta(strategy_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict:
    """
    Calculate alpha and beta vs benchmark using regression.
    """
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.loc[common_idx].values
    bench = benchmark_returns.loc[common_idx].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(bench, strat)
    
    alpha_annual = intercept * 252
    
    tracking_error = np.std(strat - bench) * np.sqrt(252)
    ir = alpha_annual / tracking_error if tracking_error > 0 else 0
    
    return {
        'alpha': alpha_annual,
        'beta': slope,
        'r_squared': r_value**2,
        'alpha_p_value': p_value,
        'alpha_significant': p_value < 0.05,
        'information_ratio': ir
    }


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_complete_validation():
    """
    Run comprehensive statistical validation on ASIAN MARKETS.
    """
    print("=" * 70)
    print("   RIGOROUS STATISTICAL EDGE VALIDATION - ASIAN MARKETS")
    print("   Testing Compounder Strategy on Asian Market ETFs")
    print("=" * 70)
    
    # Fetch Asian market data
    prices, vix = fetch_asian_data(years=10)
    
    if prices.empty:
        print("\n Cannot proceed without real market data.")
        print("   Please check your internet connection and try again.")
        return None
    
    # Split into in-sample and out-of-sample
    split_point = int(len(prices) * 0.7)
    
    is_prices = prices.iloc[:split_point]
    oos_prices = prices.iloc[split_point:]
    is_vix = vix.iloc[:split_point]
    oos_vix = vix.iloc[split_point:]
    
    print(f"\n Data Split:")
    print(f"   In-Sample:      {is_prices.index[0].date()} to {is_prices.index[-1].date()} ({len(is_prices)} days)")
    print(f"   Out-of-Sample:  {oos_prices.index[0].date()} to {oos_prices.index[-1].date()} ({len(oos_prices)} days)")
    
    # Define strategies
    strategies = {
        'Compounder (No Overlay)': compounder_strategy_no_overlay,
        'Compounder (with Kill Switch)': lambda p, v=None: compounder_strategy(p, v),
        'Compounder (Levered)': lambda p, v=None: compounder_strategy_levered(p, v)
    }
    
    # ==========================================================================
    # TEST 1: IN-SAMPLE VS OUT-OF-SAMPLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 1: IN-SAMPLE VS OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 70)
    
    results = {}
    
    print(f"\n{'Strategy':<25} {'IS Sharpe':>12} {'OOS Sharpe':>12} {'Degradation':>12}")
    print("-" * 65)
    
    for name, func in strategies.items():
        is_result = backtest_strategy(is_prices, func, is_vix)
        oos_result = backtest_strategy(oos_prices, func, oos_vix)
        
        degradation = (1 - oos_result['sharpe'] / is_result['sharpe']) * 100 if is_result['sharpe'] != 0 else 0
        
        results[name] = {
            'is': is_result,
            'oos': oos_result
        }
        
        status = "+" if oos_result['sharpe'] > 0 else "-"
        print(f"{status} {name:<23} {is_result['sharpe']:>12.2f} {oos_result['sharpe']:>12.2f} {degradation:>11.1f}%")
    
    # ==========================================================================
    # TEST 2: SHARPE RATIO SIGNIFICANCE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 2: SHARPE RATIO STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    for name, res in results.items():
        oos_returns = res['oos']['daily_returns']
        if len(oos_returns) < 30:
            continue
        
        sig_test = test_sharpe_significance(oos_returns, benchmark_sharpe=0)
        
        print(f"\n {name} (Out-of-Sample):")
        print(f"   Sharpe Ratio:     {sig_test['sharpe']:.3f}")
        print(f"   Standard Error:   {sig_test['standard_error']:.3f}")
        print(f"   T-Statistic:      {sig_test['t_statistic']:.3f}")
        print(f"   P-Value:          {sig_test['p_value']:.4f}")
        
        if sig_test['significant_1pct']:
            print(f"   Significance:      SIGNIFICANT at 1% level***")
        elif sig_test['significant_5pct']:
            print(f"   Significance:      SIGNIFICANT at 5% level**")
        else:
            print(f"   Significance:      NOT significant (p={sig_test['p_value']:.3f})")
    
    # ==========================================================================
    # TEST 3: BOOTSTRAP CONFIDENCE INTERVALS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 3: BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 70)
    
    for name, res in results.items():
        oos_returns = res['oos']['daily_returns']
        if len(oos_returns) < 100:
            continue
        
        print(f"\n   Running 10,000 bootstrap samples for {name}...", end=" ", flush=True)
        bootstrap = bootstrap_sharpe_confidence_interval(oos_returns, n_bootstrap=10000)
        
        print("Done")
        print(f"   Sharpe 95% CI: [{bootstrap['ci_lower']:.3f}, {bootstrap['ci_upper']:.3f}]")
        
        if bootstrap['excludes_zero']:
            print(f"    CI excludes zero - Statistically significant edge!")
        else:
            print(f"    CI includes zero - Edge not conclusively proven")
    
    # ==========================================================================
    # TEST 4: PROBABILISTIC SHARPE RATIO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 4: PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'OOS Sharpe':>12} {'PSR':>12} {'Interpretation':>20}")
    print("-" * 75)
    
    for name, res in results.items():
        oos_returns = res['oos']['daily_returns']
        
        psr = probabilistic_sharpe_ratio(oos_returns, benchmark_sharpe=0)
        sharpe = res['oos']['sharpe']
        
        if psr > 0.95:
            interp = " Very High"
        elif psr > 0.80:
            interp = " High"
        elif psr > 0.50:
            interp = " Moderate"
        else:
            interp = " Low"
        
        print(f"{name:<25} {sharpe:>12.3f} {psr:>11.1%} {interp:>20}")
    
    # ==========================================================================
    # TEST 5: MONTE CARLO LUCK TEST
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 5: MONTE CARLO LUCK TEST (10,000 simulations)")
    print("=" * 70)
    
    for name, res in results.items():
        oos_returns = res['oos']['daily_returns']
        if len(oos_returns) < 100:
            continue
        
        print(f"\n   Testing {name}...", end=" ", flush=True)
        mc_result = monte_carlo_test(oos_returns, n_simulations=10000)
        print("Done")
        
        print(f"   Actual Sharpe:      {mc_result['actual_sharpe']:.3f}")
        print(f"   Random Mean Sharpe: {mc_result['random_mean_sharpe']:.3f}")
        print(f"   Percentile Rank:    {mc_result['percentile']:.1f}%")
        
        if mc_result['highly_significant']:
            print(f"    Beats 99% of random strategies - HIGHLY SIGNIFICANT EDGE")
        elif mc_result['beats_random']:
            print(f"    Beats 95% of random strategies - SIGNIFICANT EDGE")
        else:
            print(f"    Does not beat 95% of random - May be luck")
    
    # ==========================================================================
    # TEST 6: WALK-FORWARD VALIDATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 6: WALK-FORWARD VALIDATION (5 folds)")
    print("=" * 70)
    
    for name, func in strategies.items():
        print(f"\n   {name}:")
        wf = walk_forward_validation(prices, func, vix, n_splits=5)
        
        print(f"   In-Sample Sharpes:      {[f'{s:.2f}' for s in wf['in_sample_sharpes']]}")
        print(f"   Out-of-Sample Sharpes:  {[f'{s:.2f}' for s in wf['out_of_sample_sharpes']]}")
        print(f"   IS Mean: {wf['is_mean']:.2f} | OOS Mean: {wf['oos_mean']:.2f}")
        print(f"   Degradation: {wf['degradation']*100:.1f}%")
        
        if wf['oos_consistent']:
            print(f"    Consistent positive OOS Sharpe across all folds")
        else:
            print(f"    Some folds had negative OOS Sharpe")
    
    # ==========================================================================
    # TEST 7: REGIME ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 7: REGIME ANALYSIS (Performance by VIX level)")
    print("=" * 70)
    
    aw_returns = results['Compounder (with Kill Switch)']['oos']['daily_returns']
    regimes = regime_analysis(aw_returns, oos_vix)
    
    print(f"\n   Compounder Strategy Performance by Market Regime:")
    print(f"\n   {'Regime':<20} {'N Days':>10} {'Sharpe':>12} {'Annual Return':>15}")
    print("   " + "-" * 60)
    
    for regime, data in regimes.items():
        if data['sharpe'] != 'N/A':
            print(f"   {regime:<20} {data['n_days']:>10} {data['sharpe']:>12.2f} {data['mean_return']:>14.1%}")
        else:
            print(f"   {regime:<20} {data['n_days']:>10} {'N/A':>12}")
    
    # ==========================================================================
    # TEST 8: ALPHA AND INFORMATION RATIO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   TEST 8: ALPHA AND INFORMATION RATIO (vs Asia ex-Japan)")
    print("=" * 70)
    
    # Get AAXJ (Asia ex-Japan) returns as benchmark
    benchmark_returns = prices['Asia ex-Japan'].pct_change().iloc[split_point:].dropna()
    
    for name, res in results.items():
        strat_returns = res['oos']['daily_returns']
        
        if len(strat_returns) < 100:
            continue
        
        alpha_beta = calculate_alpha_beta(strat_returns, benchmark_returns)
        
        print(f"\n   {name}:")
        print(f"   Alpha (annual):     {alpha_beta['alpha']:.2%}")
        print(f"   Beta:               {alpha_beta['beta']:.2f}")
        print(f"   R-Squared:          {alpha_beta['r_squared']:.2%}")
        print(f"   Information Ratio:  {alpha_beta['information_ratio']:.2f}")
        
        if alpha_beta['alpha_significant']:
            print(f"    Alpha is statistically significant (p={alpha_beta['alpha_p_value']:.4f})")
        else:
            print(f"    Alpha not significant (p={alpha_beta['alpha_p_value']:.4f})")
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("   FINAL VERDICT: DO YOU HAVE A REAL EDGE IN ASIAN MARKETS?")
    print("=" * 70)
    
    aw_oos = results['Compounder (with Kill Switch)']['oos']
    aw_oos_returns = aw_oos['daily_returns']
    
    # Compile evidence
    sharpe_sig = test_sharpe_significance(aw_oos_returns)
    bootstrap = bootstrap_sharpe_confidence_interval(aw_oos_returns)
    psr = probabilistic_sharpe_ratio(aw_oos_returns)
    mc = monte_carlo_test(aw_oos_returns)
    alpha_beta = calculate_alpha_beta(aw_oos_returns, benchmark_returns)
    
    evidence_for = 0
    evidence_against = 0
    
    print("\n   Evidence Assessment:")
    print("   " + "-" * 50)
    
    # 1. OOS Sharpe
    if aw_oos['sharpe'] > 0.5:
        print(f"    OOS Sharpe > 0.5: {aw_oos['sharpe']:.2f}")
        evidence_for += 2
    elif aw_oos['sharpe'] > 0:
        print(f"    OOS Sharpe positive but < 0.5: {aw_oos['sharpe']:.2f}")
        evidence_for += 1
    else:
        print(f"    OOS Sharpe negative: {aw_oos['sharpe']:.2f}")
        evidence_against += 2
    
    # 2. Statistical significance
    if sharpe_sig['significant_1pct']:
        print(f"    Sharpe significant at 1% level (p={sharpe_sig['p_value']:.4f})")
        evidence_for += 2
    elif sharpe_sig['significant_5pct']:
        print(f"    Sharpe significant at 5% level (p={sharpe_sig['p_value']:.4f})")
        evidence_for += 1
    else:
        print(f"    Sharpe not statistically significant (p={sharpe_sig['p_value']:.4f})")
        evidence_against += 1
    
    # 3. Bootstrap CI
    if bootstrap['excludes_zero']:
        print(f"    95% CI excludes zero: [{bootstrap['ci_lower']:.2f}, {bootstrap['ci_upper']:.2f}]")
        evidence_for += 2
    else:
        print(f"    95% CI includes zero: [{bootstrap['ci_lower']:.2f}, {bootstrap['ci_upper']:.2f}]")
        evidence_against += 1
    
    # 4. PSR
    if psr > 0.80:
        print(f"    High PSR: {psr:.1%}")
        evidence_for += 1
    else:
        print(f"    Moderate PSR: {psr:.1%}")
    
    # 5. Monte Carlo
    if mc['beats_random']:
        print(f"    Beats {mc['percentile']:.0f}% of random strategies")
        evidence_for += 2
    else:
        print(f"    Only beats {mc['percentile']:.0f}% of random strategies")
        evidence_against += 1
    
    # 6. Alpha
    if alpha_beta['alpha'] > 0.02 and alpha_beta['alpha_significant']:
        print(f"    Significant positive alpha: {alpha_beta['alpha']:.1%}")
        evidence_for += 2
    elif alpha_beta['alpha'] > 0:
        print(f"    Positive but small/insignificant alpha: {alpha_beta['alpha']:.1%}")
        evidence_for += 0.5
    else:
        print(f"    Negative alpha: {alpha_beta['alpha']:.1%}")
        evidence_against += 1
    
    print("\n   " + "-" * 50)
    print(f"   Evidence For Edge:     {evidence_for:.1f} points")
    print(f"   Evidence Against Edge: {evidence_against:.1f} points")
    print("   " + "-" * 50)
    
    if evidence_for >= 8 and evidence_against <= 1:
        print("\n   VERDICT: STRONG EVIDENCE OF REAL EDGE IN ASIAN MARKETS")
        print("   The Compounder Strategy demonstrates statistically")
        print("   significant outperformance in Asian markets.")
    elif evidence_for >= 5 and evidence_for > evidence_against * 2:
        print("\n   VERDICT: MODERATE EVIDENCE OF EDGE IN ASIAN MARKETS")
        print("   The strategy shows promise but more data or testing needed.")
    elif evidence_for > evidence_against:
        print("\n   VERDICT: WEAK EVIDENCE OF EDGE IN ASIAN MARKETS")
        print("   Some positive signals but not conclusive.")
    else:
        print("\n   VERDICT: INSUFFICIENT EVIDENCE OF EDGE IN ASIAN MARKETS")
        print("   The observed returns may be due to luck or overfitting.")
    
    print("\n" + "=" * 70)
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_complete_validation()
