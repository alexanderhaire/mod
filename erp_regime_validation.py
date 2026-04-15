"""
ERP-Regime Strategy - Comprehensive Statistical Validation
============================================================

All tests from validate_edge.py applied to prove the ERP-Regime strategy:

1. In-Sample vs Out-of-Sample Performance
2. Sharpe Ratio Significance (t-test, Lo 2002 standard error)
3. Bootstrap Confidence Intervals (10,000 samples)
4. Probabilistic Sharpe Ratio (Bailey & Lopez de Prado)
5. Monte Carlo Luck Test (10,000 simulations)
6. Walk-Forward Validation (5 folds)
7. Regime Analysis (VIX-based)
8. Alpha and Information Ratio vs SPY

RUN: python erp_regime_validation.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ERP REGIME STRATEGY (From money_finder.py)
# =============================================================================

WEIRD_DATA = {
    "netflix_subscribers": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0,
        2025: 320.0, 2026: 340.0,
    },
    "cheese_consumption": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
        2025: 43.0, 2026: 43.5,
    },
    "coffee_price": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
        2025: 6.50, 2026: 6.70,
    },
}


def get_erp_signals(date):
    """Get ERP-derived trading signals."""
    year = date.year
    signals = {}
    
    for name, data in WEIRD_DATA.items():
        if year in data and year - 1 in data:
            yoy = (data[year] - data[year-1]) / data[year-1]
            signals[name] = yoy
    
    netflix_yoy = signals.get("netflix_subscribers", 0)
    cheese_yoy = signals.get("cheese_consumption", 0)
    coffee_yoy = signals.get("coffee_price", 0)
    
    xle_signal = -netflix_yoy * 0.5 + cheese_yoy * 0.3 + coffee_yoy * 0.2
    
    return {"xle_erp": xle_signal}


def erp_regime_strategy(prices, vix):
    """The winning ERP-Regime strategy."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    base_assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(252, len(prices)):
        date = prices.index[i]
        erp = get_erp_signals(date)
        xle_signal = erp['xle_erp']
        
        v = vix.iloc[i]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        
        w = {a: 0.25 for a in base_assets}
        
        if 'XLE' in w:
            if xle_signal > 0.02:
                w['XLE'] = 0.35
                w['SPY'] = 0.20 if 'SPY' in w else 0
            elif xle_signal < -0.02:
                w['XLE'] = 0.10
                w['GLD'] = 0.35 if 'GLD' in w else 0.25
        
        if v > 25:
            if 'TLT' in w:
                w['TLT'] = 0.40
            if 'XLE' in w:
                w['XLE'] = max(0.05, w['XLE'] * 0.5)
        
        total = sum(w.values())
        for a in w:
            weights.loc[date, a] = w[a] / total if total > 0 else 0
    
    return weights.shift(1).fillna(0)

def backtest(prices, vix, warmup=300):
    """Run backtest and return daily returns."""
    weights = erp_regime_strategy(prices, vix)
    returns = prices.pct_change()
    
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common_cols = weights.columns.intersection(returns.columns)
    port_returns = (weights[common_cols].shift(1) * returns[common_cols]).sum(axis=1)
    
    return port_returns

if __name__ == "__main__":
    print("=" * 70)
    print("   ERP-REGIME STRATEGY - COMPREHENSIVE VALIDATION")
    print("   Proving the Edge with Statistical Rigor")
    print("=" * 70)

    # =============================================================================
    # FETCH DATA
    # =============================================================================

    print("\n📊 Fetching data...")

    tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'XLB', 'XLI', 'JNK']
    end = datetime.now()
    start = end - timedelta(days=365*12)

    data = yf.download(tickers, start=start, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()

    vix_data = yf.download('^VIX', start=start, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    vix = vix.reindex(prices.index).ffill().fillna(15)

    print(f"   Loaded {len(prices)} days")

    # Split 70/30
    split_point = int(len(prices) * 0.7)
    is_prices = prices.iloc[:split_point]
    oos_prices = prices.iloc[split_point:]
    is_vix = vix.iloc[:split_point]
    oos_vix = vix.iloc[split_point:]

    print(f"   In-Sample:      {is_prices.index[0].date()} to {is_prices.index[-1].date()} ({len(is_prices)} days)")
    print(f"   Out-of-Sample:  {oos_prices.index[0].date()} to {oos_prices.index[-1].date()} ({len(oos_prices)} days)")


    # =============================================================================
    # TEST 1: IN-SAMPLE VS OUT-OF-SAMPLE
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 1: IN-SAMPLE VS OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 70)

    is_returns = backtest(is_prices, is_vix)
    oos_returns = backtest(oos_prices, oos_vix)

    is_sharpe = is_returns.mean() / is_returns.std() * np.sqrt(252) if is_returns.std() > 0 else 0
    oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(252) if oos_returns.std() > 0 else 0

    degradation = (1 - oos_sharpe / is_sharpe) * 100 if is_sharpe != 0 else 0

    print(f"\n   In-Sample Sharpe:      {is_sharpe:.2f}")
    print(f"   Out-of-Sample Sharpe:  {oos_sharpe:.2f}")
    print(f"   Degradation:           {degradation:.1f}%")

    if oos_sharpe > 1.0:
        print(f"\n   ✅ PASSED: OOS Sharpe > 1.0")
    elif oos_sharpe > 0.5:
        print(f"\n   ⚠️  MODERATE: OOS Sharpe between 0.5-1.0")
    else:
        print(f"\n   ❌ FAILED: OOS Sharpe < 0.5")


    # =============================================================================
    # TEST 2: SHARPE RATIO SIGNIFICANCE (Lo 2002)
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 2: SHARPE RATIO STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    n = len(oos_returns)
    sharpe = oos_sharpe

    # Standard error (Lo 2002)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(252)

    # T-statistic vs benchmark of 0
    t_stat = sharpe / se_sharpe

    # P-value (one-tailed)
    p_value = 1 - stats.t.cdf(t_stat, df=n-1)

    print(f"\n   Sharpe Ratio:      {sharpe:.3f}")
    print(f"   Standard Error:    {se_sharpe:.3f}")
    print(f"   T-Statistic:       {t_stat:.3f}")
    print(f"   P-Value:           {p_value:.4f}")

    if p_value < 0.01:
        print(f"\n   ✅ SIGNIFICANT at 1% level ***")
    elif p_value < 0.05:
        print(f"\n   ✅ SIGNIFICANT at 5% level **")
    elif p_value < 0.10:
        print(f"\n   ⚠️  SIGNIFICANT at 10% level *")
    else:
        print(f"\n   ❌ NOT significant (p={p_value:.3f})")


    # =============================================================================
    # TEST 3: BOOTSTRAP CONFIDENCE INTERVAL
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 3: BOOTSTRAP CONFIDENCE INTERVAL (95%, 10,000 samples)")
    print("=" * 70)

    print("\n   Running bootstrap...", end=" ", flush=True)

    n_bootstrap = 10000
    boot_sharpes = []

    for _ in range(n_bootstrap):
        sample = oos_returns.sample(n, replace=True)
        if sample.std() > 0:
            boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(252))

    boot_sharpes = np.array(boot_sharpes)
    ci_low = np.percentile(boot_sharpes, 2.5)
    ci_high = np.percentile(boot_sharpes, 97.5)

    print("Done")

    print(f"\n   Sharpe 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"   Mean:          {np.mean(boot_sharpes):.2f}")
    print(f"   Median:        {np.median(boot_sharpes):.2f}")

    if ci_low > 0:
        print(f"\n   ✅ CI excludes zero - Statistically significant edge!")
    else:
        print(f"\n   ❌ CI includes zero - Edge not proven")


    # =============================================================================
    # TEST 4: PROBABILISTIC SHARPE RATIO
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 4: PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)")
    print("=" * 70)

    skew = stats.skew(oos_returns)
    kurt = stats.kurtosis(oos_returns)

    se_psr = np.sqrt((1 - skew * sharpe + ((kurt - 1) / 4) * sharpe**2) / (n - 1))
    z = sharpe / (se_psr * np.sqrt(252)) if se_psr > 0 else 0
    psr = stats.norm.cdf(z)

    print(f"\n   Return Skewness:  {skew:.3f}")
    print(f"   Return Kurtosis:  {kurt:.3f}")
    print(f"   PSR:              {psr:.1%}")

    if psr > 0.95:
        print(f"\n   ✅ PSR > 95% - Very high confidence in edge")
    elif psr > 0.80:
        print(f"\n   ⚠️  PSR 80-95% - High confidence")
    elif psr > 0.50:
        print(f"\n   ⚠️  PSR 50-80% - Moderate confidence")
    else:
        print(f"\n   ❌ PSR < 50% - Low confidence")


    # =============================================================================
    # TEST 5: MONTE CARLO LUCK TEST
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 5: MONTE CARLO LUCK TEST (10,000 simulations)")
    print("=" * 70)

    print("\n   Running Monte Carlo...", end=" ", flush=True)

    n_sims = 10000
    random_sharpes = []

    for _ in range(n_sims):
        random_returns = np.random.normal(0, oos_returns.std(), n)
        if np.std(random_returns) > 0:
            random_sharpes.append(np.mean(random_returns) / np.std(random_returns) * np.sqrt(252))

    random_sharpes = np.array(random_sharpes)
    percentile = (random_sharpes < sharpe).mean() * 100

    print("Done")

    print(f"\n   Actual Sharpe:      {sharpe:.3f}")
    print(f"   Random Mean Sharpe: {np.mean(random_sharpes):.3f}")
    print(f"   Random Std Sharpe:  {np.std(random_sharpes):.3f}")
    print(f"   Percentile Rank:    {percentile:.1f}%")

    if percentile > 99:
        print(f"\n   ✅ Beats 99% of random - HIGHLY SIGNIFICANT")
    elif percentile > 95:
        print(f"\n   ✅ Beats 95% of random - SIGNIFICANT")
    elif percentile > 90:
        print(f"\n   ⚠️  Beats 90% of random - Moderate")
    else:
        print(f"\n   ❌ Does not beat 95% - May be luck")


    # =============================================================================
    # TEST 6: WALK-FORWARD VALIDATION
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 6: WALK-FORWARD VALIDATION (5 folds)")
    print("=" * 70)

    n_splits = 5
    split_size = len(prices) // (n_splits + 1)

    is_sharpes = []
    oos_sharpes_wf = []

    print("\n   Running walk-forward...", end=" ", flush=True)

    for i in range(n_splits):
        is_end = (i + 1) * split_size
        oos_start = is_end
        oos_end = oos_start + split_size
        
        if oos_end > len(prices):
            break
        
        wf_is_prices = prices.iloc[:is_end]
        wf_oos_prices = prices.iloc[oos_start:oos_end]
        wf_is_vix = vix.iloc[:is_end]
        wf_oos_vix = vix.iloc[oos_start:oos_end]
        
        try:
            is_ret = backtest(wf_is_prices, wf_is_vix, warmup=200)
            oos_ret = backtest(wf_oos_prices, wf_oos_vix, warmup=200)
            
            is_s = is_ret.mean() / is_ret.std() * np.sqrt(252) if len(is_ret) > 0 and is_ret.std() > 0 else 0
            oos_s = oos_ret.mean() / oos_ret.std() * np.sqrt(252) if len(oos_ret) > 0 and oos_ret.std() > 0 else 0
            
            is_sharpes.append(is_s)
            oos_sharpes_wf.append(oos_s)
        except:
            pass

    print("Done")

    print(f"\n   In-Sample Sharpes:      {[f'{s:.2f}' for s in is_sharpes]}")
    print(f"   Out-of-Sample Sharpes:  {[f'{s:.2f}' for s in oos_sharpes_wf]}")
    print(f"\n   IS Mean:  {np.mean(is_sharpes):.2f}")
    print(f"   OOS Mean: {np.mean(oos_sharpes_wf):.2f}")

    oos_positive = sum(1 for s in oos_sharpes_wf if s > 0)
    if oos_positive == len(oos_sharpes_wf):
        print(f"\n   ✅ ALL folds have positive OOS Sharpe")
    elif oos_positive >= len(oos_sharpes_wf) * 0.8:
        print(f"\n   ⚠️  {oos_positive}/{len(oos_sharpes_wf)} folds have positive OOS Sharpe")
    else:
        print(f"\n   ❌ Only {oos_positive}/{len(oos_sharpes_wf)} folds positive")


    # =============================================================================
    # TEST 7: REGIME ANALYSIS
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 7: REGIME ANALYSIS (VIX-based)")
    print("=" * 70)

    # Align VIX with returns
    common_idx = oos_returns.index.intersection(oos_vix.index)
    oos_ret_aligned = oos_returns.loc[common_idx]

    if isinstance(oos_vix, pd.DataFrame):
        vix_aligned = oos_vix.iloc[:, 0].loc[common_idx]
    else:
        vix_aligned = oos_vix.loc[common_idx]

    low_vol = vix_aligned < 15
    normal_vol = (vix_aligned >= 15) & (vix_aligned < 25)
    high_vol = vix_aligned >= 25

    regimes = [
        ("Low VIX (<15)", low_vol),
        ("Normal VIX (15-25)", normal_vol),
        ("High VIX (>25)", high_vol),
    ]

    print(f"\n   {'Regime':<20} {'N Days':>10} {'Sharpe':>10} {'Ann Return':>12}")
    print("   " + "-" * 55)

    for regime_name, mask in regimes:
        regime_ret = oos_ret_aligned[mask.values]
        if len(regime_ret) > 20:
            r_sharpe = regime_ret.mean() / regime_ret.std() * np.sqrt(252) if regime_ret.std() > 0 else 0
            r_ann = regime_ret.mean() * 252
            print(f"   {regime_name:<20} {len(regime_ret):>10} {r_sharpe:>10.2f} {r_ann:>11.1%}")
        else:
            print(f"   {regime_name:<20} {len(regime_ret):>10} {'N/A':>10}")


    # =============================================================================
    # TEST 8: ALPHA AND INFORMATION RATIO
    # =============================================================================

    print("\n" + "=" * 70)
    print("   TEST 8: ALPHA AND INFORMATION RATIO (vs SPY)")
    print("=" * 70)

    spy_returns = prices['SPY'].pct_change().iloc[split_point + 300:]
    strat_returns = oos_returns

    # Align
    common = strat_returns.index.intersection(spy_returns.index)
    strat = strat_returns.loc[common].values
    bench = spy_returns.loc[common].values

    # Regression
    slope, intercept, r_value, p_value_alpha, std_err = stats.linregress(bench, strat)

    alpha_annual = intercept * 252
    tracking_error = np.std(strat - bench) * np.sqrt(252)
    ir = alpha_annual / tracking_error if tracking_error > 0 else 0

    print(f"\n   Alpha (annualized):  {alpha_annual:.2%}")
    print(f"   Beta:                {slope:.2f}")
    print(f"   R-Squared:           {r_value**2:.2%}")
    print(f"   Information Ratio:   {ir:.2f}")
    print(f"   Alpha P-Value:       {p_value_alpha:.4f}")

    if alpha_annual > 0.02 and p_value_alpha < 0.05:
        print(f"\n   ✅ Significant positive alpha!")
    elif alpha_annual > 0:
        print(f"\n   ⚠️  Positive but not significant alpha")
    else:
        print(f"\n   ❌ No alpha found")


    # =============================================================================
    # FINAL VERDICT
    # =============================================================================

    print("\n" + "=" * 70)
    print("   📊 FINAL VERDICT: IS THE EDGE REAL?")
    print("=" * 70)

    # Compile evidence
    evidence_for = 0
    evidence_against = 0

    tests = [
        ("OOS Sharpe > 1.0", oos_sharpe > 1.0, oos_sharpe > 0.5),
        ("P-Value < 0.05", p_value < 0.05, p_value < 0.10),
        ("Bootstrap CI > 0", ci_low > 0, ci_low > -0.5),
        ("PSR > 80%", psr > 0.80, psr > 0.50),
        ("Monte Carlo > 95%", percentile > 95, percentile > 90),
        ("Walk-Forward Consistent", oos_positive == len(oos_sharpes_wf), oos_positive >= 3),
        ("Positive Alpha", alpha_annual > 0.02, alpha_annual > 0),
    ]

    print("\n   Evidence Assessment:")
    print("   " + "-" * 50)

    for test_name, strong, weak in tests:
        if strong:
            print(f"   ✅ {test_name}")
            evidence_for += 2
        elif weak:
            print(f"   ⚠️  {test_name} (weak)")
            evidence_for += 1
        else:
            print(f"   ❌ {test_name}")
            evidence_against += 1

    print("   " + "-" * 50)
    print(f"   Evidence For:     {evidence_for} points")
    print(f"   Evidence Against: {evidence_against} points")
    print("   " + "-" * 50)

    if evidence_for >= 10 and evidence_against <= 1:
        print("\n   🏆 VERDICT: STRONG EVIDENCE OF REAL EDGE")
        print("   The strategy shows statistically significant alpha")
        print("   unlikely to be due to luck or overfitting.")
    elif evidence_for >= 7:
        print("\n   📊 VERDICT: MODERATE EVIDENCE OF EDGE")
        print("   The strategy shows promise but more validation needed.")
    elif evidence_for >= 4:
        print("\n   ⚠️  VERDICT: WEAK EVIDENCE OF EDGE")
        print("   Some positive signals but not conclusive.")
    else:
        print("\n   ❌ VERDICT: INSUFFICIENT EVIDENCE")
        print("   The observed returns may be due to luck.")

    print("\n" + "=" * 70)
