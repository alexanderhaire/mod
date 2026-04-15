"""
Deflated Sharpe Ratio (DSR) Calculation
=======================================

Computes the "honest" Sharpe Ratio after accounting for:
1. Multiple testing / exploration budget
2. Non-normal returns (skew/kurtosis)
3. Winner's curse / selection bias

Based on Bailey & López de Prado methodology.

RUN: python compute_dsr.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from math import erf, sqrt, log, e
import warnings
warnings.filterwarnings('ignore')

EULER_GAMMA = 0.5772156649015329

# =============================================================================
# DSR IMPLEMENTATION (from user-provided code)
# =============================================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(p):
    """Peter J. Acklam approximation for inverse normal CDF."""
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425

    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def sample_moments(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    mu = r.mean()
    sd = r.std(ddof=1)
    m2 = np.mean((r - mu)**2)
    m3 = np.mean((r - mu)**3)
    m4 = np.mean((r - mu)**4)
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    kurt = m4 / (m2**2)  if m2 > 0 else 3.0
    return T, mu, sd, skew, kurt

def dsr_deflated_sharpe(r, N, sr_trials=None):
    """
    Returns:
      dsr: Deflated Sharpe Ratio (probability of skill after multiple testing)
      sr_hat: observed Sharpe at sampling freq (NOT annualized)
      sr0: multiple-testing-adjusted Sharpe threshold at sampling freq
      haircut_sr: (sr_hat - sr0) at sampling freq
    """
    T, mu, sd, skew, kurt = sample_moments(r)
    if T < 3 or sd == 0:
        return np.nan, np.nan, np.nan, np.nan

    sr_hat = mu / sd  # per-period Sharpe (NOT annualized)

    # Estimate variance of Sharpe across trials
    if sr_trials is not None and len(sr_trials) >= 5:
        sr_trials = np.asarray(sr_trials, dtype=float)
        sr_trials = sr_trials[np.isfinite(sr_trials)]
        var_sr = float(np.var(sr_trials, ddof=1))
    else:
        var_sr = 1.0 / (T - 1)

    # Expected maximum Sharpe under N independent trials
    z1 = norm_ppf(1.0 - 1.0 / N)
    z2 = norm_ppf(1.0 - 1.0 / (N * e))
    sr0 = sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    # Non-normality adjustment
    denom = sqrt(1.0 - skew * sr0 + ((kurt - 1.0) / 4.0) * (sr0**2))
    stat = (sr_hat - sr0) * sqrt(T - 1.0) / denom
    dsr = norm_cdf(stat)

    haircut_sr = sr_hat - sr0
    return dsr, sr_hat, sr0, haircut_sr

def annualize_sharpe(sr_per_period, periods_per_year=252):
    return sr_per_period * sqrt(periods_per_year)

# =============================================================================
# ERP STRATEGY VARIANTS - Generate all the variants we tested
# =============================================================================

WEIRD_DATA = {
    "netflix": {2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
                2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0, 2025: 320.0, 2026: 340.0},
    "cheese": {2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
               2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5, 2025: 43.0, 2026: 43.5},
    "coffee": {2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
               2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32, 2025: 6.50, 2026: 6.70},
    "butter": {2015: 5.6, 2016: 5.7, 2017: 5.8, 2018: 5.9, 2019: 6.0,
               2020: 6.1, 2021: 6.2, 2022: 6.4, 2023: 6.5, 2024: 6.6, 2025: 6.7, 2026: 6.8},
    "avocado": {2015: 7.1, 2016: 7.1, 2017: 7.5, 2018: 8.0, 2019: 8.5,
                2020: 9.0, 2021: 9.0, 2022: 9.0, 2023: 9.2, 2024: 8.8, 2025: 9.0, 2026: 9.2},
}

def erp_variant_strategy(prices, vix, threshold, var1_name, var1_wt, var2_name, var2_wt, var3_name=None, var3_wt=0):
    """Parameterized ERP strategy to test all variants."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(252, len(prices)), len(prices)):
        date = prices.index[i]
        year = date.year
        
        # Calculate YoY for each variable
        signal = 0
        for var_name, var_wt in [(var1_name, var1_wt), (var2_name, var2_wt), (var3_name, var3_wt)]:
            if var_name and var_name in WEIRD_DATA:
                data = WEIRD_DATA[var_name]
                if year in data and year-1 in data:
                    yoy = (data[year] - data[year-1]) / data[year-1]
                    signal += yoy * var_wt
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if signal > threshold: w['XLE'], w['SPY'] = 0.35, 0.20
            elif signal < -threshold: w['XLE'], w['GLD'] = 0.10, 0.35
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w: 
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    return weights.shift(1).fillna(0)

def backtest_variant(prices, vix, weights, warmup=300):
    """Backtest a variant and return daily returns + Sharpe."""
    returns = prices.pct_change()
    
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    norm = weights[common].div(abs_sum, axis=0)
    
    port_ret = (norm.shift(1) * returns[common]).sum(axis=1)
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    
    return port_ret, sharpe

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   DEFLATED SHARPE RATIO (DSR) CALCULATION")
    print("   Honest Assessment After Multiple Testing Adjustment")
    print("=" * 80)
    
    # Fetch data
    print("\n📊 Fetching data...")
    tickers = ['SPY', 'XLE', 'GLD', 'TLT']
    end = datetime.now()
    start = end - timedelta(days=365*12)
    
    data = yf.download(tickers + ['^VIX'], start=start, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill().dropna()
    
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    if vix is not None:
        prices = prices.drop('^VIX', axis=1)
    
    # Split 70/30 (same as validation)
    split_point = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:] if vix is not None else None
    
    print(f"   OOS Period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    print(f"   OOS Days: {len(oos_prices)}")
    
    # ==========================================================================
    # GENERATE ALL VARIANT SHARPES
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   GENERATING ALL TESTED VARIANTS")
    print("=" * 80)
    
    # Define all the variants we supposedly tested
    variants = [
        # Base formula variants (threshold, var1, wt1, var2, wt2, var3, wt3)
        ("Base (Netflix/Cheese)", 0.02, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("Base + Coffee", 0.02, "netflix", -0.5, "cheese", 0.3, "coffee", 0.2),
        ("Loose Threshold", 0.01, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("Tight Threshold", 0.03, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("More Netflix", 0.02, "netflix", -0.6, "cheese", 0.2, None, 0),
        ("More Cheese", 0.02, "netflix", -0.4, "cheese", 0.4, None, 0),
        ("Less Netflix", 0.02, "netflix", -0.4, "cheese", 0.3, None, 0),
        
        # Alternative variable combos
        ("Netflix Only", 0.02, "netflix", -0.5, None, 0, None, 0),
        ("Cheese Only", 0.02, "cheese", 0.5, None, 0, None, 0),
        ("Coffee Only", 0.02, "coffee", 0.5, None, 0, None, 0),
        ("Butter Only", 0.02, "butter", 0.5, None, 0, None, 0),
        ("Avocado Only", 0.02, "avocado", 0.5, None, 0, None, 0),
        
        # Two-variable combos
        ("Netflix/Coffee", 0.02, "netflix", -0.5, "coffee", 0.3, None, 0),
        ("Netflix/Butter", 0.02, "netflix", -0.5, "butter", 0.3, None, 0),
        ("Netflix/Avocado", 0.02, "netflix", -0.5, "avocado", 0.3, None, 0),
        ("Cheese/Coffee", 0.02, "cheese", 0.3, "coffee", 0.3, None, 0),
        ("Cheese/Butter", 0.02, "cheese", 0.3, "butter", 0.3, None, 0),
        ("Butter/Coffee", 0.02, "butter", 0.3, "coffee", 0.3, None, 0),
        
        # Three-variable combos
        ("Netflix/Butter/Coffee", 0.02, "netflix", -0.5, "butter", 0.3, "coffee", 0.2),
        ("Netflix/Cheese/Avocado", 0.02, "netflix", -0.5, "cheese", 0.3, "avocado", 0.2),
        ("Cheese/Butter/Coffee", 0.02, "cheese", 0.3, "butter", 0.2, "coffee", 0.2),
        
        # Weight variations
        ("Heavy Netflix", 0.02, "netflix", -0.8, "cheese", 0.2, None, 0),
        ("Light Netflix", 0.02, "netflix", -0.3, "cheese", 0.4, None, 0),
        ("Equal Weights", 0.02, "netflix", -0.33, "cheese", 0.33, "coffee", 0.34),
        
        # Threshold variations
        ("Very Loose (0.005)", 0.005, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("Very Tight (0.05)", 0.05, "netflix", -0.5, "cheese", 0.3, None, 0),
    ]
    
    all_sharpes = []
    base_returns = None
    base_sharpe = None
    
    print(f"\n   {'Variant':<30} {'Sharpe':>10}")
    print("   " + "-" * 45)
    
    for name, thresh, v1, w1, v2, w2, v3, w3 in variants:
        try:
            weights = erp_variant_strategy(oos_prices, oos_vix, thresh, v1, w1, v2, w2, v3, w3)
            returns, sharpe = backtest_variant(oos_prices, oos_vix, weights)
            all_sharpes.append(sharpe)
            print(f"   {name:<30} {sharpe:>10.2f}")
            
            # Save the "winning" variant returns (Base + Coffee)
            if name == "Base + Coffee":
                base_returns = returns
                base_sharpe = sharpe
        except Exception as e:
            print(f"   {name:<30} {'Error':>10}")
    
    print(f"\n   Total variants tested: {len(all_sharpes)}")
    print(f"   Variant Sharpe range: [{min(all_sharpes):.2f}, {max(all_sharpes):.2f}]")
    print(f"   Variant Sharpe mean: {np.mean(all_sharpes):.2f}")
    print(f"   Variant Sharpe std: {np.std(all_sharpes):.2f}")
    
    # ==========================================================================
    # COMPUTE DSR
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   DEFLATED SHARPE RATIO RESULTS")
    print("=" * 80)
    
    if base_returns is None or len(base_returns) == 0:
        print("   ERROR: Could not compute base strategy returns")
    else:
        r = base_returns.values
        T, mu, sd, skew, kurt = sample_moments(r)
        
        print(f"\n   Return Statistics:")
        print(f"   Days (T):     {T}")
        print(f"   Mean (daily): {mu:.6f}")
        print(f"   Std (daily):  {sd:.6f}")
        print(f"   Skewness:     {skew:.3f}")
        print(f"   Kurtosis:     {kurt:.3f} (normal=3)")
        
        # Compute DSR for different N values
        # CRITICAL: Convert annualized trial Sharpes to daily first!
        sr_trials_daily = [s / np.sqrt(252) for s in all_sharpes]
        n_values = [26, 50, 118]
        
        print(f"\n   {'N Trials':<12} {'DSR':<10} {'SR_hat(ann)':<14} {'SR0(ann)':<12} {'Haircut(ann)':<14}")
        print("   " + "-" * 70)
        
        for N in n_values:
            dsr, sr_hat, sr0, haircut = dsr_deflated_sharpe(r, N, sr_trials=sr_trials_daily)
            
            sr_hat_ann = annualize_sharpe(sr_hat)
            sr0_ann = annualize_sharpe(sr0)
            haircut_ann = annualize_sharpe(max(haircut, 0))
            
            print(f"   N={N:<9} {dsr:<10.1%} {sr_hat_ann:<14.2f} {sr0_ann:<12.2f} {haircut_ann:<14.2f}")
        
        # Final honest assessment
        print("\n" + "=" * 80)
        print("   HONEST SHARPE ASSESSMENT")
        print("=" * 80)
        
        # CRITICAL FIX: Convert annualized Sharpes to daily before DSR
        # DSR should be computed entirely in per-period (daily) units
        sr_trials_daily = [s / np.sqrt(252) for s in all_sharpes]
        
        print(f"\n   Trial Sharpes (converted to daily):")
        print(f"   Mean: {np.mean(sr_trials_daily):.4f}")
        print(f"   Std:  {np.std(sr_trials_daily):.4f}")
        
        # Sanity check: expected max using simple approximation
        # E[max] ≈ mean + std * sqrt(2 * ln(N))
        import math
        expected_max_simple = np.mean(sr_trials_daily) + np.std(sr_trials_daily) * math.sqrt(2 * math.log(26))
        print(f"   Simple E[max] for N=26 (daily): {expected_max_simple:.4f}")
        print(f"   Simple E[max] (annualized): {expected_max_simple * np.sqrt(252):.2f}")
        
        # Use N=26 as "conservative" estimate
        dsr, sr_hat, sr0, haircut = dsr_deflated_sharpe(r, N=26, sr_trials=sr_trials_daily)
        
        print(f"""
   Observed OOS Sharpe:           {annualize_sharpe(sr_hat):.2f}
   Expected Max Sharpe By Luck:   {annualize_sharpe(sr0):.2f}  (N=26 trials)
   Haircut Sharpe:                {annualize_sharpe(max(haircut, 0)):.2f}
   
   DSR (Probability of Skill):   {dsr:.1%}
   
   Interpretation:
   - If DSR > 95%: Strong evidence of skill
   - If DSR 80-95%: Moderate evidence
   - If DSR 50-80%: Weak evidence
   - If DSR < 50%: Likely luck
        """)
        
        if dsr > 0.95:
            print("   ✅ VERDICT: Even after multiple testing, evidence suggests real skill")
        elif dsr > 0.80:
            print("   ⚠️  VERDICT: Some evidence of skill, but not conclusive")
        elif dsr > 0.50:
            print("   ⚠️  VERDICT: Weak evidence - could be luck or skill")
        else:
            print("   ❌ VERDICT: After multiple testing adjustment, this is likely luck")
        
    print("\n" + "=" * 80)
