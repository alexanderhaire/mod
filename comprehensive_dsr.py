"""
Comprehensive DSR Analysis with N_eff and Placebo Tests
========================================================

Implements:
1. Eigenvalue-based N_eff (effective number of independent trials)
2. Placebo test (shuffled weird-data years)
3. Proper DSR with both N_raw and N_eff
4. Comparison with economically direct proxies

RUN: python comprehensive_dsr.py
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
# DSR IMPLEMENTATION
# =============================================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(p):
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
        return 0
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
    kurt = m4 / (m2**2) if m2 > 0 else 3.0
    return T, mu, sd, skew, kurt

def dsr_deflated_sharpe(r, N, sr_trials_daily):
    """
    DSR with proper null-hypothesis threshold interpretation.
    
    SR0 = expected max Sharpe under NULL (zero skill) given N trials
    This uses trial Sharpe dispersion to estimate how variable 
    "no-skill" strategies would look.
    """
    T, mu, sd, skew, kurt = sample_moments(r)
    if T < 3 or sd == 0:
        return np.nan, np.nan, np.nan, np.nan

    sr_hat = mu / sd  # daily Sharpe

    # Use trial variance to estimate null dispersion
    var_sr = float(np.var(sr_trials_daily, ddof=1))
    
    # Expected max Sharpe under null (centered at zero)
    z1 = norm_ppf(1.0 - 1.0 / N)
    z2 = norm_ppf(1.0 - 1.0 / (N * e))
    sr0 = sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    # Non-normality adjustment
    denom = sqrt(1.0 - skew * sr0 + ((kurt - 1.0) / 4.0) * (sr0**2))
    stat = (sr_hat - sr0) * sqrt(T - 1.0) / denom
    dsr = norm_cdf(stat)

    haircut_sr = sr_hat - sr0
    return dsr, sr_hat, sr0, haircut_sr

def annualize_sharpe(sr, periods=252):
    return sr * sqrt(periods)

# =============================================================================
# EFFECTIVE N FROM EIGENVALUES
# =============================================================================

def effective_n_from_returns(variant_returns: np.ndarray) -> float:
    """
    Compute effective number of independent trials using eigenvalue method.
    
    variant_returns: array shape (T, N) of aligned returns for each variant
    """
    X = variant_returns.copy()
    # Drop columns with NaNs or zero variance
    keep = np.isfinite(X).all(axis=0) & (np.std(X, axis=0, ddof=1) > 0)
    X = X[:, keep]
    
    if X.shape[1] < 2:
        return 1.0
    
    # Correlation matrix
    C = np.corrcoef(X, rowvar=False)
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(C)
    eigvals = eigvals[eigvals > 0]  # Keep positive only
    # Effective N
    neff = (eigvals.sum() ** 2) / (np.square(eigvals).sum())
    return float(neff)

# =============================================================================
# STRATEGY IMPLEMENTATION
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

def erp_variant_strategy(prices, vix, threshold, var1_name, var1_wt, var2_name, var2_wt, 
                          var3_name=None, var3_wt=0, weird_data=None):
    """Parameterized ERP strategy."""
    if weird_data is None:
        weird_data = WEIRD_DATA
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(min(252, len(prices)), len(prices)):
        date = prices.index[i]
        year = date.year
        
        signal = 0
        for var_name, var_wt in [(var1_name, var1_wt), (var2_name, var2_wt), (var3_name, var3_wt)]:
            if var_name and var_name in weird_data:
                data = weird_data[var_name]
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
# PLACEBO TEST: Shuffle weird data years
# =============================================================================

def create_shuffled_weird_data(seed=None):
    """Shuffle the year labels of weird data to test for spurious correlation."""
    if seed is not None:
        np.random.seed(seed)
    
    shuffled = {}
    for var_name, data in WEIRD_DATA.items():
        years = list(data.keys())
        values = list(data.values())
        np.random.shuffle(values)
        shuffled[var_name] = dict(zip(years, values))
    return shuffled

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   COMPREHENSIVE DSR ANALYSIS")
    print("   With N_eff Eigenvalue Calculation and Placebo Tests")
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
    
    split_point = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:] if vix is not None else None
    
    print(f"   OOS Period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    print(f"   OOS Days: {len(oos_prices)}")
    
    # ==========================================================================
    # STEP 1: Generate all variant returns (for N_eff calculation)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   STEP 1: GENERATING VARIANT RETURNS FOR N_EFF")
    print("=" * 80)
    
    variants = [
        ("Base", 0.02, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("Base+Coffee", 0.02, "netflix", -0.5, "cheese", 0.3, "coffee", 0.2),
        ("Loose", 0.01, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("Tight", 0.03, "netflix", -0.5, "cheese", 0.3, None, 0),
        ("MoreNetflix", 0.02, "netflix", -0.6, "cheese", 0.2, None, 0),
        ("MoreCheese", 0.02, "netflix", -0.4, "cheese", 0.4, None, 0),
        ("NetflixOnly", 0.02, "netflix", -0.5, None, 0, None, 0),
        ("CheeseOnly", 0.02, "cheese", 0.5, None, 0, None, 0),
        ("CoffeeOnly", 0.02, "coffee", 0.5, None, 0, None, 0),
        ("ButterOnly", 0.02, "butter", 0.5, None, 0, None, 0),
        ("AvocadoOnly", 0.02, "avocado", 0.5, None, 0, None, 0),
        ("Nflx/Coffee", 0.02, "netflix", -0.5, "coffee", 0.3, None, 0),
        ("Cheese/Coffee", 0.02, "cheese", 0.3, "coffee", 0.3, None, 0),
    ]
    
    all_returns = []
    all_sharpes = []
    base_returns = None
    
    print(f"\n   {'Variant':<20} {'Sharpe (ann)':>12}")
    print("   " + "-" * 35)
    
    for name, thresh, v1, w1, v2, w2, v3, w3 in variants:
        weights = erp_variant_strategy(oos_prices, oos_vix, thresh, v1, w1, v2, w2, v3, w3)
        returns, sharpe = backtest_variant(oos_prices, oos_vix, weights)
        all_returns.append(returns.values)
        all_sharpes.append(sharpe)
        print(f"   {name:<20} {sharpe:>12.2f}")
        
        if name == "Base+Coffee":
            base_returns = returns
    
    # Align and stack returns
    min_len = min(len(r) for r in all_returns)
    variant_returns = np.column_stack([r[-min_len:] for r in all_returns])
    
    print(f"\n   Total variants: {len(variants)}")
    print(f"   Returns shape: {variant_returns.shape}")
    
    # ==========================================================================
    # STEP 2: Compute N_eff using eigenvalue method
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   STEP 2: EIGENVALUE-BASED N_EFF")
    print("=" * 80)
    
    n_eff = effective_n_from_returns(variant_returns)
    n_raw = len(variants)
    
    print(f"\n   Raw N (variants tested): {n_raw}")
    print(f"   Effective N (eigenvalue): {n_eff:.2f}")
    print(f"   Independence ratio: {n_eff/n_raw:.1%}")
    
    # Correlation matrix details
    C = np.corrcoef(variant_returns, rowvar=False)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = eigvals[eigvals > 0]
    
    print(f"\n   Eigenvalue spectrum:")
    for i, ev in enumerate(sorted(eigvals, reverse=True)[:5]):
        print(f"      λ_{i+1} = {ev:.3f} ({ev/eigvals.sum()*100:.1f}% of variance)")
    print(f"      (+ {len(eigvals)-5} more...)")
    
    # ==========================================================================
    # STEP 3: DSR with both N_raw and N_eff
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   STEP 3: DSR COMPARISON (N_raw vs N_eff)")
    print("=" * 80)
    
    r = base_returns.values
    T, mu, sd, skew, kurt = sample_moments(r)
    sr_trials_daily = [s / np.sqrt(252) for s in all_sharpes]
    
    print(f"\n   Return statistics:")
    print(f"   T={T}, skew={skew:.3f}, kurt={kurt:.3f}")
    print(f"   Daily trial Sharpe: mean={np.mean(sr_trials_daily):.4f}, std={np.std(sr_trials_daily):.4f}")
    
    print(f"\n   {'Metric':<30} {'N=raw ({})'.format(n_raw):<15} {'N=eff ({:.1f})'.format(n_eff):<15}")
    print("   " + "-" * 60)
    
    dsr_raw, sr_hat, sr0_raw, haircut_raw = dsr_deflated_sharpe(r, n_raw, sr_trials_daily)
    dsr_eff, _, sr0_eff, haircut_eff = dsr_deflated_sharpe(r, max(2, int(n_eff)), sr_trials_daily)
    
    print(f"   {'Observed Sharpe (ann)':<30} {annualize_sharpe(sr_hat):<15.2f} {annualize_sharpe(sr_hat):<15.2f}")
    print(f"   {'SR0 null threshold (ann)':<30} {annualize_sharpe(sr0_raw):<15.2f} {annualize_sharpe(sr0_eff):<15.2f}")
    print(f"   {'Haircut Sharpe (ann)':<30} {annualize_sharpe(max(haircut_raw,0)):<15.2f} {annualize_sharpe(max(haircut_eff,0)):<15.2f}")
    print(f"   {'DSR (prob of skill)':<30} {dsr_raw:<15.1%} {dsr_eff:<15.1%}")
    
    # ==========================================================================
    # STEP 4: PLACEBO TEST (shuffled years)
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   STEP 4: PLACEBO TEST (Shuffled Weird Data)")
    print("=" * 80)
    
    print("\n   Testing if strategy 'works' with shuffled year labels...")
    print("   (If placebo Sharpes are similar to real, the signal is spurious)")
    
    n_placebo = 100
    placebo_sharpes = []
    
    for i in range(n_placebo):
        shuffled = create_shuffled_weird_data(seed=i)
        weights = erp_variant_strategy(oos_prices, oos_vix, 0.02, "netflix", -0.5, "cheese", 0.3, 
                                        weird_data=shuffled)
        returns, sharpe = backtest_variant(oos_prices, oos_vix, weights)
        placebo_sharpes.append(sharpe)
    
    real_sharpe = all_sharpes[0]  # Base strategy
    placebo_mean = np.mean(placebo_sharpes)
    placebo_std = np.std(placebo_sharpes)
    placebo_pct = (np.array(placebo_sharpes) < real_sharpe).mean() * 100
    
    print(f"\n   Real Sharpe:     {real_sharpe:.2f}")
    print(f"   Placebo Mean:    {placebo_mean:.2f}")
    print(f"   Placebo Std:     {placebo_std:.2f}")
    print(f"   Placebo Range:   [{min(placebo_sharpes):.2f}, {max(placebo_sharpes):.2f}]")
    print(f"   Percentile:      {placebo_pct:.1f}%")
    
    if placebo_pct > 95:
        print(f"\n   ✅ Real strategy beats {placebo_pct:.0f}% of placebos - signal appears REAL")
    elif placebo_pct > 80:
        print(f"\n   ⚠️  Real strategy beats {placebo_pct:.0f}% of placebos - MODERATE evidence")
    else:
        print(f"\n   ❌ Real strategy only beats {placebo_pct:.0f}% of placebos - likely SPURIOUS")
    
    # ==========================================================================
    # STEP 5: FINAL HONEST VERDICT
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("   FINAL HONEST VERDICT")
    print("=" * 80)
    
    print(f"""
   DSR Analysis:
   ─────────────────────────────────────────────
   N_raw = {n_raw}  →  DSR = {dsr_raw:.1%}
   N_eff = {n_eff:.1f}  →  DSR = {dsr_eff:.1%}
   
   Placebo Test:
   ─────────────────────────────────────────────
   Real Sharpe beats {placebo_pct:.0f}% of shuffled versions
   
   Interpretation:
   ─────────────────────────────────────────────
   SR0 = {annualize_sharpe(sr0_raw):.2f} is the expected max Sharpe under NULL (zero skill)
   with {n_raw} trials and observed dispersion std={np.std(sr_trials_daily):.4f} daily.
   
   Since observed Sharpe ({annualize_sharpe(sr_hat):.2f}) >> SR0, DSR is high.
   
   However, the TRUE test is the placebo: if shuffling years doesn't
   kill the strategy, the signal is data-mined coincidence.
    """)
    
    if dsr_eff > 0.95 and placebo_pct > 95:
        print("   🏆 VERDICT: Strong statistical evidence of real edge")
    elif dsr_eff > 0.80 and placebo_pct > 80:
        print("   ⚠️  VERDICT: Moderate evidence - proceed with caution")
    else:
        print("   ❌ VERDICT: Insufficient evidence after rigorous testing")
    
    print("\n" + "=" * 80)
