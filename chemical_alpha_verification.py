"""
Chemical Alpha Verification
===========================

Rigorous statistical stress-testing of Chemical/LME Alpha.
Verifies if the alpha found is statistically significant or just luck/noise.

Metrics:
1. T-Statistic (Is Alpha > 0 with > 95% confidence?)
2. Rolling Alpha (Does it persist over time?)
3. Information Ratio (Return per unit of active risk)
4. Tail Risk (Skew/Kurtosis - does it blow up?)

RUN: python chemical_alpha_verification.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Focus on the Top Candidates found in previous steps
TARGETS = {
    'Antofagasta (UK)': {'ticker': 'ANTO.L', 'benchmark': '^FTSE'},
    'ProShares Copper': {'ticker': 'CPER',   'benchmark': 'SPY'},
    'SQM (Lithium)':    {'ticker': 'SQM',    'benchmark': 'SPY'},
    'Glencore (UK)':    {'ticker': 'GLEN.L', 'benchmark': '^FTSE'},
    'Copper Miners':    {'ticker': 'COPX',   'benchmark': 'SPY'}
}

# =============================================================================
# ENGINE
# =============================================================================

def fetch_data():
    print("📊 Fetching Verification Data (Long History)...")
    all_tickers = [t['ticker'] for t in TARGETS.values()] + list(set(t['benchmark'] for t in TARGETS.values()))
    
    data = yf.download(all_tickers, start='2010-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna()
    print(f"   History: {len(prices)} trading days")
    return prices

def run_statistical_tests(prices):
    print("\n🔬 STATISTICAL VERIFICATION REPORT")
    print("=" * 100)
    print(f"{'Asset':<20} {'Alpha (Ann)':<12} {'T-Stat':<8} {'P-Value':<8} {'Sig (>95%)':<10} {'IR':<6} {'Skew':<6}")
    print("-" * 100)
    
    results = {}
    
    for name, config in TARGETS.items():
        ticker = config['ticker']
        bench = config['benchmark']
        
        if ticker not in prices.columns or bench not in prices.columns:
            continue
            
        # Aligned Returns
        asset_ret = prices[ticker].pct_change().dropna()
        bench_ret = prices[bench].pct_change().dropna()
        
        common = asset_ret.index.intersection(bench_ret.index)
        y = asset_ret.loc[common] # Asset
        x = bench_ret.loc[common] # Benchmark
        
        if len(y) < 100:
            continue
            
        # 1. Regression (Alpha/Beta)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        alpha_daily = intercept
        alpha_ann = alpha_daily * 252
        
        # 2. T-Statistic for Alpha
        # t = coeff / std_err_of_coeff
        # We need std error of the intercept
        # residual variance
        y_pred = slope * x + intercept
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (len(y) - 2)
        
        # Standard error of intercept
        x_mean = np.mean(x)
        s_xx = np.sum((x - x_mean)**2)
        se_intercept = np.sqrt(mse * (1/len(y) + x_mean**2/s_xx))
        
        t_stat = alpha_daily / se_intercept
        p_val_alpha = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y)-2))
        
        is_sig = "✅ YES" if p_val_alpha < 0.05 else "❌ NO"
        
        # 3. Information Ratio
        active_ret = y - x
        tracking_error = active_ret.std() * np.sqrt(252)
        ir = (active_ret.mean() * 252) / tracking_error
        
        # 4. Tail Risk
        skew = stats.skew(y)
        
        print(f"{name:<20} {alpha_ann:<12.1%} {t_stat:<8.2f} {p_val_alpha:<8.3f} {is_sig:<10} {ir:<6.2f} {skew:<6.1f}")
        
        results[name] = {
            'prices': prices[ticker],
            'bench_prices': prices[bench],
            'alpha': alpha_ann,
            'p_val': p_val_alpha
        }
        
    print("-" * 100)
    return results

def check_stability(results):
    print("\n📅 ROLLING ALPHA STABILITY (12-Month Window)")
    print("-" * 60)
    
    for name, data in results.items():
        p = data['prices']
        b = data['bench_prices']
        
        # Rolling 1-year returns
        r_asset = p.pct_change(252)
        r_bench = b.pct_change(252)
        
        # Rolling Alpha (approx as return diff for speed)
        active = (r_asset - r_bench).dropna()
        
        win_rate = (active > 0).mean()
        avg_outperformance = active.mean()
        
        status = "STABLE 🟢" if win_rate > 0.6 else "UNSTABLE ⚠️"
        if win_rate < 0.4: status = "FAIL 🔴"
        
        print(f"{name:<20} Win Rate: {win_rate:<6.1%} Avg Excess: {avg_outperformance:<6.1%} {status}")

if __name__ == "__main__":
    prices = fetch_data()
    res = run_statistical_tests(prices)
    check_stability(res)
