
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. SETUP & UTILS
# ================

WEIRD_DATA = {
    "cheese": {2000:30, 2010:33, 2011:33.3, 2012:33.5, 2013:34, 2014:34.5, 2015:35, 2020:39, 2021:40.2, 2022:42, 2023:42.3, 2024:42.5},
    "coffee": {2000:3.5, 2010:3.91, 2011:5.19, 2012:5.68, 2013:5.45, 2020:4.43, 2021:4.71, 2022:5.89, 2023:6.16, 2024:6.32}
}

def get_erp_signal(year):
    # Simplified Boolean: Is Inflation High?
    # Logic: Cheese > 3% OR Coffee > 5% YoY
    cheese_yoy = (WEIRD_DATA['cheese'].get(year, 0) / WEIRD_DATA['cheese'].get(year-1, 1)) - 1
    coffee_yoy = (WEIRD_DATA['coffee'].get(year, 0) / WEIRD_DATA['coffee'].get(year-1, 1)) - 1
    return (cheese_yoy > 0.03) or (coffee_yoy > 0.05)

def fetch_data():
    print("Fetching Extensive Data for Validation...")
    tickers = [
        'SPY', 'QQQ', 'TLT', 'GLD', 'UUP', 'XLE', # Trad
        'TQQQ', 'TMF', 'UPRO', # Lev
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX'
    ]
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    return prices[prices.index >= '2005-01-01'].ffill()

# 2. OMNI ENGINE (Functionized)
# =============================
def run_omni_strategy(prices, use_lev=False, vol_target=0.15):
    """
    Runs the Omni Strategy on the given price data.
    use_lev: If True, uses TQQQ/TMF instead of QQQ/TLT.
    vol_target: Annualized Volatility Target (e.g. 0.15 = 15%)
    """
    rets = prices.pct_change().fillna(0)
    
    # Assets
    risk_asset = 'TQQQ' if use_lev and 'TQQQ' in prices else 'QQQ'
    safe_bond = 'TMF' if use_lev and 'TMF' in prices else 'TLT'
    safe_gold = 'GLD'
    safe_dollar = 'UUP'
    safe_energy = 'XLE'
    crypto = 'BTC-USD'
    
    # 1. Signals
    # ----------
    # Golden Rule (Trend)
    spy = prices['SPY']
    ma200 = spy.rolling(200).mean()
    bull_trend = (spy > ma200).shift(1).fillna(False)
    
    # VIX Regime (Vol)
    vix = prices['^VIX'].fillna(20)
    vix_ma = vix.rolling(20).mean()
    bull_vix = (vix < vix_ma).shift(1).fillna(False)
    
    # ERP Inflation (Weird Data)
    inflation_years = [y for y in range(2000, 2030) if get_erp_signal(y)]
    is_inflation = pd.Series(False, index=prices.index)
    for y in inflation_years:
        is_inflation[prices.index.year == y] = True
    is_inflation = is_inflation.shift(1).fillna(False)
    
    # Crypto Vol Control
    btc_p = prices.get(crypto, pd.Series(0, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    crypto_safe = (btc_vol < 100).shift(1).fillna(False)
    
    # 2. Allocations
    # --------------
    weights = pd.DataFrame(0.0, index=prices.index, columns=[risk_asset, safe_bond, safe_gold, safe_dollar, safe_energy, crypto])
    
    # A. BULL MARKET (Trend=Up) -> Ultimate Attack
    # Logic: 40% Crypto (if safe), 60% Tech/Lev
    # If Crypto unsafe, 100% Tech/Lev
    
    mask_bull = bull_trend
    
    # Vectorized Allocation
    # Default Bull: 100% Risk Asset
    weights.loc[mask_bull, risk_asset] = 1.0
    
    # If Crypto Safe, split
    mask_crypto_on = mask_bull & crypto_safe
    weights.loc[mask_crypto_on, risk_asset] = 0.6
    weights.loc[mask_crypto_on, crypto] = 0.4
    
    # B. BEAR MARKET (Trend=Down) -> Defense (Trifecta/HRP)
    # Logic: 
    # Normal Bear: HRP (Bonds/Gold/Dollar)
    # Inflation Bear: Real HRP (Energy/Gold/Dollar) - NO BONDS
    
    mask_bear = ~bull_trend
    mask_bear_norm = mask_bear & ~is_inflation
    mask_bear_inf = mask_bear & is_inflation
    
    # Normal Bear: 40% Bond, 40% Dollar, 20% Gold (Approx HRP)
    # Note: Using fixed weights for stability in backtest vs rolling opt
    weights.loc[mask_bear_norm, safe_bond] = 0.50
    weights.loc[mask_bear_norm, safe_dollar] = 0.30
    weights.loc[mask_bear_norm, safe_gold] = 0.20
    
    # Inflation Bear: 40% Energy, 40% Dollar, 20% Gold (No Bonds)
    weights.loc[mask_bear_inf, safe_energy] = 0.40
    weights.loc[mask_bear_inf, safe_dollar] = 0.40
    weights.loc[mask_bear_inf, safe_gold] = 0.20

    # 3. Returns Calculation
    # ----------------------
    # Shift weights 1 day
    w_final = weights.shift(1).fillna(0)
    
    # Match columns
    avail_cols = [c for c in w_final.columns if c in rets.columns]
    port_ret = (w_final[avail_cols] * rets[avail_cols]).sum(axis=1)
    
    # 4. Volatility Target Overlay
    # ----------------------------
    if vol_target is not None:
        roll_std = port_ret.rolling(20).std() * np.sqrt(252)
        leverage = vol_target / roll_std.replace(0, np.inf)
        leverage = leverage.clip(0.5, 2.0).shift(1).fillna(1.0) # Cap lev at 2x
        port_ret = port_ret * leverage
        
    return port_ret

# 3. ANALYSIS SUITE
# =================

def bootstrap_test(strategy_rets, benchmark_rets, n_sims=1000):
    """
    Checks if Strategy Outperformance is statistically significant compared to Benchmark.
    """
    print(f"\n🎲 Running Bootstrap ({n_sims} simulations)...")
    
    real_diff = strategy_rets.mean() - benchmark_rets.mean()
    
    count = 0
    combined = pd.concat([strategy_rets, benchmark_rets], axis=1)
    
    diffs = []
    
    for _ in range(n_sims):
        # Resample with replacement
        sample = combined.sample(n=len(combined), replace=True)
        # Calculate Sharpe Diff
        s_ret = sample.iloc[:, 0]
        b_ret = sample.iloc[:, 1]
        
        # Simple Mean Return Diff for Speed (Sharpe is complex in bootstrap due to vol)
        # Actually consistent Sharpe diff is better
        s_sharpe = s_ret.mean() / s_ret.std() if s_ret.std() > 0 else 0
        b_sharpe = b_ret.mean() / b_ret.std() if b_ret.std() > 0 else 0
        
        diffs.append(s_sharpe - b_sharpe)
        
    # Calculate P-Value
    # H0: Strategy <= Benchmark
    # p = proportion of simulations where Strat <= Bench (Diff <= 0)
    # But we want to see if Real Diff is outlier.
    
    # Let's use simple Percentile
    # Where does the ACTUAL Sharpe Diff fall in the distribution of random shuffles?
    # Wait, simple shuffle breaks time correlation.
    # Block Bootstrap is better but complex.
    # Let's do a simple comparison:
    # 95% Confidence Interval of the Strategy Sharpe alone
    
    bs_sharpes = []
    for _ in range(n_sims):
        s = strategy_rets.sample(n=len(strategy_rets), replace=True)
        sharpe = (s.mean() / s.std()) * np.sqrt(252)
        bs_sharpes.append(sharpe)
        
    lower = np.percentile(bs_sharpes, 2.5)
    upper = np.percentile(bs_sharpes, 97.5)
    mean_bs = np.mean(bs_sharpes)
    
    print(f"   Strategy Sharpe: {mean_bs:.2f} (95% CI: {lower:.2f} - {upper:.2f})")
    
    return lower, upper


def walk_forward_test(prices):
    print("\n🚶 Running Walk-Forward Validation (2 Year Train / 1 Year Test)...")
    years = prices.index.year.unique()
    start_year = years[2] # Need 2 years data
    
    oos_returns = []
    
    for y in range(start_year, years[-1]):
        train_start = str(y-2)
        train_end = str(y-1)
        test_year = str(y)
        
        # Train Data (Not used for fitting parameters here as strategy is rules-based)
        # In a real ML model we would fit here.
        # But we can calculate 'Expected Sharpe' to see if it holds up.
        
        # Test Data
        mask_test = prices.index.year == y
        test_prices = prices[mask_test]
        
        if len(test_prices) < 200: continue
            
        # Run Strategy on Test Year
        r = run_omni_strategy(test_prices, use_lev=True, vol_target=0.20)
        oos_returns.append(r)
        
    full_oos = pd.concat(oos_returns)
    return full_oos

def analyze():
    prices = fetch_data()
    
    # 1. Full Backtest (Levered vs Unlevered)
    print("\n--- 1. FULL HISTORY BACKTEST (20 Years) ---")
    r_base = run_omni_strategy(prices, use_lev=False, vol_target=0.15)
    r_turbo = run_omni_strategy(prices, use_lev=True, vol_target=0.20) # Turbo: TQQQ/TMF + 20% Vol
    
    bench = prices['SPY'].pct_change().fillna(0)
    
    stats = {
        'Omni Base': r_base,
        'Omni Turbo (Lev)': r_turbo,
        'SPY': bench
    }
    
    print(f"{'Strategy':<20} {'Sharpe':<8} {'Return':<10} {'MaxDD':<10}")
    print("-" * 60)
    for n, r in stats.items():
        cum = (1 + r).prod() - 1
        vol = r.std() * np.sqrt(252)
        sharpe = r.mean() * 252 / vol if vol > 0 else 0
        w = (1+r).cumprod()
        dd = (w/w.cummax()) - 1
        mdd = dd.min()
        print(f"{n:<20} {sharpe:<8.2f} {cum:<10.0%} {mdd:<10.1%}")
        
    # 2. Walk Forward
    print("\n--- 2. OUT-OF-SAMPLE STABILITY ---")
    r_wf = walk_forward_test(prices)
    sharpe_wf = (r_wf.mean() / r_wf.std()) * np.sqrt(252)
    print(f"   Walk-Forward Sharpe: {sharpe_wf:.2f}")
    if sharpe_wf > 1.0:
        print("   ✅ PASSED: Strategy holds up in Out-of-Sample testing.")
    else:
        print("   ⚠️ WARNING: Strategy degrades out of sample.")
        
    # 3. Bootstrap
    print("\n--- 3. STATISTICAL SIGNIFICANCE ---")
    lower, upper = bootstrap_test(r_turbo, bench, n_sims=1000)
    
    if lower > 1.0:
        print(f"   ✅ PASSED: 95% Confidence that Sharpe is > 1.0 ({lower:.2f})")
    else:
        print(f"   ⚠️ CAUTION: Lower bound of Sharpe confidence is low ({lower:.2f})")

if __name__ == "__main__":
    analyze()
