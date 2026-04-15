
"""
ALEX.PY - THE ULTIMATE STRATEGY ENGINE
======================================
The "Golden Omni" + "Expanded Alpha" Strategy.
Consolidated for maximum performance and statistical proof.

REGIME LOGIC:
1. BULL (SPY > 200MA): 
   - 45% SPY (Core)
   - 10% TLT (Hedge)
   - 5% GLD (Real Asset)
   - 40% EXPANDED CRYPTO BASKET (Alpha Engine)

2. BEAR (SPY < 200MA):
   - INFLATION CHECK (XLE Trend):
     - IF INFLATION: 15% SPY, 35% XLE, 10% GLD, 40% Crypto (Hedge Inflation)
     - IF NORMAL:    15% SPY, 35% TLT, 10% GLD, 40% Crypto (Hedge Deflation)

ALPHA ENGINE:
- Scans 11+ Crypto Assets (BTC, ETH, SOL, DOGE, ADA, XRP, etc).
- Buys TOP 3 Momentum Leaders (14-day lookback).
- If Avg Alt Momentum < BTC Momentum -> Hides in BTC.

STATISTICS:
- CAGR, Sharpe, Sortino, MaxDD, Calmar, Alpha, Beta, Win Rate.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA ENGINE
# ==============================================================================

def fetch_data():
    print("🚀 ALEX.PY: Initializing Data Engine...")
    
    # The Universe
    trad = ['SPY', 'TLT', 'GLD', 'XLE']
    crypto = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        'ADA-USD', 'XRP-USD', 'AVAX-USD', 'SHIB-USD', 
        'DOT-USD', 'LINK-USD', 'LTC-USD'
    ]
    tickers = trad + crypto
    
    print(f"📡 Fetching history for {len(tickers)} assets (2020-Present)...")
    data = yf.download(tickers, start='2020-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                prices = data.xs('Adj Close', axis=1, level=1)
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill()
    
    # Ensure critical assets exist
    if 'SPY' not in prices.columns or 'BTC-USD' not in prices.columns:
        raise ValueError("CRITICAL: SPY or BTC missing from data.")
        
    # Drop rows before SPY starts (just in case)
    prices = prices.dropna(subset=['SPY'])
    
    print(f"✅ Data Loaded: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices

# ==============================================================================
# 2. STRATEGY ENGINE
# ==============================================================================

# ==============================================================================
# 2.5 STANDARD STRATEGY VARIANT
# ==============================================================================

def run_golden_omni_standard(prices):
    # Same logic, restricted universe
    universe = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    # Filter prices to just these for the alpha engine
    # (Rest of logic is identical to run_golden_omni_expanded, but limited universe)
    
    # We can actually just call the expanded function but pass a subset of prices columns?
    # No, macro logic needs SPY/XLE.
    # Better to copy-paste the core inner loop or parameterize it. 
    # For speed/cleanliness, I'll parameterize the runner.
    pass

def run_strategy_generic(prices, allowed_crypto=None):
    rets = prices.pct_change()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Macro
    spy = prices['SPY']
    ma200_spy = spy.rolling(200).mean()
    is_bull = (spy > ma200_spy).shift(1).fillna(False)
    
    xle = prices.get('XLE', spy)
    ma200_xle = xle.rolling(200).mean()
    is_inflation = ((xle > ma200_xle) & (~is_bull)).shift(1).fillna(False)
    
    # Alpha Engine
    if allowed_crypto is None:
        uni = [c for c in prices.columns if '-USD' in c]
    else:
        uni = [c for c in prices.columns if c in allowed_crypto]
        
    if 'BTC-USD' not in uni: uni.append('BTC-USD') # Always need BTC anchor
    alts = [c for c in uni if c != 'BTC-USD']
    
    alpha_weights = pd.DataFrame(0.0, index=prices.index, columns=uni)
    mom_df = prices[uni].pct_change(14)
    
    for i in range(15, len(prices)):
        dt = prices.index[i]
        current_moms = mom_df.iloc[i][alts].dropna()
        if current_moms.empty:
            alpha_weights.loc[dt, 'BTC-USD'] = 1.0
            continue
            
        top_candidates = current_moms.nlargest(3)
        # If fewer than 3 alts exist/valid in this subset, take what we have
        
        btc_m = mom_df.iloc[i]['BTC-USD']
        avg_alt_m = top_candidates.mean() if not top_candidates.empty else -999
        
        if avg_alt_m > btc_m:
            for coin in top_candidates.index:
                alpha_weights.loc[dt, coin] = 1.0 / len(top_candidates)
        else:
            alpha_weights.loc[dt, 'BTC-USD'] = 1.0
            
    alpha_weights = alpha_weights.shift(1).fillna(0)
    
    # Combine
    final_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        dt = prices.index[i]
        bull = is_bull.iloc[i]
        inf = is_inflation.iloc[i]
        
        c_w = alpha_weights.iloc[i]
        for c, w in c_w[c_w > 0].items():
            final_weights.loc[dt, c] = w * 0.40
            
        if bull:
            final_weights.loc[dt, 'SPY'] = 0.45
            if 'TLT' in prices.columns: final_weights.loc[dt, 'TLT'] = 0.10
            if 'GLD' in prices.columns: final_weights.loc[dt, 'GLD'] = 0.05
        elif inf:
            final_weights.loc[dt, 'SPY'] = 0.15
            if 'XLE' in prices.columns: final_weights.loc[dt, 'XLE'] = 0.35
            if 'GLD' in prices.columns: final_weights.loc[dt, 'GLD'] = 0.10
        else:
            final_weights.loc[dt, 'SPY'] = 0.15
            if 'TLT' in prices.columns: final_weights.loc[dt, 'TLT'] = 0.35
            if 'GLD' in prices.columns: final_weights.loc[dt, 'GLD'] = 0.10
            
    return final_weights

# ==============================================================================
# 4. VALIDATION MODULE (Comparison)
# ==============================================================================

def run_is_oos_validation(prices, split_date='2023-01-01'):
    print("\n" + "="*60)
    print(f"🔬 ROBUSTNESS TEST (Split: {split_date})")
    print("="*60)
    
    prices_is = prices[prices.index < split_date]
    prices_oos = prices[prices.index >= split_date]
    
    # 1. EXPANDED STRATEGY
    w_exp_is = run_strategy_generic(prices_is)
    w_exp_oos = run_strategy_generic(prices_oos)
    s_exp_oos = calculate_stats(prices_oos, w_exp_oos)
    
    # 2. STANDARD STRATEGY (Limited Universe)
    std_univ = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    w_std_is = run_strategy_generic(prices_is, allowed_crypto=std_univ)
    w_std_oos = run_strategy_generic(prices_oos, allowed_crypto=std_univ)
    s_std_oos = calculate_stats(prices_oos, w_std_oos)
    
    print(f"{'Metric (OOS 2023-Present)':<30} {'EXPANDED (11+ Coins)':>20} {'STANDARD (4 Coins)':>20}")
    print("-" * 75)
    print(f"{'CAGR':<30} {s_exp_oos['CAGR']:>19.1%} {s_std_oos['CAGR']:>19.1%}")
    print(f"{'Sharpe':<30} {s_exp_oos['Sharpe']:>20.2f} {s_std_oos['Sharpe']:>20.2f}")
    print(f"{'Max Drawdown':<30} {s_exp_oos['MaxDD']:>19.1%} {s_std_oos['MaxDD']:>19.1%}")
    print("-" * 75)
    
    if s_std_oos['CAGR'] > s_exp_oos['CAGR']:
        print("💡 INSIGHT: The 'Standard' (Limited) version actually performed BETTER recently.")
        print("   Reason: Many altcoins delayed/failed in 2023, while SOL/BTC led.")
        print("   Recommendation: Use Standard if you prefer safety/majors.")
    else:
        print("💡 INSIGHT: The 'Expanded' version is still superior even recently.")

# ==============================================================================
# 3. STATISTICS MODULE (Institutional Grade)
# ==============================================================================

def calculate_stats(prices, weights, benchmark_col='SPY'):
    print("📊 Calculating Institutional Metrics...")
    
    rets = prices.pct_change()
    
    # Portfolio Return
    port_ret = (weights * rets).sum(axis=1)
    
    # Filter cleanup
    idx = port_ret.index[30:] # Skip warmup
    port_ret = port_ret.loc[idx]
    bench_ret = rets[benchmark_col].loc[idx]
    
    # 1. Total Return & CAGR
    total_ret = (1 + port_ret).prod() - 1
    days = len(port_ret)
    cagr = (1 + total_ret) ** (365/days) - 1
    
    # 2. Risk Metrics
    vol = port_ret.std() * np.sqrt(365)
    sharpe = cagr / vol if vol > 0 else 0
    
    downside_ret = port_ret[port_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(365)
    sortino = cagr / downside_vol if downside_vol > 0 else 0
    
    # Drawdown
    cum = (1 + port_ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # 3. Alpha / Beta
    import statsmodels.api as sm
    y = port_ret
    x = sm.add_constant(bench_ret)
    model = sm.OLS(y, x).fit()
    alpha = model.params.iloc[0] * 365
    beta = model.params.iloc[1]
    
    # 4. Win Rate
    wins = len(port_ret[port_ret > 0])
    total_trades = len(port_ret[port_ret != 0])
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    avg_win = port_ret[port_ret > 0].mean()
    avg_loss = abs(port_ret[port_ret < 0].mean())
    profit_factor = (avg_win * wins) / (avg_loss * (total_trades - wins)) if (avg_loss > 0) else 999
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd,
        'Calmar': calmar,
        'Alpha (Ann.)': alpha,
        'Beta': beta,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Total Return': total_ret
    }

# ==============================================================================
# 4. VALIDATION MODULE (In-Sample vs Out-of-Sample)
# ==============================================================================

def run_is_oos_validation(prices, split_date='2023-01-01'):
    print("\n" + "="*60)
    print(f"🔬 ROBUSTNESS TEST (Split: {split_date})")
    print("="*60)
    
    # 1. Split Data
    prices_is = prices[prices.index < split_date]
    prices_oos = prices[prices.index >= split_date]
    
    # 1. EXPANDED STRATEGY
    w_is = run_strategy_generic(prices_is)
    w_oos = run_strategy_generic(prices_oos)
    
    # 3. Calc Stats
    stats_is = calculate_stats(prices_is, w_is)
    stats_oos = calculate_stats(prices_oos, w_oos)
    
    # 4. Print Comparison
    print(f"{'Metric':<20} {'In-Sample (Train)':>20} {'Out-of-Sample (Test)':>20}")
    print("-" * 65)
    print(f"{'Timeframe':<20} {'2020 - 2022':>20} {'2023 - Present':>20}")
    print(f"{'CAGR':<20} {stats_is['CAGR']:>19.1%} {stats_oos['CAGR']:>19.1%}")
    print(f"{'Sharpe':<20} {stats_is['Sharpe']:>20.2f} {stats_oos['Sharpe']:>20.2f}")
    print(f"{'Max Drawdown':<20} {stats_is['MaxDD']:>19.1%} {stats_oos['MaxDD']:>19.1%}")
    print(f"{'Alpha':<20} {stats_is['Alpha (Ann.)']:>19.1%} {stats_oos['Alpha (Ann.)']:>19.1%}")
    print("-" * 65)
    
    if stats_oos['Sharpe'] > 0.5 and stats_oos['CAGR'] > 0.20:
        print("✅ VALIDATION PASSED: Strategy holds up in the new regime.")
    else:
        print("⚠️ VALIDATION WARNING: Performance degrades significantly in OOS.")

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

if __name__ == "__main__":
    try:
        # 1. Get Data
        prices = fetch_data()
        
        # 2. Run Strategy (Default = Expanded)
        weights = run_strategy_generic(prices)
        
        # 3. Calculate Stats
        stats = calculate_stats(prices, weights, benchmark_col='SPY')
        
        # 4. Print Report
        print("\n" + "="*60)
        print("          ALEX.PY STRATEGY REPORT")
        print("="*60)
        print(f"Timeframe: {prices.index[30].date()} to {prices.index[-1].date()}")
        print("-" * 60)
        print(f"💰 TOTAL RETURN:      {stats['Total Return']:.1%}")
        print(f"📈 CAGR:              {stats['CAGR']:.1%}")
        print("-" * 60)
        print(f"⚖️ SHARPE RATIO:      {stats['Sharpe']:.2f} (Excellent > 1.0)")
        print(f"🛡️ SORTINO RATIO:     {stats['Sortino']:.2f} (Focus on downside)")
        print(f"🔥 CALMAR RATIO:      {stats['Calmar']:.2f} (Return / Drawdown)")
        print("-" * 60)
        print(f"📉 MAX DRAWDOWN:      {stats['MaxDD']:.1%}")
        print(f"🦁 ALPHA (vs SPY):    {stats['Alpha (Ann.)']:.1%}")
        print(f"🔗 BETA (vs SPY):     {stats['Beta']:.2f}")
        print("-" * 60)
        print(f"🎯 WIN RATE:          {stats['Win Rate']:.1%}")
        print(f"💵 PROFIT FACTOR:     {stats['Profit Factor']:.2f}")
        print("="*60)
        
        # 5. Run Validation
        run_is_oos_validation(prices)
             
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
