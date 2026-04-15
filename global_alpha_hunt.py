"""
Global Alpha Hunt
==================

Testing 4 global market hypotheses:
1. DeFi Rotation (UNI/AAVE/MKR/LDO vs ETH)
2. Hard Money Cycle (Gold vs BTC)
3. Global Liquidity (Dollar vs EM/China)
4. Commodity Trends (Oil/Ag vs Inflation)

Rigorous testing with Active IR.

RUN: python global_alpha_hunt.py
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

def fetch_global_data():
    print("📊 Fetching global universe (DeFi, Commodities, Forex, EM)...")
    
    # DeFi Blue Chips
    defi = ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'LDO-USD']
    # Core Crypto
    crypto = ['BTC-USD', 'ETH-USD']
    # Hard Money
    hard_money = ['GLD']
    # Global Macro
    macro = ['UUP', 'FXI', 'EEM']
    # Commodities
    commodities = ['USO', 'DBA', 'TIP'] # TIP as inflation proxy
    # Benchmark
    bench = ['SPY']
    
    tickers = defi + crypto + hard_money + macro + commodities + bench
    
    # LDO is newest (start 2021/2022 approx), UNI from 2020.
    # Start mid-2021 to capture all
    data = yf.download(tickers, start='2021-06-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    prices = prices.dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# SIGNALS
# =============================================================================

def defi_rotation_signal(prices, lookback=14):
    """
    Hypothesis: DeFi moves independently of ETH.
    Signal: Long DeFi Basket if Average DeFi Mom > ETH Mom
    """
    defi_assets = ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'LDO-USD']
    # Filter for assets present in data
    defi_assets = [a for a in defi_assets if a in prices.columns]
    
    if not defi_assets or 'ETH-USD' not in prices.columns:
        return None
        
    eth_mom = prices['ETH-USD'].pct_change(lookback)
    
    # Basket momentum
    basket_mom = pd.Series(0.0, index=prices.index)
    for asset in defi_assets:
        basket_mom += prices[asset].pct_change(lookback)
    basket_mom /= len(defi_assets)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if basket_mom.iloc[i] > eth_mom.iloc[i] * 1.1: # 10% buffer
            signal.iloc[i] = 1
        elif basket_mom.iloc[i] < eth_mom.iloc[i]:
            signal.iloc[i] = -1 # Rotate back to ETH (or cash if long-only logic)
        else:
            signal.iloc[i] = 0
            
    return signal

def hard_money_signal(prices, lookback=60):
    """
    Hypothesis: Capital flows between Gold and BTC.
    Signal: Long winner of relative momentum.
    """
    if 'GLD' not in prices.columns or 'BTC-USD' not in prices.columns:
        return None
        
    gld_mom = prices['GLD'].pct_change(lookback)
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        # Normalize volatility difference? BTC moves 5x GLD.
        # Simple relative strength:
        if btc_mom.iloc[i] > gld_mom.iloc[i]:
            signal.iloc[i] = 1 # BTC
        else:
            signal.iloc[i] = -1 # GLD
            
    return signal

def global_liquidity_signal(prices, lookback=20):
    """
    Hypothesis: Weak Dollar (UUP) -> Long Emerging Markets (EEM/FXI).
    """
    if 'UUP' not in prices.columns or 'EEM' not in prices.columns:
        return None
        
    uup_ret = prices['UUP'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if uup_ret.iloc[i] < -0.01: # Dollar weakening > 1%
            signal.iloc[i] = 1 # Long EM
        elif uup_ret.iloc[i] > 0.01: # Dollar strengthening
            signal.iloc[i] = -1 # Cash/Short
        else:
            signal.iloc[i] = 0
            
    return signal

def commodity_trend_signal(prices, lookback=100):
    """
    Hypothesis: Commodities trend when Inflation (TIP) is rising.
    Signal: Long USO/DBA if Prices > SMA(100) AND TIP > SMA(50)
    """
    comm_assets = ['USO', 'DBA']
    comm_assets = [a for a in comm_assets if a in prices.columns]
    
    if not comm_assets or 'TIP' not in prices.columns:
        return None
        
    tip_sma = prices['TIP'].rolling(50).mean()
    
    signal = pd.DataFrame(0.0, index=prices.index, columns=comm_assets)
    
    for asset in comm_assets:
        sma = prices[asset].rolling(lookback).mean()
        for i in range(max(lookback, 50), len(prices)):
            # Condition 1: Commodity Uptrend
            # Condition 2: Inflation Support (TIP > SMA -- crude proxy for real rates/inf expectations)
            if prices[asset].iloc[i] > sma.iloc[i] and prices['TIP'].iloc[i] > tip_sma.iloc[i]:
                signal[asset].iloc[i] = 1
    
    return signal

# =============================================================================
# STRATEGIES
# =============================================================================

def apply_defi_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    defi_assets = ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'LDO-USD']
    defi_assets = [a for a in defi_assets if a in prices.columns]
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            for d in defi_assets:
                weights.iloc[i][d] = 1.0 / len(defi_assets)
        else:
            # Fallback to ETH? Or Cash? Bench is ETH likely.
            # Let's fallback to ETH to test "Alpha over ETH"
            if 'ETH-USD' in prices.columns:
                weights.iloc[i]['ETH-USD'] = 1.0
                
    return weights.shift(1).fillna(0)

def apply_hard_money_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            if 'BTC-USD' in prices.columns:
                weights.iloc[i]['BTC-USD'] = 1.0
        else:
             if 'GLD' in prices.columns:
                 weights.iloc[i]['GLD'] = 1.0
                 
    return weights.shift(1).fillna(0)

def apply_global_macro_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    em_assets = ['EEM', 'FXI']
    em_assets = [a for a in em_assets if a in prices.columns]
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            for e in em_assets:
                weights.iloc[i][e] = 1.0 / len(em_assets)
        else:
            # Cash or short? Cash.
            pass
            
    return weights.shift(1).fillna(0)

def apply_commodity_strategy(prices, signal_df):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        active_comms = []
        for col in signal_df.columns:
            if signal_df[col].iloc[i] == 1:
                active_comms.append(col)
        
        if active_comms:
            for a in active_comms:
                weights.iloc[i][a] = 1.0 / len(active_comms)
                
    return weights.shift(1).fillna(0)

def strategy_benchmark(prices, asset):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if asset in prices.columns:
        weights[asset] = 1.0
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def compute_metrics_and_stats(prices, weights, benchmark_weights):
    returns = prices.pct_change().fillna(0)
    
    strat_ret = (weights.shift(1) * returns).sum(axis=1)
    bench_ret = (benchmark_weights.shift(1) * returns).sum(axis=1)
    
    common = strat_ret.index.intersection(bench_ret.index).drop(returns.index[:1]) # Drop first NaN
    strat_ret = strat_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    
    # Filter for active days (if strategy is mostly cash, Active IR is meaningless)
    # Actually, Active IR is valid vs benchmark even if cash.
    
    active = strat_ret - bench_ret
    
    if len(active) < 30: return None
    
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    ann_ret = strat_ret.mean() * 252
    
    if active.std() == 0:
        ir = 0
        t_stat = 0
        p_val = 1
    else:
        ir = active.mean() / active.std() * np.sqrt(252)
        t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {
        'Sharpe': sharpe,
        'Ann Ret': ann_ret,
        'Active IR': ir,
        'p-value': p_val
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🌍 GLOBAL ALPHA HUNT")
    print("="*60)
    
    prices = fetch_global_data()
    
    print("\n TESTING HYPOTHESES...")
    
    results = []
    
    # 1. DeFi Rotation (Bench: ETH)
    s_defi = defi_rotation_signal(prices)
    w_defi = apply_defi_strategy(prices, s_defi)
    b_eth = strategy_benchmark(prices, 'ETH-USD')
    m_defi = compute_metrics_and_stats(prices, w_defi, b_eth)
    if m_defi: results.append(({'Name': 'DeFi Rotation (vs ETH)', **m_defi}))
    
    # 2. Hard Money (Bench: 50/50 split? or just GLD? Let's use GLD as conservative bench)
    s_hard = hard_money_signal(prices)
    w_hard = apply_hard_money_strategy(prices, s_hard)
    b_gld = strategy_benchmark(prices, 'GLD')
    m_hard = compute_metrics_and_stats(prices, w_hard, b_gld)
    if m_hard: results.append(({'Name': 'Hard Money (vs GLD)', **m_hard}))
    
    # 3. Global Macro (Bench: SPY? or Cash? Let's use SPY as opp cost)
    s_macro = global_liquidity_signal(prices)
    w_macro = apply_global_macro_strategy(prices, s_macro)
    b_spy = strategy_benchmark(prices, 'SPY')
    m_macro = compute_metrics_and_stats(prices, w_macro, b_spy)
    if m_macro: results.append(({'Name': 'Global Macro (vs SPY)', **m_macro}))
    
    # 4. Commodities (Bench: SPY - pure diversification play)
    s_comm = commodity_trend_signal(prices)
    w_comm = apply_commodity_strategy(prices, s_comm)
    m_comm = compute_metrics_and_stats(prices, w_comm, b_spy)
    if m_comm: results.append(({'Name': 'Commodity Trend (vs SPY)', **m_comm}))

    # PRINT RESULTS
    print(f"\n{'Name':<25} {'Sharpe':<8} {'Active IR':<10} {'p-value':<10} {'Sig'}")
    print("-" * 65)
    
    for r in results:
        sig = "✅" if r['p-value'] < 0.05 else "⚠️" if r['p-value'] < 0.10 else "❌" 
        print(f"{r['Name']:<25} {r['Sharpe']:<8.2f} {r['Active IR']:<10.2f} {r['p-value']:<10.3f} {sig}")

    print("\n" + "=" * 60)
