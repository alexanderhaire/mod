"""
Supply Chain & Metals Value Hunt
=================================

Testing 4 heavy industry hypotheses:
1. Freight Super-Cycle (BDRY Trend)
2. Miners vs Metal Valuation (Long GDX / Short GLD Mean Reversion)
3. Dr. Copper (COPX leading SPY)
4. Strategic Scarcity (URA/REMX Momentum)

Rigorous testing with Active IR.

RUN: python supply_chain_hunt.py
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

def fetch_supply_chain_data():
    print("📊 Fetching supply chain universe (Freight, Miners, Metals)...")
    
    # Freight
    freight = ['BDRY']
    # Miners & Metals
    miners = ['GDX', 'COPX', 'URA', 'REMX', 'XME']
    metals = ['GLD']
    # Bench
    bench = ['SPY']
    
    tickers = freight + miners + metals + bench
    
    # BDRY inception 2018. Let's use 2019 start.
    data = yf.download(tickers, start='2019-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    prices = prices.dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# SIGNALS
# =============================================================================

def freight_trend_signal(prices, lookback=50):
    """
    Hypothesis: Freight rates trend strongly.
    Signal: Long BDRY if Price > SMA(50)
    """
    if 'BDRY' not in prices.columns:
        return None
        
    sma = prices['BDRY'].rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if prices['BDRY'].iloc[i] > sma.iloc[i]:
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = 0 # Cash
            
    return signal

def miner_value_signal(prices, lookback=252):
    """
    Hypothesis: Buyers should step in when Miners (GDX) are cheap vs Gold (GLD).
    Signal: Z-score of GDX/GLD ratio. Long GDX / Short GLD when Z < -1.5
    """
    if 'GDX' not in prices.columns or 'GLD' not in prices.columns:
        return None
        
    ratio = prices['GDX'] / prices['GLD']
    z_score = (ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        z = z_score.iloc[i]
        if z < -1.5:
            signal.iloc[i] = 1 # Long Miners, Short Gold (or just Long Miners vs neutral)
        elif z > 1.5:
            signal.iloc[i] = -1 # Short Miners, Long Gold
        else:
            signal.iloc[i] = 0
            
    return signal

def copper_leader_signal(prices, lookback=20):
    """
    Hypothesis: Copper Miners (COPX) leading indicator for SPY.
    Signal: Long SPY if COPX momentum > SPY momentum.
    """
    if 'COPX' not in prices.columns or 'SPY' not in prices.columns:
        return None
        
    copx_mom = prices['COPX'].pct_change(lookback)
    spy_mom = prices['SPY'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if copx_mom.iloc[i] > spy_mom.iloc[i]:
             signal.iloc[i] = 1 # Bullish SPY
        else:
             signal.iloc[i] = 0 # Neutral/Cash
             
    return signal

def strategic_momentum(prices, lookback=100):
    """
    Hypothesis: Strategic metals (URA, REMX) trend on geopolitics.
    Signal: Long URA/REMX if Price > SMA(100)
    """
    assets = ['URA', 'REMX']
    assets = [a for a in assets if a in prices.columns]
    
    if not assets:
        return None
        
    signal = pd.DataFrame(0.0, index=prices.index, columns=assets)
    
    for asset in assets:
        sma = prices[asset].rolling(lookback).mean()
        for i in range(lookback, len(prices)):
            if prices[asset].iloc[i] > sma.iloc[i]:
                signal[asset].iloc[i] = 1
                
    return signal

# =============================================================================
# STRATEGIES
# =============================================================================

def apply_freight_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        if signal.iloc[i] == 1:
            if 'BDRY' in prices.columns:
                weights.iloc[i]['BDRY'] = 1.0
    return weights.shift(1).fillna(0)

def apply_miner_value_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1: # Long GDX
            weights.iloc[i]['GDX'] = 1.0
        elif s == -1: # Long GLD
            weights.iloc[i]['GLD'] = 1.0
        else:
             # Neutral? Maybe 50/50 or Cash? Let's say Cash for signal purity
             pass
    return weights.shift(1).fillna(0)

def apply_copper_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        if signal.iloc[i] == 1:
            weights.iloc[i]['SPY'] = 1.0
    return weights.shift(1).fillna(0)

def apply_strategic_strategy(prices, signal_df):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        active = []
        for col in signal_df.columns:
            if signal_df[col].iloc[i] == 1:
                active.append(col)
        if active:
            for a in active:
                weights.iloc[i][a] = 1.0 / len(active)
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
    
    common = strat_ret.index.intersection(bench_ret.index).drop(returns.index[:1])
    strat_ret = strat_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    
    active = strat_ret - bench_ret
    
    if len(active) < 30: return None
    
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    
    if active.std() == 0:
        ir = 0
        p_val = 1
    else:
        ir = active.mean() / active.std() * np.sqrt(252)
        t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {
        'Sharpe': sharpe,
        'Active IR': ir,
        'p-value': p_val
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("⚓ SUPPLY CHAIN & METALS HUNT")
    print("="*60)
    
    prices = fetch_supply_chain_data()
    b_spy = strategy_benchmark(prices, 'SPY')
    b_gld = strategy_benchmark(prices, 'GLD')
    
    print("\n TESTING HYPOTHESES...")
    results = []
    
    # 1. Freight (Bench: SPY - generic risk)
    if 'BDRY' in prices.columns:
        s_freight = freight_trend_signal(prices)
        w_freight = apply_freight_strategy(prices, s_freight)
        m_freight = compute_metrics_and_stats(prices, w_freight, b_spy)
        if m_freight: results.append(({'Name': 'Freight Trend (BDRY)', **m_freight}))
        
    # 2. Miner Value (Bench: GLD - holding the metal)
    if 'GDX' in prices.columns:
        s_miner = miner_value_signal(prices)
        w_miner = apply_miner_value_strategy(prices, s_miner)
        m_miner = compute_metrics_and_stats(prices, w_miner, b_gld)
        if m_miner: results.append(({'Name': 'Miner/Metal Value (GDX/GLD)', **m_miner}))
        
    # 3. Dr Copper (Bench: SPY)
    if 'COPX' in prices.columns:
        s_copx = copper_leader_signal(prices)
        w_copx = apply_copper_strategy(prices, s_copx)
        m_copx = compute_metrics_and_stats(prices, w_copx, b_spy)
        if m_copx: results.append(({'Name': 'Dr. Copper Leader', **m_copx}))
        
    # 4. Strategic (Bench: SPY)
    if 'URA' in prices.columns:
        s_strat = strategic_momentum(prices)
        w_strat = apply_strategic_strategy(prices, s_strat)
        m_strat = compute_metrics_and_stats(prices, w_strat, b_spy)
        if m_strat: results.append(({'Name': 'Strategic Metals Trend', **m_strat}))
        
    # PRINT RESULTS
    print(f"\n{'Name':<30} {'Sharpe':<8} {'Active IR':<10} {'p-value':<10} {'Sig'}")
    print("-" * 70)
    
    for r in results:
        sig = "✅" if r['p-value'] < 0.05 and r['Active IR'] > 0 else "❌" 
        print(f"{r['Name']:<30} {r['Sharpe']:<8.2f} {r['Active IR']:<10.2f} {r['p-value']:<10.3f} {sig}")


    print("\n" + "=" * 60)
