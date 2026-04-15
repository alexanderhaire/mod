"""
Alternative & Niche Alpha Hunt
===============================

Testing 5 niche market hypotheses:
1. Prediction Markets (DJT/Event Volatility)
2. Specialized REITs (Data Centers vs Housing)
3. Carbon Credits (KRBN Trend)
4. Water Scarcity (PHO Seasonality)
5. Soft Commodities (Corn/Wheat Trend)

Rigorous testing with Active IR.

RUN: python alternative_alpha_hunt.py
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

def fetch_alternative_data():
    print("📊 Fetching alternative universe (Carbon, Water, REITs, Prediction)...")
    
    # Prediction / Event Proxies
    pred = ['DJT'] # Volatility of DJT as proxy for election prediction volume?
    # Specialized REITs
    reits = ['EQIX', 'DLR', 'REZ', 'VNQ']
    # Carbon
    carbon = ['KRBN']
    # Water
    water = ['PHO', 'CGW']
    # Softs
    softs = ['CORN', 'WEAT', 'SOYB', 'DBA']
    # Bench
    bench = ['SPY', 'QQQ']
    
    tickers = pred + reits + carbon + water + softs + bench
    
    # KRBN started 2020. DJT recent (SPAC).
    # Need consistent data. Use 2021 start.
    data = yf.download(tickers, start='2021-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    
    # Clean NaN columns if whole history missing (DJT before merger?)
    prices = prices.dropna(axis=1, how='all')
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# SIGNALS
# =============================================================================

def prediction_market_proxy_signal(prices, lookback=5):
    """
    Hypothesis: DJT volatility predicts market turbulence or specific regime.
    Signal: If DJT Momentum > SPY Momentum -> Long Vol / Short Market?
    Actually, let's test if DJT acts as a hedge or distinct alpha.
    Signal: Trend Following on prediction asset itself.
    """
    if 'DJT' not in prices.columns:
        return None
        
    djt_mom = prices['DJT'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if djt_mom.iloc[i] > 0:
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = -1
            
    return signal

def reit_arbitrage_signal(prices, lookback=60):
    """
    Hypothesis: Data Centers (DLR/EQIX) outperform Housing (REZ) in tech cycles.
    Signal: Long Data Centers / Short REZ when QQQ > SPY (Tech leading)
    """
    dc_assets = ['EQIX', 'DLR']
    dc_assets = [a for a in dc_assets if a in prices.columns]
    
    if not dc_assets or 'REZ' not in prices.columns or 'QQQ' not in prices.columns or 'SPY' not in prices.columns:
        return None
        
    qqq_mom = prices['QQQ'].pct_change(lookback)
    spy_mom = prices['SPY'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if qqq_mom.iloc[i] > spy_mom.iloc[i]:
            signal.iloc[i] = 1 # Tech cycle -> Long DC, Short REZ
        else:
            signal.iloc[i] = -1 # Mean reversion? Or value cycle? -> Long REZ
            
    return signal

def carbon_trend_signal(prices, lookback=50):
    """
    Hypothesis: Carbon trends are regulatory driven and persistent.
    """
    if 'KRBN' not in prices.columns:
        return None
        
    sma = prices['KRBN'].rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if prices['KRBN'].iloc[i] > sma.iloc[i]:
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = 0 # Cash
            
    return signal

def water_scarcity_signal(prices):
    """
    Hypothesis: Water stocks outperform in Summer (Q2/Q3).
    """
    if 'PHO' not in prices.columns:
        return None
        
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(len(prices)):
        month = prices.index[i].month
        if month in [5, 6, 7, 8]: # May-Aug
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = 0 # Cash (or just hold SPY?)
            
    return signal

def soft_commodity_trend(prices, lookback=100):
    """
    Hypothesis: Softs trend on weather/supply shocks.
    """
    softs = ['CORN', 'WEAT', 'SOYB']
    softs = [a for a in softs if a in prices.columns]
    
    if not softs:
        return None
        
    signal = pd.DataFrame(0.0, index=prices.index, columns=softs)
    
    for asset in softs:
        sma = prices[asset].rolling(lookback).mean()
        for i in range(lookback, len(prices)):
            if prices[asset].iloc[i] > sma.iloc[i]:
                signal[asset].iloc[i] = 1
                
    return signal

# =============================================================================
# STRATEGIES
# =============================================================================

def apply_prediction_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            if 'DJT' in prices.columns:
                weights.iloc[i]['DJT'] = 1.0 # Very risky!
                
    return weights.shift(1).fillna(0)

def apply_reit_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    dc_assets = ['EQIX', 'DLR']
    dc_assets = [a for a in dc_assets if a in prices.columns]
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            for d in dc_assets:
                weights.iloc[i][d] = 1.0 / len(dc_assets) # Long DC
        elif s == -1:
            if 'REZ' in prices.columns:
                 weights.iloc[i]['REZ'] = 1.0 # Long Housing
                 
    return weights.shift(1).fillna(0)

def apply_carbon_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            if 'KRBN' in prices.columns:
                weights.iloc[i]['KRBN'] = 1.0
                
    return weights.shift(1).fillna(0)

def apply_water_strategy(prices, signal):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        if s == 1:
            if 'PHO' in prices.columns:
                weights.iloc[i]['PHO'] = 1.0
        else:
            if 'SPY' in prices.columns:
                weights.iloc[i]['SPY'] = 1.0 # Rotate to SPY in winter
                
    return weights.shift(1).fillna(0)

def apply_softs_strategy(prices, signal_df):
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
    ann_ret = strat_ret.mean() * 252
    
    if active.std() == 0:
        ir = 0
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
    print("🔮 ALTERNATIVE / NICHE ALPHA HUNT")
    print("="*60)
    
    prices = fetch_alternative_data()
    b_spy = strategy_benchmark(prices, 'SPY')
    b_reit = strategy_benchmark(prices, 'VNQ') # REIT bench
    
    print("\n TESTING HYPOTHESES...")
    results = []
    
    # 1. Prediction/DJT (Bench: SPY - high beta?)
    if 'DJT' in prices.columns:
        s_pred = prediction_market_proxy_signal(prices)
        w_pred = apply_prediction_strategy(prices, s_pred)
        m_pred = compute_metrics_and_stats(prices, w_pred, b_spy)
        if m_pred: results.append(({'Name': 'Prediction Proxy (DJT)', **m_pred}))
    
    # 2. REIT Arb (Bench: VNQ)
    s_reit = reit_arbitrage_signal(prices)
    w_reit = apply_reit_strategy(prices, s_reit)
    m_reit = compute_metrics_and_stats(prices, w_reit, b_reit)
    if m_reit: results.append(({'Name': 'REIT Tech Arb (DC vs Resi)', **m_reit}))
    
    # 3. Carbon (Bench: SPY - or Energy?)
    if 'KRBN' in prices.columns:
        s_carb = carbon_trend_signal(prices)
        w_carb = apply_carbon_strategy(prices, s_carb)
        m_carb = compute_metrics_and_stats(prices, w_carb, b_spy)
        if m_carb: results.append(({'Name': 'Carbon Trend (KRBN)', **m_carb}))
    
    # 4. Water (Bench: SPY)
    s_water = water_scarcity_signal(prices)
    w_water = apply_water_strategy(prices, s_water)
    m_water = compute_metrics_and_stats(prices, w_water, b_spy)
    if m_water: results.append(({'Name': 'Water Seasonality (PHO)', **m_water}))
    
    # 5. Softs (Bench: SPY)
    s_softs = soft_commodity_trend(prices)
    w_softs = apply_softs_strategy(prices, s_softs)
    m_softs = compute_metrics_and_stats(prices, w_softs, b_spy)
    if m_softs: results.append(({'Name': 'Softs Trend (Corn/Wheat)', **m_softs}))
    
    # PRINT RESULTS
    print(f"\n{'Name':<30} {'Sharpe':<8} {'Active IR':<10} {'p-value':<10} {'Sig'}")
    print("-" * 70)
    
    for r in results:
        sig = "✅" if r['p-value'] < 0.05 and r['Active IR'] > 0 else "❌" 
        print(f"{r['Name']:<30} {r['Sharpe']:<8.2f} {r['Active IR']:<10.2f} {r['p-value']:<10.3f} {sig}")


    print("\n" + "=" * 60)
