"""
Advanced Crypto Signal Hunt
============================

Testing 5 specific hypotheses for crypto edge:
1. Meme Canary (DOGE/SHIB leading indicator)
2. Correlation Decoupling (Buying when BTC uncorrelated to SPY)
3. ETH/BTC Mean Reversion (Trading the ratio)
4. Infrastructure Divergence (COIN/MARA vs BTC)
5. Volatility Squeeze (Breakout from low vol)

Rigorous testing with Active IR and p-values.

RUN: python advanced_crypto_hunt.py
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

def fetch_advanced_data():
    print("📊 Fetching extended universe (Crypto + Infra + Memes)...")
    
    # Core
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    memes = ['DOGE-USD', 'SHIB-USD']
    equity_infra = ['COIN', 'MARA']
    trad = ['SPY']
    
    tickers = crypto + memes + equity_infra + trad
    
    # Need recent history for SHIB/COIN (COIN IPO 2021)
    data = yf.download(tickers, start='2021-04-15', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    
    # Align all to common index
    prices = prices.dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# SIGNALS
# =============================================================================

def meme_canary_signal(prices, lookback=7):
    """
    Hypothesis: Retail mania starts in memes before major alts.
    Signal: Long Crypto when Meme Momentum > BTC Momentum
    """
    if 'DOGE-USD' not in prices.columns or 'BTC-USD' not in prices.columns:
        return None
        
    meme_prices = prices['DOGE-USD']
    # If SHIB exists, average them
    if 'SHIB-USD' in prices.columns:
        # Normalize to start at 1
        d = prices['DOGE-USD'] / prices['DOGE-USD'].iloc[0]
        s = prices['SHIB-USD'] / prices['SHIB-USD'].iloc[0]
        meme_prices = (d + s) / 2
        
    meme_mom = meme_prices.pct_change(lookback)
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        # If Memes are flying but BTC is lagging -> Bullish for broader crypto (retail is here)
        if meme_mom.iloc[i] > btc_mom.iloc[i] * 1.5 and meme_mom.iloc[i] > 0:
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = 0
            
    return signal

def correlation_decoupling_signal(prices, lookback=60):
    """
    Hypothesis: Crypto Alpha is highest when uncorrelated to SPY.
    Signal: Long BTC when corr(BTC, SPY) < 0.2
    """
    if 'SPY' not in prices.columns or 'BTC-USD' not in prices.columns:
        return None
        
    btc_ret = prices['BTC-USD'].pct_change()
    spy_ret = prices['SPY'].pct_change()
    
    corr = btc_ret.rolling(lookback).corr(spy_ret)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        c = corr.iloc[i]
        if c < 0.1: # Decoupled / Hedge behavior
            signal.iloc[i] = 1
        elif c > 0.6: # Highly correlated risk asset
            signal.iloc[i] = -1 # Avoid/Hedge
        else:
            signal.iloc[i] = 0
            
    return signal

def eth_btc_mean_reversion(prices, lookback=90):
    """
    Hypothesis: ETH/BTC ratio mean reverts.
    Signal: Long ETH when Ratio < Z-score -1.5, Long BTC when Ratio > Z-score 1.5
    """
    if 'ETH-USD' not in prices.columns or 'BTC-USD' not in prices.columns:
        return None
        
    ratio = prices['ETH-USD'] / prices['BTC-USD']
    z_score = (ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        z = z_score.iloc[i]
        
        if z < -1.5:
            signal.iloc[i] = 1 # Long ETH (Cheap vs BTC)
        elif z > 1.5:
            signal.iloc[i] = -1 # Long BTC (ETH exp vs BTC)
        else:
            signal.iloc[i] = 0 # Neutral
            
    return signal

def infra_divergence_signal(prices, lookback=14):
    """
    Hypothesis: COIN/MARA are high-beta proxies. If they outperform BTC, it signals insti flow.
    Signal: Long BTC if COIN momentum > BTC momentum
    """
    if 'COIN' not in prices.columns or 'BTC-USD' not in prices.columns:
        return None
        
    coin_mom = prices['COIN'].pct_change(lookback)
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if coin_mom.iloc[i] > btc_mom.iloc[i]:
             signal.iloc[i] = 1 # Insti buying infra -> Good for spot
        else:
             signal.iloc[i] = -1 # Infra lagging -> bad sign
             
    return signal

def volatility_squeeze_signal(prices, lookback=20):
    """
    Hypothesis: Low vol leads to expansion.
    Signal: If Vol < 10th percentile, Buy. (Direction?? Usually assumes Long biased asset)
    """
    if 'BTC-USD' not in prices.columns:
        return None
        
    ret = prices['BTC-USD'].pct_change()
    vol = ret.rolling(lookback).std()
    
    # Percentile relative to history (rolling 1 year)
    vol_rank = vol.rolling(252).rank(pct=True)
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(252, len(prices)):
        if vol_rank.iloc[i] < 0.10: # Squeeze
            signal.iloc[i] = 1
        elif vol_rank.iloc[i] > 0.90: # Climax?
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = 0
            
    return signal

# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_btc_buy_hold(prices):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights['BTC-USD'] = 1.0
    return weights.shift(1).fillna(0)

def apply_signal(prices, signal, target_asset='BTC-USD', secondary_asset=None):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        s = signal.iloc[i]
        
        if s == 1:
            if target_asset == 'ETH_BTC_PAIR': # Special case for mean reversion
                weights.iloc[i]['ETH-USD'] = 1.0
            else:
                weights.iloc[i][target_asset] = 1.0
        elif s == -1:
             if target_asset == 'ETH_BTC_PAIR':
                 weights.iloc[i]['BTC-USD'] = 1.0
             elif secondary_asset:
                 weights.iloc[i][secondary_asset] = 1.0
             else:
                 weights.iloc[i][target_asset] = 0.0 # Cash
        else:
             # Neutral - what to do? Cash or 50%? Let's say Cash for purity of signal test
             weights.iloc[i][target_asset] = 0.0
             
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def compute_metrics_and_stats(prices, weights, benchmark_weights):
    # Returns
    returns = prices.pct_change().fillna(0)
    
    strat_ret = (weights.shift(1) * returns).sum(axis=1)
    bench_ret = (benchmark_weights.shift(1) * returns).sum(axis=1)
    
    # Align
    common = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    
    # Active
    active = strat_ret - bench_ret
    
    if len(active) < 30: return None
    
    # Metrics
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
    ann_ret = strat_ret.mean() * 252
    
    # Significance
    t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    ir = active.mean() / active.std() * np.sqrt(252)
    
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
    print("🚀 ADVANCED CRYPTO SIGNAL HUNT")
    print("="*60)
    
    prices = fetch_advanced_data()
    bench_weights = strategy_btc_buy_hold(prices)
    
    print("\n TESTING HYPOTHESES...")
    
    results = []
    
    # 1. Meme Canary
    s_meme = meme_canary_signal(prices)
    w_meme = apply_signal(prices, s_meme, 'BTC-USD') # Signal says Buy Crypto
    m_meme = compute_metrics_and_stats(prices, w_meme, bench_weights)
    results.append(({'Name': 'Meme Canary', **m_meme}))

    # 2. Correlation Decoupling
    s_corr = correlation_decoupling_signal(prices)
    w_corr = apply_signal(prices, s_corr, 'BTC-USD')
    m_corr = compute_metrics_and_stats(prices, w_corr, bench_weights)
    results.append(({'Name': 'Correlation Decoupling', **m_corr}))
    
    # 3. ETH/BTC Reversion
    s_eth = eth_btc_mean_reversion(prices)
    w_eth = apply_signal(prices, s_eth, 'ETH_BTC_PAIR') # Swaps
    m_eth = compute_metrics_and_stats(prices, w_eth, bench_weights) # Bench is BTC buy hold
    results.append(({'Name': 'ETH/BTC Reversion', **m_eth}))
    
    # 4. Infra Divergence
    s_infra = infra_divergence_signal(prices)
    w_infra = apply_signal(prices, s_infra, 'BTC-USD')
    m_infra = compute_metrics_and_stats(prices, w_infra, bench_weights)
    results.append(({'Name': 'Infra Divergence (COIN)', **m_infra}))
    
    # 5. Vol Squeeze
    s_vol = volatility_squeeze_signal(prices)
    w_vol = apply_signal(prices, s_vol, 'BTC-USD')
    m_vol = compute_metrics_and_stats(prices, w_vol, bench_weights)
    results.append(({'Name': 'Vol Squeeze', **m_vol}))
    
    # PRINT RESULTS
    print(f"\n{'Name':<25} {'Sharpe':<8} {'Active IR':<10} {'p-value':<10} {'Sig'}")
    print("-" * 65)
    
    for r in results:
        sig = "✅" if r['p-value'] < 0.05 else "⚠️" if r['p-value'] < 0.10 else "❌" 
        print(f"{r['Name']:<25} {r['Sharpe']:<8.2f} {r['Active IR']:<10.2f} {r['p-value']:<10.3f} {sig}")


    print("\n" + "=" * 60)

