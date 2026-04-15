"""
Risk Parity Meta-Rebalancing Hunt
=================================

Refining the Meta-Allocation Logic.
Shift from Fixed Dollar (50/50) to Equal Risk Contribution (Risk Parity).
Hypothesis: 50% Dollar != 50% Risk. Ultimate Strategy is higher vol, so it dominates.
Solution: Inverse Volatility Weighting.

Weight_i = (1 / Vol_i) / Sum(1 / Vol_j)

RUN: python risk_parity_meta_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def fetch_data():
    print("⚖️ Fetching Data for Risk Parity Lab...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'SHY', 'QQQ', # Trad
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX'
    ]
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    
    cols_to_drop = [c for c in ['^VIX'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop).ffill().dropna()
    
    if vix is not None: vix = vix.reindex(prices.index).ffill()
    print(f"   Data: {len(prices)} days")
    return prices, vix

def get_hrp_proxy(prices):
    # Proxy HRP (Inverse Vol)
    returns = prices.pct_change().dropna()
    weights = pd.DataFrame(index=returns.index, columns=prices.columns)
    lookback = 126
    m_dates = returns.resample('M').last().index
    for t in m_dates:
        hist = returns[returns.index <= t].tail(lookback)
        if len(hist) < 60: continue
        vol = hist.std()
        w = (1/vol) / (1/vol).sum()
        try: weights.loc[t] = w
        except: pass
    weights = weights.ffill().dropna()
    return (weights.shift(1) * returns).sum(axis=1)

def get_ultimate(prices, vix):
    # Ultimate Strategy logic (VIX SMA Proxy)
    vix_ma = vix.rolling(20).mean()
    vix_sig = pd.Series(0, index=prices.index)
    vix_sig[vix < vix_ma] = 1 
    vix_sig[vix > vix_ma] = -1 
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(1, len(prices)):
        vs = vix_sig.iloc[i]
        # Trad
        if vs > 0: w_t = {'SPY': 0.45, 'TLT': 0.10}
        elif vs < 0: w_t = {'SPY': 0.15, 'TLT': 0.35}
        else: w_t = {'SPY': 0.30, 'TLT': 0.22}
        # Crypto
        if 'BTC-USD' in prices.columns:
            btc = prices['BTC-USD'].iloc[i]
            prev = prices['BTC-USD'].iloc[i-1] if i > 0 else btc
            w_c = {'BTC-USD': 0.25} if btc > prev else {'BTC-USD': 0.10}
        for t, w in w_t.items(): 
             if t in prices.columns: weights.iloc[i][t] = w
        for t, w in w_c.items():
             if t in prices.columns: weights.iloc[i][t] = w
    return (weights.shift(1) * prices.pct_change()).sum(axis=1)

# =============================================================================
# 2. RISK PARITY ALLOCATION
# =============================================================================

def simulate_risk_parity(r_ult, r_hrp):
    print("\n⚖️ CALCULATING DYNAMIC RISK WEIGHTS...")
    
    # 1. Calculate Volatility (63 days / Quarterly lookback)
    # Why 63? Enough to be stable, fast enough to react to regime change.
    
    vol_ult = r_ult.rolling(63).std()
    vol_hrp = r_hrp.rolling(63).std()
    
    # Avoid zero div
    vol_ult = vol_ult.replace(0, 0.001)
    vol_hrp = vol_hrp.replace(0, 0.001)
    
    # 2. Inverse Vol Weights
    inv_ult = 1 / vol_ult
    inv_hrp = 1 / vol_hrp
    
    total_inv = inv_ult + inv_hrp
    
    w_ult = inv_ult / total_inv
    w_hrp = inv_hrp / total_inv
    
    # Shift weights (calculated at T, applied at T+1)
    w_ult = w_ult.shift(1)
    w_hrp = w_hrp.shift(1)
    
    # Drop NaNs
    df = pd.DataFrame({'Ret_Ult': r_ult, 'Ret_HRP': r_hrp, 'W_Ult': w_ult, 'W_HRP': w_hrp})
    df = df.dropna()
    
    # 3. Simulate
    df['Ret_RP'] = df['W_Ult'] * df['Ret_Ult'] + df['W_HRP'] * df['Ret_HRP']
    df['Ret_5050'] = 0.5 * df['Ret_Ult'] + 0.5 * df['Ret_HRP']
    
    # Metrics
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | Avg Ult Wgt: {df['W_Ult'].mean():.1%}")
        return sharpe

    print("-" * 80)
    s_base = get_stats(df['Ret_5050'], "Static 50/50")
    s_rp = get_stats(df['Ret_RP'], "Risk Parity (Inv Vol)")
    print("-" * 80)
    
    if s_rp > s_base + 0.05:
        print("✅ SUCCESS: Risk Parity beats Static Allocation.")
        print("   Reason: It reduced exposure to Ultimate Strategy during high-vol regimes.")
    elif s_rp > s_base:
        print("⚠️ MARGINAL: Slight improvement but adds complexity.")
    else:
        print("❌ FAILED: Static 50/50 wins. Volatility prediction didn't help.")

if __name__ == "__main__":
    prices, vix = fetch_data()
    r_ult = get_ultimate(prices, vix)
    r_hrp = get_hrp_proxy(prices)
    simulate_risk_parity(r_ult, r_hrp)
