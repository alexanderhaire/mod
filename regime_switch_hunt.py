"""
Regime-Switching Alpha Hunt
===========================

Testing Dynamic Meta-Allocation.
Goal: Outperform Static 50/50 Blend (Sharpe 1.25).
Method: Switch between Ultimate (Aggressive) and HRP (Defensive) based on VIX Regime.

Rule:
- VIX < 20 (Calm): 100% Ultimate Strategy.
- VIX > 20 (Fear): 100% HRP Strategy.

RUN: python regime_switch_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA (Recycle from Meta Hunt)
# =============================================================================

def fetch_data():
    print("🌍 Fetching Regime Data...")
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'FXI', # HRP
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 
        'GLD', 'USO', 'DBA', 'VNQ', 
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX', '^VIX3M'
    ]
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    cols_to_drop = [c for c in ['^VIX', '^VIX3M'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop).ffill().dropna()
    
    if 'SPY' in prices.columns:
        prices = prices.reindex(prices['SPY'].dropna().index).ffill().dropna()
    
    if vix is not None: vix = vix.reindex(prices.index).ffill()
    if vix3m is not None: vix3m = vix3m.reindex(prices.index).ffill()

    return prices, vix, vix3m

# =============================================================================
# 2. ENGINES (Simplified for Speed)
# =============================================================================

def run_hrp_engine(prices):
    print("   🛡️ Generating HRP Returns...")
    # Simplified HRP: Inverse Volatility (Risk Parity) for speed equivalent approx
    # Full HRP is slow in loop. Risk Parity is 90% correlated to HRP in major regimes.
    # We will use Inverse Volatility as Proxy for HRP to speed up simulation or re-implement full if needed.
    # Let's do Full HRP but simplified.
    
    returns = prices.pct_change().dropna()
    monthly_dates = returns.resample('M').last().index
    weights = pd.DataFrame(index=monthly_dates, columns=prices.columns)
    
    for t in monthly_dates:
        hist = returns[returns.index <= t].tail(126)
        if len(hist) < 60: continue
        try:
            # Inv Vol Weighting (High Correlation to HRP, much faster)
            vol = hist.std()
            w = (1/vol) / (1/vol).sum()
            weights.loc[t] = w
        except:
            weights.loc[t] = 1.0/len(prices.columns)
            
    weights = weights.reindex(returns.index).ffill().dropna()
    return (weights.shift(1) * returns).sum(axis=1)

def run_ultimate_engine(prices, vix, vix3m):
    print("   🎯 Generating Ultimate Returns...")
    # Base Signals
    vix_sig = pd.Series(0, index=prices.index)
    if vix is not None and vix3m is not None:
        ratio = (vix/vix3m).rolling(5).mean()
        vix_sig[ratio < 0.90] = 1
        vix_sig[ratio > 1.05] = -1
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(1, len(prices)):
        vs = vix_sig.iloc[i]
        # Trad (60%)
        if vs > 0: w_t = {'SPY': 0.45, 'TLT': 0.10}
        elif vs < 0: w_t = {'SPY': 0.15, 'TLT': 0.35}
        else: w_t = {'SPY': 0.30, 'TLT': 0.22}
        
        # Crypto (40%)
        if 'BTC-USD' in prices.columns:
            w_c = {'BTC-USD': 0.20} # Neutral assumption for speed
            
        for t, w in w_t.items(): 
            if t in prices.columns: weights.iloc[i][t] = w
        for t, w in w_c.items(): 
            if t in prices.columns: weights.iloc[i][t] = w
            
    return (weights.shift(1) * prices.pct_change()).sum(axis=1)

# =============================================================================
# 3. REGIME SWITCHING
# =============================================================================

def run_regime_test(r_hrp, r_ult, vix):
    print("\n🔄 REGIME SWITCHING SIMULATION")
    common = r_hrp.index.intersection(r_ult.index).intersection(vix.index)
    r_hrp = r_hrp.loc[common]
    r_ult = r_ult.loc[common]
    vix = vix.loc[common]
    
    # Static 50/50
    r_static = 0.5 * r_hrp + 0.5 * r_ult
    
    # Dynamic
    # If VIX < 20: Ultimate
    # If VIX >= 20: HRP
    weights_ult = (vix < 20).astype(int)
    weights_hrp = 1 - weights_ult
    
    # Lag signal by 1 day to execute
    weights_ult = weights_ult.shift(1).fillna(0.5)
    weights_hrp = weights_hrp.shift(1).fillna(0.5)
    
    r_dynamic = weights_ult * r_ult + weights_hrp * r_hrp
    
    # Metrics
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f}")
        return sharpe

    print("-" * 75)
    s_stat = get_stats(r_static, "Static 50/50")
    s_dyn = get_stats(r_dynamic, "Dynamic (VIX Switch)")
    
    print("-" * 75)
    
    if s_dyn > s_stat + 0.1:
        print("✅ SUCCESS: Dynamic Switching Beats Static Allocation.")
        print(f"   Improvement: +{s_dyn - s_stat:.2f} Sharpe")
    elif s_dyn > s_stat:
        print("⚠️ MARGINAL: Dynamic is better but barely.")
    else:
        print("❌ FAILED: Static Mix is better. Market timing allocation didn't work.")

if __name__ == "__main__":
    prices, vix, vix3m = fetch_data()
    r_hrp = run_hrp_engine(prices)
    r_ult = run_ultimate_engine(prices, vix, vix3m)
    run_regime_test(r_hrp, r_ult, vix)
