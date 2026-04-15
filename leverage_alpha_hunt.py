"""
Leverage Alpha Hunt (Kelly Criterion)
=====================================

Optimizing the "Money Management" of the Ultimate Strategy.
We simulate varying degrees of leverage (1x to 3x) to find the Geometric Growth Maximizer.

Method: Synthetic Leverage on Base Assets.
Assumption: Borrowing Cost = Risk Free Rate + Spread (approx 4% avg).

RUN: python leverage_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA
# =============================================================================

def fetch_data():
    print("🏎️ Fetching Data for Leverage Simulation...")
    tickers = [
        'SPY', 'TLT', 'GLD', # Trad
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX', '^VIX3M' # Signals
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
    prices = prices.drop(columns=cols_to_drop)
    
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    
    return prices, vix, vix3m

# =============================================================================
# 2. ULTIMATE STRATEGY LOGIC (Base Engine)
# =============================================================================

def run_strategy_base(prices, vix, vix3m):
    # Signals
    if vix is None or vix3m is None:
        vix_sig = pd.Series(0, index=prices.index)
    else:
        ratio = (vix / vix3m).rolling(5).mean()
        vix_sig = pd.Series(0, index=prices.index)
        vix_sig[ratio < 0.90] = 1 # Bullish
        vix_sig[ratio > 1.05] = -1 # Bearish
    
    # Crypto Altseason (Simplified based on BTC mom)
    if 'BTC-USD' in prices.columns:
        btc_mom = prices['BTC-USD'].pct_change(14)
        crypto_sig = pd.Series(0, index=prices.index)
        crypto_sig[btc_mom > 0] = 1 # Simple momentum proxy for simulation
    else:
        crypto_sig = pd.Series(0, index=prices.index)
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(1, len(prices)):
        # Trad (60%)
        vs = vix_sig.iloc[i]
        if vs > 0:
             w_t = {'SPY': 0.45, 'TLT': 0.10, 'GLD': 0.05}
        elif vs < 0:
             w_t = {'SPY': 0.15, 'TLT': 0.35, 'GLD': 0.10}
        else:
             w_t = {'SPY': 0.30, 'TLT': 0.22, 'GLD': 0.08}
             
        # Crypto (40%)
        cs = crypto_sig.iloc[i]
        if cs > 0:
             w_c = {'BTC-USD': 0.25, 'ETH-USD': 0.15} if 'ETH-USD' in prices.columns else {'BTC-USD': 0.40}
        else:
             w_c = {'BTC-USD': 0.10} # Defensive crypto
             
        # Assign
        for t, w in w_t.items(): 
            if t in prices.columns: weights.iloc[i][t] = w
        for t, w in w_c.items():
            if t in prices.columns: weights.iloc[i][t] = w
            
    # Normalize checks (should sum approx 1.0 or less)
    # weights = weights.div(weights.sum(axis=1), axis=0).fillna(0) # Don't force 100% if signal says cash
    
    return weights.shift(1).fillna(0)

# =============================================================================
# 3. LEVERAGE SIMULATION
# =============================================================================

def simulate_leverage(prices, weights, leverage_levels=[1.0, 1.5, 2.0, 3.0]):
    print("\n🏎️ CALCULATING OPTIMAL LEVERAGE...")
    
    returns = prices.pct_change().dropna()
    weights = weights.reindex(returns.index).ffill().dropna()
    
    # Base Portfolio Return (1x)
    base_port_ret = (weights * returns).sum(axis=1)
    
    cost_of_debt_annual = 0.04 # 4% borrowing cost
    cost_daily = cost_of_debt_annual / 252
    
    results = {}
    
    print(f"\n   {'Leverage':<10} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>10}")
    print("   " + "-" * 45)
    
    for lev in leverage_levels:
        # Leveraged Return = L * r - (L-1)*cost
        # Note: This assumes daily rebalancing of leverage (ETF style)
        lev_ret = lev * base_port_ret - (lev - 1) * cost_daily
        
        # Metrics
        ann = lev_ret.mean() * 252
        vol = lev_ret.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        
        cum = (1 + lev_ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        n_years = len(lev_ret) / 252
        total_ret = cum.iloc[-1] - 1
        cagr = (1 + total_ret) ** (1/n_years) - 1 if total_ret > -1 else -1.0
        
        results[lev] = {'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe}
        
        print(f"   {lev:<10.1f}x {cagr:>9.1%} {max_dd:>9.1%} {sharpe:>10.2f}")
        
    return results

def analyze_kelly(results):
    print("\n🏁 KELLY OPTIMIZATION VERDICT")
    print("-" * 60)
    
    best_cagr = -1.0
    best_lev = 1.0
    safe_lev = 1.0
    
    for lev, res in results.items():
        if res['cagr'] > best_cagr:
            best_cagr = res['cagr']
            best_lev = lev
            
        if res['max_dd'] > -0.50: # Safe zone
            safe_lev = max(safe_lev, lev)
            
    print(f"   🚀 MAXIMUM GROWTH (Kelly): {best_lev}x Leverage (CAGR {best_cagr:.1%})")
    print(f"   🛡️ SAFE LIMIT (MaxDD < 50%): {safe_lev}x Leverage")
    
    if best_lev > 2.0:
        print("\n   ⚠️ WARNING: Optimal leverage is high.")
        print("   Recommend capping at 1.5x - 2.0x for psychological survival.")
    elif best_lev > 1.0:
        print("\n   ✅ SUCCESS: Moderate leverage (1.5x - 2.0x) is the sweet spot.")
    else:
        print("\n   🛑 STOP: 1x (No Leverage) is optimal. Do not leverage.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    prices, vix, vix3m = fetch_data()
    weights_1x = run_strategy_base(prices, vix, vix3m)
    res = simulate_leverage(prices, weights_1x, leverage_levels=[1.0, 1.5, 2.0, 2.5, 3.0])
    analyze_kelly(res)
