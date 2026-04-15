"""
Correlation Alpha Hunt (The Crisis Signal)
==========================================

Testing Stock-Bond Correlation Regimes.
Hypothesis:
- Negative Correlation (Normal): Diversification works. Leverage is safe.
- Positive Correlation (Crisis): Diversification fails (Inflation/Liquidity). Cash is King.

Strategy:
- Calculate Rolling 60-day Correlation(SPY, TLT).
- If Corr < 0.2: Risk ON (1.5x SPY).
- If Corr > 0.4: Risk OFF (Cash).
- Else: Neutral (1.0x SPY).

RUN: python correlation_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA
# =============================================================================

def fetch_data():
    print("🤝 Fetching SPY & TLT Data...")
    tickers = ['SPY', 'TLT']
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
     # Robust extraction
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    return prices

# =============================================================================
# 2. ANALYSIS
# =============================================================================

def backtest_correlation(prices):
    if 'SPY' not in prices.columns or 'TLT' not in prices.columns:
        print("   ❌ Missing Data")
        return
        
    spy = prices['SPY']
    tlt = prices['TLT']
    
    spy_ret = spy.pct_change()
    tlt_ret = tlt.pct_change()
    
    # Rolling Correlation
    window = 60
    rolling_corr = spy_ret.rolling(window).corr(tlt_ret).shift(1) # Avg correlation leading up to today
    
    # Dropna
    valid_idx = rolling_corr.dropna().index
    spy_ret = spy_ret.loc[valid_idx]
    rolling_corr = rolling_corr.loc[valid_idx]
    
    print(f"   Avg Correlation: {rolling_corr.mean():.2f}")
    
    # Strategy
    # Regimes
    safe_zone = rolling_corr < 0.2
    danger_zone = rolling_corr > 0.4
    
    # Position Scaling
    pos = pd.Series(1.0, index=valid_idx) # Default 1x
    pos[safe_zone] = 1.0 # Or 1.5 leverage? Let's test 1.0 vs 0.0 first (Defensive)
    # Actually, hypothesis is: Positive Corr = Danger.
    pos[danger_zone] = 0.0 # Cash
    
    # Strategy Returns
    strat_ret = pos * spy_ret # pos is shift(1) based effectively since rolling_corr is shift(1) applied?
    # Wait, rolling_corr is calculated using data UP TO T.
    # If we use rolling_corr(T) to trade T+1:
    # We should shift signals.
    # In my code: rolling_corr = ...shift(1). So it represents correlation of T-60 to T-1 available at Open T? 
    # Yes.
    
    strat_ret = strat_ret.dropna()
    bh_ret = spy_ret.dropna()
    
    # Leverage Version (1.5x in Safe)
    pos_lev = pos.copy()
    pos_lev[safe_zone] = 1.5
    strat_lev_ret = (pos_lev * spy_ret) - (pos_lev[safe_zone]-1.0)*(0.04/252) # Cost of borrowing check? Ignore for now
    
    # Metrics
    def get_stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        return ann, sharpe, dd
        
    b_ann, b_sharpe, b_tot = get_stats(bh_ret)
    s_ann, s_sharpe, s_tot = get_stats(strat_ret)
    l_ann, l_sharpe, l_tot = get_stats(strat_lev_ret)
    
    print("\n🔗 CORRELATION REGIME RESULTS (2005-Present):")
    print(f"   Buy & Hold SPY: Ann {b_ann:.1%} | Sharpe {b_sharpe:.2f} | Total {b_tot:.0%}")
    print(f"   Crisis Avoider (Cash if Corr > 0.4): Ann {s_ann:.1%} | Sharpe {s_sharpe:.2f} | Total {s_tot:.0%}")
    print(f"   Smart Leverage (1.5x if Corr < 0.2): Ann {l_ann:.1%} | Sharpe {l_sharpe:.2f} | Total {l_tot:.0%}")
    
    if s_sharpe > b_sharpe + 0.1:
        print("   ✅ EDGE FOUND: Avoiding Positive Correlation Regimes works!")
    else:
        print("   ❌ NO EDGE: Correlation regimes don't predict crashes reliably.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🤝 CORRELATION ALPHA HUNT")
    print("="*60)
    
    prices = fetch_data()
    backtest_correlation(prices)
    
    print("\n" + "=" * 60)
