"""
Asian Alpha Hunt
================

Testing "Weird Asian Anomalies" driving global returns.
1. The Yen Carry Trade (FXY): Predicting QQQ based on Yen Weakness.
2. The Dragon Divergence: Long India (INDA) / Short China (FXI).
3. The Korea Proxy: Long Korea (EWY) vs Semis (SOXX).

RUN: python asian_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH ASIA DATA
# =============================================================================

def fetch_asia_data():
    print("🌏 Fetching Asian Markets (Yen, China, India, Korea)...")
    tickers = ['FXY', 'QQQ', 'SPY', 'INDA', 'FXI', 'EWY', 'SOXX']
    
    data = yf.download(tickers, start='2010-01-01', progress=False)
    
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
# 2. STRATEGY: YEN CARRY SWITCH
# =============================================================================

def test_yen_carry(prices):
    """
    Hypothesis: Weak Yen (FXY Down) = Global Liquidity ON = Buy QQQ.
    Strong Yen (FXY Up) = Liquidity Crunch = Cash/Short.
    """
    print("\n💴 TESTING: Yen Carry Trade (The Master Switch)...")
    
    if 'FXY' not in prices.columns or 'QQQ' not in prices.columns:
        print("   ❌ Missing Data (FXY/QQQ)")
        return
        
    yen = prices['FXY']
    qqq = prices['QQQ']
    
    # Signal: FXY Trend
    sma = yen.rolling(50).mean()
    
    # Weak Yen = Price < SMA (Down Trend) -> Risk ON
    # Strong Yen = Price > SMA (Up Trend) -> Risk OFF
    
    risk_on = (yen < sma).astype(int)
    
    # Shift Signal
    strat_ret = risk_on.shift(1) * qqq.pct_change()
    strat_ret = strat_ret.dropna()
    
    # Benchmark (Hold QQQ)
    bh_ret = qqq.pct_change().dropna()
    
    def get_stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        return ann, sharpe
        
    s_ann, s_sharpe = get_stats(strat_ret)
    b_ann, b_sharpe = get_stats(bh_ret)
    
    print(f"   Hypothesis: Weak Yen = Bullish QQQ")
    print(f"   QQQ Hold: {b_ann:.1%} | Sharpe {b_sharpe:.2f}")
    print(f"   Yen Strat: {s_ann:.1%} | Sharpe {s_sharpe:.2f}")
    
    if s_sharpe > b_sharpe + 0.2:
        print("   ✅ EDGE FOUND: The Yen Switch works!")
    else:
        print("   ❌ NO EDGE: Yen signal doesn't beat Buying & Holding QQQ.")

# =============================================================================
# 3. STRATEGY: DRAGON DIVERGENCE (India vs China)
# =============================================================================

def test_dragon_divergence(prices):
    """
    Hypothesis: Long India (INDA) / Short China (FXI).
    Structural Demographics play.
    """
    print("\n🐉 TESTING: Dragon Divergence (Long India / Short China)...")
    
    if 'INDA' not in prices.columns or 'FXI' not in prices.columns:
        print("   ❌ Missing Data (INDA/FXI)")
        return
        
    inda = prices['INDA']
    fxi = prices['FXI']
    
    # Strategy: Equal Weight Long INDA / Short FXI
    # Rebalance Daily (Simplified)
    r_inda = inda.pct_change()
    r_fxi = fxi.pct_change()
    
    # Spread Return = 0.5 * INDA - 0.5 * FXI
    # Or just Long/Short 100% each? Let's say 50/50 capital.
    strat_ret = 0.5 * r_inda - 0.5 * r_fxi
    strat_ret = strat_ret.dropna()
    
    def get_stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod()
        total = dd.iloc[-1] - 1
        return ann, sharpe, total
        
    ann, sharpe, total = get_stats(strat_ret)
    
    print(f"   Long INDA / Short FXI Results:")
    print(f"   Ann Ret: {ann:.1%} | Sharpe: {sharpe:.2f} | Total: {total:.1%}")
    
    if sharpe > 0.5:
        print("   ✅ EDGE FOUND: Geopolitics pays.")
    else:
        print("   ❌ NO EDGE: Arbitraged away or too volatile.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🧧 ASIAN ALPHA HUNT")
    print("="*60)
    
    prices = fetch_asia_data()
    test_yen_carry(prices)
    test_dragon_divergence(prices)
    
    print("\n" + "=" * 60)
