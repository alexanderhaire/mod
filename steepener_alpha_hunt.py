"""
Yield Curve Steepener Alpha Hunt
================================

Testing the "Doomsday Detector".
Hypothesis: Rapid Steepening (Un-Inversion) of the Yield Curve precedes major crashes (2000, 2008, 2020).
Signal: Change in 30Y - 5Y Spread.
Trigger: If Spread increases > 0.4% (40bps) in 20 days.

Strategy:
- Normal: Long SPY.
- Trigger: Eject to TLT (Bonds usually rally in crashes).

RUN: python steepener_alpha_hunt.py
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
    print("📉 Fetching Yield Curve Data (^FVX, ^TYX)...")
    tickers = ['^FVX', '^TYX', 'SPY', 'TLT']
    data = yf.download(tickers, start='1995-01-01', progress=False) # Get long history (2000, 2008)
    
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
# 2. STRATEGY ENGINE
# =============================================================================

def run_steepener_strategy(prices):
    if '^FVX' not in prices.columns or '^TYX' not in prices.columns: return None
    
    fvx = prices['^FVX'] # 5 Year
    tyx = prices['^TYX'] # 30 Year
    
    # Spread (Yields are in %, so 5.0 = 5%)
    spread = tyx - fvx
    
    # Change in Spread (20 day rolling)
    spread_chg = spread.diff(20)
    
    # Signal: Panic Steepening
    # Threshold: +0.40 (40bps) in 20 days
    panic = (spread_chg > 0.40).astype(int)
    
    # Strategy
    # If Panic -> Long TLT. Else Long SPY.
    # Lag 1 day
    w_tlt = panic.shift(1).fillna(0)
    w_spy = 1.0 - w_tlt
    
    spy_ret = prices['SPY'].pct_change()
    tlt_ret = prices['TLT'].pct_change() # Keep TLT (Crash Hedge)
    
    strat_ret = w_spy * spy_ret + w_tlt * tlt_ret
    
    return strat_ret.fillna(0), panic

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_steepener(prices):
    print("\n📉 YIELD CURVE STEEPENER RESULT:")
    print("-" * 75)
    
    spy_ret = prices['SPY'].pct_change().fillna(0)
    strat_ret, panic_sig = run_steepener_strategy(prices)
    
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        
        cum = (1+r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        mdd = drawdown.min()
        
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | MDD {mdd:>6.1%}")
        return sharpe

    s_bh = get_stats(spy_ret, "SPY (Buy & Hold)")
    s_strat = get_stats(strat_ret, "Steepener Protocol")
    
    print("-" * 75)
    
    # Check Specific Crashes
    # 2008 (Lehman: Sept 2008. But Yield Curve steepened before?)
    y08_spy = spy_ret.loc['2008-01-01':'2008-12-31'].sum()
    y08_strat = strat_ret.loc['2008-01-01':'2008-12-31'].sum()
    
    # 2000 (Tech Wreck)
    y00_spy = spy_ret.loc['2000-01-01':'2002-12-31'].sum() if '2000-01-01' in spy_ret.index else 0
    y00_strat = strat_ret.loc['2000-01-01':'2002-12-31'].sum() if '2000-01-01' in strat_ret.index else 0
    
    print(f"   2008 Return: SPY {y08_spy:.1%} vs Protocol {y08_strat:.1%}")
    if y00_spy != 0:
         print(f"   2000-02 Return: SPY {y00_spy:.1%} vs Protocol {y00_strat:.1%}")
         
    # How often is it triggered?
    days_panic = panic_sig.sum()
    print(f"   Panic Days Triggered: {days_panic} ({days_panic/len(prices):.1%})")
    
    if s_strat > s_bh + 0.05:
         print("✅ ALPHA FOUND: Steepener Detector saved the portfolio.")
         print("   Action: Add to Defensive Logic.")
    else:
         print("❌ FAILED: False alarms cost too much upside.")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_steepener(prices)
