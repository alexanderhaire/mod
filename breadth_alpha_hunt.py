"""
Breadth Divergence Alpha Hunt
=============================

Testing Market Internals (Breadth) as a Regime Filter.
Hypothesis: "Narrow Rallies" (SPY up, RSP down) are fragile and predict reversals.
Signal: Divergence betwen Cap-Weight (SPY) and Equal-Weight (RSP).

Strategy:
- Base: Long SPY if Price > SMA(200).
- Filter: If SPY is making highs but RSP/SPY Ratio is below its SMA(50), forcing exit/hedging.
- Logic: Avoid "Generals leading without Soldiers".

RUN: python breadth_alpha_hunt.py
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
    print("⚔️ Fetching Breadth Data (SPY, RSP)...")
    tickers = ['SPY', 'RSP']
    data = yf.download(tickers, start='2003-01-01', progress=False) # RSP started ~2003
    
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

def run_breadth_strategy(prices):
    if 'RSP' not in prices.columns or 'SPY' not in prices.columns: return None
    
    spy = prices['SPY']
    rsp = prices['RSP']
    ret = spy.pct_change()
    
    # Breadth Ratio
    ratio = rsp / spy
    
    # Trends
    spy_sma200 = spy.rolling(200).mean()
    ratio_sma50 = ratio.rolling(50).mean()
    
    # 1. Base Strategy (Trend)
    # Long if SPY > SMA200
    base_signal = (spy > spy_sma200).astype(int)
    
    # 2. Breadth Filter
    # Danger Zone: Ratio < Ratio SMA50 (Breadth weakening)
    # Or Divergence: SPY > SMA50 (Rising) AND Ratio < SMA50 (Falling)
    # Let's try simple Ratio Trend.
    # If Ratio < SMA50, it means Equal Weight is underperforming -> Weakness.
    # Filter: If Ratio < SMA50, go to Cash (or Neutral).
    
    breadth_ok = (ratio > ratio_sma50).astype(int)
    
    # Combined Signal: Trend MUST be Up AND Breadth MUST be Healthy
    filtered_signal = base_signal * breadth_ok
    
    # Shift
    w_base = base_signal.shift(1).fillna(0)
    w_filt = filtered_signal.shift(1).fillna(0)
    
    # Returns
    r_base = w_base * ret
    r_filt = w_filt * ret
    
    return r_base, r_filt

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_breadth(prices):
    print("\n⚔️ BREADTH DIVERGENCE RESULTS:")
    print("-" * 75)
    
    spy_ret = prices['SPY'].pct_change().dropna()
    r_base, r_filt = run_breadth_strategy(prices)
    
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        
        # Max Drawdown
        cum = (1+r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        mdd = drawdown.min()
        
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | MDD {mdd:>6.1%}")
        return sharpe

    s_bh = get_stats(spy_ret, "SPY (Buy & Hold)")
    s_base = get_stats(r_base, "Base Trend (SMA200)")
    s_filt = get_stats(r_filt, "Breadth Filtered")
    
    print("-" * 75)
    
    if s_filt > s_base:
        if s_filt > s_base + 0.1:
            print("✅ ALPHA FOUND: Breadth Filter significantly improved performance.")
            print("   Action: Add RSP/SPY check to Ultimate Strategy.")
        else:
             print("⚠️ MARGINAL: Tiny improvement. Not worth complexity.")
    else:
        print("❌ FAILED: Breadth Filter reduced returns. 'Narrow Rallies' are still profitable.")
        print("   Sometimes the Generals (Mega Caps) carry the market alone (e.g. Magnificent 7).")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_breadth(prices)
