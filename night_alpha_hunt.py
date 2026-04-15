"""
Night Effect Alpha Hunt
=======================

Testing the "Night Effect" Anomaly.
Hypothesis: Most equity returns occur Overnight (Close-to-Open).
Intraday (Open-to-Close) returns are noise or negative.

Strategy: "The Vampire" -> Buy Close, Sell Open. 
Comparison: "The Day Trader" -> Buy Open, Sell Close.

RUN: python night_alpha_hunt.py
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
    print("🌙 Fetching SPY Data (Open/Close)...")
    data = yf.download('SPY', start='2000-01-01', progress=False)
    
    # Need Open and Close. Adj Close is tricky for intraday split.
    # We will use Close/Open ratio and apply to Adj Close for total return accuracy.
    # Or just use raw Close/Open relative changes.
    
    # Fixing extraction for single ticker where it might return Series or DataFrame with 1 col
    prices = data
    
    # Try different access patterns based on yf version/structure
    try:
        if isinstance(prices.columns, pd.MultiIndex):
             # Multi-level (Price, Ticker)
             op = prices['Open']['SPY'] if 'SPY' in prices['Open'].columns else prices.xs('Open', axis=1, level=0)
             cl = prices['Close']['SPY'] if 'SPY' in prices['Close'].columns else prices.xs('Close', axis=1, level=0)
             adj = prices['Adj Close']['SPY'] if 'Adj Close' in prices.columns else cl
        else:
             # Flat
             op = prices['Open']
             cl = prices['Close']
             adj = prices['Adj Close'] if 'Adj Close' in prices.columns else cl
    except:
         # Fallback for simple structure
         op = prices['Open']
         cl = prices['Close']
         adj = prices['Close']

    # Ensure they are Series
    if isinstance(op, pd.DataFrame): op = op.iloc[:,0]
    if isinstance(cl, pd.DataFrame): cl = cl.iloc[:,0]
    if isinstance(adj, pd.DataFrame): adj = adj.iloc[:,0]
    
    df = pd.DataFrame({'Open': op, 'Close': cl, 'Adj Close': adj}, index=op.index)
    df = df.dropna()
    print(f"   Data: {len(df)} days")
    return df

# =============================================================================
# 2. ANALYSIS
# =============================================================================

def test_night_effect(df):
    print("\n🧛 RUNNING THE VAMPIRE STRATEGY...")
    
    # 1. Calculate Returns
    # Night: Buy Close(T-1), Sell Open(T)
    # Day: Buy Open(T), Sell Close(T)
    
    # Night Return = (Open(T) - Close(T-1)) / Close(T-1)
    df['Night_Ret'] = (df['Open'] / df['Close'].shift(1)) - 1
    
    # Day Return = (Close(T) - Open(T)) / Open(T)
    df['Day_Ret'] = (df['Close'] / df['Open']) - 1
    
    # Buy & Hold (Close to Close)
    # Note: Using Adj Close for Buy Hold to capture dividends
    df['BuyHold_Ret'] = df['Adj Close'].pct_change()
    
    df = df.dropna()
    
    # 2. Results
    
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        cum = (1+r).cumprod().iloc[-1] - 1
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f} | Total {cum:.0%}")
        return sharpe

    print("-" * 80)
    get_stats(df['BuyHold_Ret'], "Buy & Hold (SPY)")
    n_sharpe = get_stats(df['Night_Ret'], "Night Strategy (Vampire)")
    d_sharpe = get_stats(df['Day_Ret'], "Day Strategy (Intraday)")
    print("-" * 80)
    
    # 3. Correlation
    corr = df['Night_Ret'].corr(df['Day_Ret'])
    print(f"   Correlation (Night vs Day): {corr:.3f}")
    
    # 4. Verdict
    if n_sharpe > d_sharpe + 0.2:
        print("\n✅ ANOMALY CONFIRMED: The Vampire wins.")
        print("   Strategy Implication: Execute trades at the Close, not the Open.")
        if d_sharpe < 0:
             print("   Day Trading is a Loser's Game (Negative Expectancy).")
    else:
        print("\n❌ ANOMALY FADED: Markets are efficient 24/7.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = fetch_data()
    test_night_effect(df)
