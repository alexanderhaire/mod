"""
Currency Trend Alpha Hunt
=========================

Testing the "Wrecking Ball" Hedge.
Hypothesis: The US Dollar (UUP) acts as Crisis Alpha during inflationary/rate shock regimes (e.g. 2022).
Strategy: Trend Following (Golden Cross: SMA 50 > SMA 200).
Allocation: If Trend is UP, hold UUP. Else Cash.

Comparison: Does this alleviate the 2022 Drawdown of a traditional 60/40 portfolio?

RUN: python currency_trend_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA
# =============================================================================

def fetch_data():
    print("💵 Fetching Currency Data (UUP, SPY, TLT)...")
    tickers = ['UUP', 'SPY', 'TLT']
    data = yf.download(tickers, start='2008-01-01', progress=False)
    
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

def run_dollar_trend(prices):
    if 'UUP' not in prices.columns: return None
    
    uup = prices['UUP']
    ret = uup.pct_change()
    
    # Golden Cross
    sma_50 = uup.rolling(50).mean()
    sma_200 = uup.rolling(200).mean()
    
    signal = pd.Series(0, index=uup.index)
    signal[sma_50 > sma_200] = 1.0
    
    # Lag
    weight = signal.shift(1).fillna(0)
    
    # Return
    strat_ret = weight * ret
    return strat_ret.fillna(0)

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_currency(prices):
    print("\n💵 DOLLAR TREND RESULTS (SMA 50/200):")
    print("-" * 75)
    
    # Baselines
    spy_ret = prices['SPY'].pct_change().fillna(0)
    tlt_ret = prices['TLT'].pct_change().fillna(0)
    
    # The Problem Portfolio (60/40)
    port_6040 = 0.6 * spy_ret + 0.4 * tlt_ret
    
    # The Solution? (UUP Trend)
    uup_trend = run_dollar_trend(prices)
    if uup_trend is None: return
    
    # Metrics
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
        
        # Return during 2022?
        y2022 = r.loc['2022-01-01':'2022-12-31']
        ret22 = (1+y2022).prod() - 1
        
        print(f"   {name:<25} | Sharpe {sharpe:.2f} | 2022 Ret {ret22:>6.1%} | MDD {mdd:>6.1%}")
        return r

    b_6040 = get_stats(port_6040, "Classic 60/40")
    b_uup = get_stats(uup_trend, "Dollar Trend (UUP)")
    
    print("-" * 75)
    
    # The "Hedged" Portfolio
    # 50% Classic + 50% Dollar Trend? Or 80% Classic + 20% Dollar Trend
    # Let's try 80/20
    
    w_hedge = 0.20
    hedged_ret = (1 - w_hedge) * port_6040 + w_hedge * uup_trend
    
    get_stats(hedged_ret, "Hedged (80% 60/40 + 20% UUP)")
    
    # Correlation Check
    corr = uup_trend.corr(port_6040)
    print(f"\n   Correlation (UUP Trend vs 60/40): {corr:.2f}")
    
    if uup_trend.loc['2022-01-01':'2022-12-31'].sum() > 0:
         print("✅ CRISIS ALPHA FOUND: Dollar Trend made money in 2022.")
    else:
         print("❌ FAILED: Dollar Trend did not protect in 2022.")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_currency(prices)
