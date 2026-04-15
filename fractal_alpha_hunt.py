"""
Fractal Alpha Hunt (Chaos Theory / Hurst Exponent)
==================================================

Testing the "Fractal Market Hypothesis".
Metric: Hurst Exponent (H).
- H = 0.5: Random Walk (Efficient Market).
- H > 0.5: Persistent (Trend).
- H < 0.5: Anti-Persistent (Mean Reversion).

Strategy:
- Calculate Rolling Hurst (100 days).
- If H > 0.6: MOMENTUM (Buy Breakouts).
- If H < 0.4: MEAN REVERSION (Buy Dips).
- Else: Cash.

RUN: python fractal_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. MATH: HURST EXPONENT (R/S Analysis)
# =============================================================================

def calculate_hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts.
    Using simplified R/S analysis.
    """
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0 
    # Note: There are many estimators. This is a Variance-Time estimator approximation.
    # Ideally we'd use full R/S but it's slow for rolling.
    # Let's try the standard RS method if fast enough.

def get_rolling_hurst(series, window=100):
    # Vectorized Rolling Hurst is hard. Loop is cleaner.
    hurst_series = pd.Series(index=series.index, dtype=float)
    
    # Pre-compute
    vals = series.values
    
    # We will use a standard library simplified logic:
    # H = log(R / S) / log(N)
    # Where R is range of cumulative deviate, S is std dev.
    
    for i in range(window, len(vals)):
        chunk = vals[i-window:i]
        
        # Calculate R/S
        # 1. Log returns
        # Actually Hurst is usually on log prices or returns.
        # Let's use Log Prices.
        
        # Simple implementation of R/S
        # Mean centered series
        mu = np.mean(chunk)
        centered = chunk - mu
        
        # Cumulative deviation
        y = np.cumsum(centered)
        
        # Range
        R = np.max(y) - np.min(y)
        
        # Standard Deviation
        S = np.std(chunk)
        
        if S == 0:
            H = 0.5
        else:
            H = np.log(R/S) / np.log(window)
            
        hurst_series.iloc[i] = H
        
    return hurst_series

# =============================================================================
# 2. DATA
# =============================================================================

def fetch_data():
    print("🌌 Fetching Market Data (Chaos Mode)...")
    data = yf.download("SPY", start='2015-01-01', progress=False)
    # Robust extraction
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices

# =============================================================================
# 3. BACKTEST
# =============================================================================

def backtest_fractal(prices):
    print("🌀 Calculating Fractal Dimensions (Hurst Exponent)...")
    
    # Use Log Prices for Hurst
    log_prices = np.log(prices)
    
    # Rolling Hurst
    hurst = get_rolling_hurst(log_prices, window=126) # 6 months
    
    print(f"   Avg Hurst: {hurst.mean():.2f} (Theoretical Random = 0.5)")
    
    # Strategy
    # If H > 0.55 -> Trend Following (Long if Price > SMA 50)
    # If H < 0.45 -> Mean Reversion (Long if Price < SMA 50) - wait, Mean Rev means fade.
    # Let's keep it simple: Regimes.
    
    sma = prices.rolling(50).mean()
    returns = prices.pct_change()
    
    # Signals
    # Trend Mode: Hurst High. Signal = Price > SMA
    trend_mode = (hurst > 0.55)
    trend_signal = (prices > sma).astype(int)
    
    # Mean Rev Mode: Hurst Low. Signal = Price < SMA (Buy the dip)
    mr_mode = (hurst < 0.45)
    mr_signal = (prices < sma).astype(int) 
    
    # Combined Position
    # Ensure 1D
    if hasattr(trend_signal, 'iloc'): trend_signal = trend_signal.iloc[:,0] if trend_signal.ndim > 1 else trend_signal
    if hasattr(mr_signal, 'iloc'): mr_signal = mr_signal.iloc[:,0] if mr_signal.ndim > 1 else mr_signal
    
    pos = pd.Series(0, index=prices.index)
    
    pos[trend_mode] = trend_signal[trend_mode] # If High H: Buy if above SMA
    pos[mr_mode] = mr_signal[mr_mode]          # If Low H: Buy if below SMA
    
    # Shift
    strat_ret = pos.shift(1) * returns
    strat_ret = strat_ret.dropna()
    
    # Benchmark
    bh_ret = returns.dropna()
    
    # Metrics
    def get_stats(r):
        print(f"   [Debug] Stats Input Shape: {r.shape if hasattr(r, 'shape') else 'NoShape'}")
        if hasattr(r, 'squeeze'): r = r.squeeze()
        if hasattr(r, 'shape') and len(r.shape) > 1 and r.shape[1] > 1:
             # Force mean across columns if multiple
             r = r.mean(axis=1)
             
        mean = r.mean()
        std = r.std()
        
        # Convert to float
        if hasattr(mean, 'item'): mean = mean.item()
        if hasattr(std, 'item'): std = std.item()
        
        ann = mean * 252
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0
        return ann, sharpe
        
    s_ann, s_sharpe = get_stats(strat_ret)
    b_ann, b_sharpe = get_stats(bh_ret)
    
    print("\n🌪️ FRACTAL STRATEGY RESULTS:")
    print(f"   Buy & Hold: {b_ann:.1%} | Sharpe {b_sharpe:.2f}")
    print(f"   Fractal Switch: {s_ann:.1%} | Sharpe {s_sharpe:.2f}")
    
    if s_sharpe > b_sharpe + 0.1:
        print("   ✅ EDGE FOUND: Chaos Theory works!")
    else:
        print("   ❌ NO EDGE: The market is a Random Walk (H ~ 0.5).")

if __name__ == "__main__":
    print("="*60)
    print("🌀 FRACTAL ALPHA HUNT (CHAOS THEORY)")
    print("="*60)
    
    prices = fetch_data()
    backtest_fractal(prices)
    
    print("\n" + "=" * 60)
