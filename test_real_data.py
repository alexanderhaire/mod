"""
Test script for ML backtesting with REAL Yahoo Finance data.
This eliminates the random-walk problem where models learn nothing.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add local module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine import Backtester
from external_data import fetch_market_data_pool

# Test assets - mix of stocks, ETFs, crypto
TEST_ASSETS = [
    "SPY (S&P 500)", "QQQ (Nasdaq 100)", "IWM (Russell 2000)",
    "XLE (Energy)", "XLF (Financials)", "XLK (Technology)",
    "GLD (Gold)", "USO (Oil)", "TLT (20+ Yr Treasury)",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "BTCUSD", "ETHUSD"
]

def fetch_real_data():
    """Fetch 2 years of real market data from Yahoo Finance."""
    print("📥 Fetching real market data from Yahoo Finance...")
    print("   (This may take 30-60 seconds)")
    
    def progress(pct):
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r   [{bar}] {pct*100:.0f}%", end="", flush=True)
    
    pool = fetch_market_data_pool(TEST_ASSETS, _progress_callback=progress)
    print()  # Newline after progress bar
    
    # Convert to price DataFrame
    price_data = {}
    for asset, info in pool.items():
        if info.get("data") and len(info["data"]) > 0:
            dates = [d["date"] for d in info["data"]]
            prices = [d["price_index"] for d in info["data"]]
            price_data[asset] = pd.Series(prices, index=pd.to_datetime(dates), name=asset)
    
    # Combine into DataFrame
    if not price_data:
        raise ValueError("No data fetched - check internet connection")
        
    df = pd.DataFrame(price_data)
    
    # Forward-fill and drop any remaining NaN
    df = df.ffill().dropna()
    
    print(f"✅ Loaded {len(df)} days of data for {len(df.columns)} assets")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def run_backtest(price_data):
    """Run the backtester on real data."""
    print("\n🔄 Running backtest with enhanced features (12 signals)...")
    
    def progress(pct):
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r   [{bar}] {pct*100:.0f}%", end="", flush=True)
    
    bt = Backtester(initial_capital=100_000)
    result = bt.run(price_data, progress_callback=progress)
    print()  # Newline
    
    return result


def main():
    print("=" * 60)
    print("   ML BACKTEST WITH REAL MARKET DATA")
    print("   Enhanced Features: 12 alpha signals")
    print("=" * 60)
    
    # Fetch data
    price_data = fetch_real_data()
    
    # Run backtest
    result = run_backtest(price_data)
    
    # Print results
    print("\n" + "=" * 60)
    print("   BACKTEST RESULTS")
    print("=" * 60)
    # Extract metrics from nested dict
    metrics = result.get("metrics", {})
    sharpe = metrics.get("Sharpe", 0)
    ret = metrics.get("Total Return", 0)
    cagr = metrics.get("CAGR", 0)
    vol = metrics.get("Volatility", 0)
    
    print(f"📈 Total Return: {ret*100:.2f}%")
    print(f"📊 CAGR:         {cagr*100:.2f}%")
    print(f"⚡ Sharpe Ratio: {sharpe:.3f}")
    print(f"📉 Volatility:   {vol*100:.2f}%")
    
    # Performance assessment
    print("\n" + "-" * 60)
    if sharpe >= 1.2:
        print("✅ EXCELLENT: Sharpe >= 1.2 - Strategy is working well!")
    elif sharpe >= 0.8:
        print("🟡 GOOD: Sharpe >= 0.8 - Strategy is functional")
    elif sharpe >= 0.5:
        print("🟠 MODERATE: Sharpe >= 0.5 - Room for improvement")
    else:
        print("🔴 NEEDS WORK: Sharpe < 0.5 - Consider parameter tuning")
    
    print("-" * 60)
    
    return result


if __name__ == "__main__":
    result = main()
