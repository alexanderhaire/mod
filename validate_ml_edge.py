
import pandas as pd
import numpy as np
import yfinance as yf
from ml_engine import Backtester
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start="2016-01-01", end="2026-01-01", progress=False)
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    prices = prices.dropna()
    return prices

def validate_ml():
    print("Running ML Strategy Validation (Lasso Model)...")
    
    # 1. Fetch Data
    prices = fetch_data()
    if prices.empty:
        print("Failed to fetch data.")
        return

    # 2. Results holder
    print(f"Data shape: {prices.shape}")
    
    # 3. Initialize Backtester
    # Check if we can enable 3x leverage by modifying code or if it's hardcoded
    # The backtester in ml_engine.py uses 'target_weights' which are normalized to 1.0
    # We will run it 'as is' first to see the Sharpe.
    
    backtester = Backtester(initial_capital=10000.0)
    
    # 4. Run Backtest (Walk-Forward)
    print("Starting Walk-Forward Validation (Stream)...")
    results = backtester.run(prices, window_size=252, progress_callback=lambda x: print(f".", end="", flush=True))
    print("\n")
    
    if "metrics" in results:
        m = results["metrics"]
        print("-" * 40)
        print(f"Total Return: {m['Total Return']:.2%}")
        print(f"CAGR:         {m['CAGR']:.2%}")
        print(f"Sharpe Ratio: {m['Sharpe']:.4f}")
        print(f"Volatility:   {m['Volatility']:.2%}")
        print("-" * 40)
        
        if m['Sharpe'] > 0.5:
             print("✅ Result: POSITIVE EDGE DETECTED in ML Model!")
        else:
             print("❌ Result: NO EDGE (Sharpe < 0.5)")
    else:
        print("Backtest failed or returned no metrics.")

if __name__ == "__main__":
    validate_ml()
