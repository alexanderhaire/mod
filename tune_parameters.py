"""
Systematic Parameter Tuning Script
Tests multiple configurations and reports best performers.
"""
import pandas as pd
import numpy as np
import time
import sys
import os

# Ensure we can import ml_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine import Backtester, PredictiveAlphaEngine, PortfolioOptimizer
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress INFO spam

def generate_dummy_data(days=300, assets=7, seed=42):
    """Generate reproducible random walk data."""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=days, freq="B")
    data = {}
    for i in range(assets):
        prices = [100.0]
        # Add slight drift and volatility variation per asset
        drift = np.random.uniform(-0.0001, 0.0003)
        vol = np.random.uniform(0.008, 0.02)
        for _ in range(days - 1):
            change = np.random.normal(drift, vol)
            prices.append(prices[-1] * (1 + change))
        data[f"Asset_{i}"] = prices
    return pd.DataFrame(data, index=dates)

def run_backtest(price_data, config_name="default"):
    """Run a single backtest and return metrics."""
    backtester = Backtester(initial_capital=10000.0, transaction_cost_pct=0.0005)
    
    start = time.time()
    results = backtester.run(price_data, window_size=30)
    duration = time.time() - start
    
    metrics = results.get("metrics", {})
    return {
        "config": config_name,
        "sharpe": metrics.get("Sharpe", 0),
        "return": metrics.get("Total Return", 0),
        "cagr": metrics.get("CAGR", 0),
        "volatility": metrics.get("Volatility", 0),
        "duration": duration
    }

def main():
    print("=" * 60)
    print("NEURAL ALPHA ENGINE - PARAMETER TUNING")
    print("=" * 60)
    
    # Generate consistent test data
    print("\nGenerating test data (300 days, 7 assets)...")
    df = generate_dummy_data(days=300, assets=7, seed=42)
    
    results = []
    
    # Test 1: Current Configuration
    print("\n[1/5] Testing: Current Config (Confidence Gate 0.25)...")
    r = run_backtest(df, "Current (0.25 gate)")
    results.append(r)
    print(f"       Sharpe: {r['sharpe']:.3f}, Return: {r['return']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(results_df.to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n🏆 BEST CONFIG: {best['config']}")
    print(f"   Sharpe: {best['sharpe']:.3f}, Return: {best['return']*100:.1f}%")

if __name__ == "__main__":
    main()
