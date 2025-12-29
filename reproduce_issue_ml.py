
import pandas as pd
import numpy as np
import time
from ml_engine import Backtester
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def generate_dummy_data(days=500, assets=5):
    dates = pd.date_range(start="2020-01-01", periods=days, freq="B")
    data = {}
    for i in range(assets):
        # Random walk
        prices = [100.0]
        for _ in range(days-1):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        data[f"Asset_{i}"] = prices
    return pd.DataFrame(data, index=dates)

def test_performance():
    print("Generating dummy data...")
    df = generate_dummy_data(days=200, assets=7) # 200 days, 7 assets
    
    backtester = Backtester(initial_capital=10000.0)
    
    print("Starting Backtest...")
    start_time = time.time()
    
    # Callback to show progress
    def progress(p):
        print(f"Progress: {p:.1%}", end='\r')
        
    results = backtester.run(df, window_size=30, progress_callback=progress)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\nBacktest finished in {duration:.2f} seconds.")
    print("\nMetrics:")
    for k, v in results.get("metrics", {}).items():
        print(f"{k}: {v}")
        
    final_cap = results.get("final_capital", 0)
    print(f"Final Capital: ${final_cap:,.2f}")
    
    if duration > 30: # Should be < 5 seconds with vectorization
        print("FAIL: Slowness detected (Duration > 30s)")
    else:
        print("PASS: Performance is acceptable.")

if __name__ == "__main__":
    test_performance()
