
import logging
from auto_trader import AutoTrader
import pandas as pd

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_heart_beat():
    print("Initializing AutoTrader...")
    try:
        trader = AutoTrader(mode="paper")
        print("AutoTrader initialized.")
        
        # Test heart_beat with no market data (triggers _simulate_tick)
        print("Testing heart_beat() with no market data...")
        result = trader.heart_beat(market_data_feed=None)
        
        print("heart_beat() returned:", result)
        print("SUCCESS: No crash encountered.")
        
    except Exception as e:
        print(f"FAILURE: Crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_heart_beat()
