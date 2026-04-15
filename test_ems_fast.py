
import pandas as pd
import numpy as np
from compounder_strategy import compounder_strategy
from validate_edge import fetch_real_data

def run_fast_test():
    print("Fetching data...")
    prices, vix = fetch_real_data(years=5) # 5 years is enough for test
    
    print(f"Running Compounder Strategy on {len(prices)} days...")
    
    weights = compounder_strategy(prices, vix)
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    # Weights already include T+1 lag from strategy
    port_returns = (weights * returns).sum(axis=1)
    
    sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
    total_return = (1 + port_returns).prod() - 1
    
    print("\n--- RESULTS ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    if sharpe > 0.5:
        print("Verdict: Strategy is WORKING (Positive Sharpe)")
    else:
        print("Verdict: Strategy needs tuning (Low Sharpe)")

if __name__ == "__main__":
    run_fast_test()
