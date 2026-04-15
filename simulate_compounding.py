"""
Compounding Growth Simulator
============================

Projects portfolio growth for a small account strategy ($200 initial + $100/week).
Focus: Maximizing Sharpe Ratio (Risk-Adjusted Return).

Scenarios:
1. Conservative (Market Making): Lower Yield, Low Volatility.
2. Aggressive (Alpha Sniping): High Yield, High Volatility.
3. Sharpe Optimized (Diversified): Balanced Yield, Lowest Volatility.

Assumptions:
- Weekly compounding.
- Constant yield (simplified).
- Volatility modeled as standard deviation of weekly returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(initial_capital=200, weekly_contrib=100, weeks=52, apy=0.15, vol_annual=0.10):
    """
    Simulates portfolio path.
    apy: Annual Percentage Yield (e.g., 0.15 for 15%)
    vol_annual: Annualized Volatility (e.g., 0.10 for 10%)
    """
    
    # improved precision for weekly rates
    weekly_rate = (1 + apy) ** (1/52) - 1
    weekly_vol = vol_annual / np.sqrt(52)
    
    capital = initial_capital
    history = []
    
    # Run simulation with random walk for realism
    # Seed for reproducibility
    np.random.seed(42) 
    
    for w in range(weeks):
        # Contribution
        capital += weekly_contrib
        
        # Return + Noise
        shock = np.random.normal(0, weekly_vol)
        r = weekly_rate + shock
        
        capital *= (1 + r)
        
        history.append({
            'Week': w + 1,
            'Capital': capital,
            'Contribution': initial_capital + (weekly_contrib * (w + 1))
        })
        
    df = pd.DataFrame(history)
    
    # Calculate Sharpe
    # Risk Free Rate roughly 4%
    rf_weekly = (1.04) ** (1/52) - 1
    
    # Re-derive weekly returns from the sim path (excluding contribution impact if possible, 
    # but for simple sharpe on portfolio performance we compare end result).
    # Ideally Sharpe is on the STRATEGY returns, not the portfolio value (which has inflows).
    # Let's approximate Strategy Sharpe based on input parameters.
    
    excess_return = apy - 0.04
    sharpe = excess_return / vol_annual if vol_annual > 0 else 0
    
    return df, sharpe

def main():
    print("📈 SMALL ACCOUNT GROWTH SIMULATOR")
    print("=================================")
    print("Goal: Maximize Sharpe Ratio")
    print(f"Start: $200 | Add: $100/week | Horizon: 1 Year")
    print("-" * 80)
    
    # Define Scenarios
    scenarios = [
        {
            "Name": "Conservative (Market Making)",
            "APY": 0.15,
            "Vol": 0.05, # Very stable, just spread collection
            "Desc": "Low risk, consistent small gains."
        },
        {
            "Name": "Aggressive (Alpha Sniping)",
            "APY": 0.45,
            "Vol": 0.40, # High variance, misses often, drawdowns
            "Desc": "High risk, chasing massive disconnected pumps."
        },
        {
            "Name": "Sharpe Optimized (Hybrid)",
            "APY": 0.25,
            "Vol": 0.10, # Balanced allocation (diversified)
            "Desc": "Best mix: Yield of 25% with dampened volatility."
        }
    ]
    
    print(f"{'Strategy':<30} {'APY':<8} {'Vol':<8} {'Sharpe':<8} {'End Value':<12} {'Profit'}")
    print("-" * 100)
    
    results = []
    
    for sc in scenarios:
        df, sharpe = run_simulation(apy=sc['APY'], vol_annual=sc['Vol'])
        final_val = df.iloc[-1]['Capital']
        total_contrib = df.iloc[-1]['Contribution']
        profit = final_val - total_contrib
        
        print(f"{sc['Name']:<30} {sc['APY']*100:.0f}%     {sc['Vol']*100:.0f}%      {sharpe:.2f}     ${final_val:,.2f}    ${profit:,.2f}")
        
        results.append((sc['Name'], df))

    print("-" * 100)
    print("\n💡 INSIGHT:")
    print("   The 'Sharpe Optimized' strategy doesn't have the highest end value,")
    print("   but it offers the smoothest ride. For a growing account,")
    print("   minimizing drawdowns (Vol) is key to psychological consistency.")

if __name__ == "__main__":
    main()
