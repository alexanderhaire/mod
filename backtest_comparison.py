import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autonomous_fund.strategies.smart_bond import SmartBondStrategy

def run_simulation():
    print("⚔️  BACKTEST BATTLE: STATIC (Old) vs. DYNAMIC (Hedge Fund) ⚔️")
    
    # 1. Setup Environments
    initial_capital = 10000.0
    static_capital = initial_capital
    dynamic_capital = initial_capital
    
    # Parameters
    n_days = 365
    trades_per_day = 5
    
    # Trackers
    static_equity = [static_capital]
    dynamic_equity = [dynamic_capital]
    
    # Initialize Strategy Logic for calculation
    strat = SmartBondStrategy(portfolio_size=initial_capital)
    
    print(f"Simulating {n_days} days of trading ({n_days*trades_per_day} potential opportunities)...")
    
    for day in range(n_days):
        # Update portfolio size in strategy
        strat.portfolio_size = dynamic_capital
        
        for _ in range(trades_per_day):
            # Generate Synthetic Market Opportunity
            # Most markets have no edge. Some have huge edge.
            
            # True Probability of winning (The "God" view)
            # Most bonds are safe (95-99%)
            true_prob = np.random.uniform(0.90, 0.999)
            
            # Market Price (Inefficiency)
            # Market usually prices correctly, but sometimes undershoots
            # Price is usually slightly below True Prob (Risk Premium)
            # Sometimes Price is way below True Prob (Alpha!)
            noise = np.random.normal(0, 0.02)
            market_price = max(0.85, min(0.99, true_prob - 0.01 + noise))
            
            # Outcome (Did it win?)
            did_win = np.random.rand() < true_prob
            
            # --- STATIC STRATEGY ---
            # Buying Logic: Price < 0.94 AND > 0.00 EDGE ?
            # Static size: 10% of INITIAL capital (non-compounding usually, or slow compounding)
            static_bet = 0
            if market_price < 0.94: 
                static_bet = min(static_capital * 0.10, 500) # Capped at $500
            
            if static_bet > 0:
                if did_win:
                    profit = (static_bet / market_price) - static_bet
                    static_capital += profit
                else:
                    static_capital -= static_bet

            # --- DYNAMIC STRATEGY ---
            # Buying Logic: Kelly Criterion > 0
            # Ask the strategy how much to bet!
            # We assume strategy estimates prob correctly-ish (e.g. 98%)
            dynamic_bet = strat.calculate_kelly_bet(market_price, 0.98, days=10)
            
            if dynamic_bet > 0:
                if did_win:
                    profit = (dynamic_bet / market_price) - dynamic_bet
                    dynamic_capital += profit
                else:
                    dynamic_capital -= dynamic_bet

        static_equity.append(static_capital)
        dynamic_equity.append(dynamic_capital)

    # Results
    print("\n🏁 FINAL RESULTS 🏁")
    print(f"Static Strategy:  ${static_capital:,.2f}  ({(static_capital/initial_capital -1)*100:.1f}%)")
    print(f"Dynamic Strategy: ${dynamic_capital:,.2f}  ({(dynamic_capital/initial_capital -1)*100:.1f}%)")
    
    if dynamic_capital > static_capital:
        print("\n🏆 WINNER: DYNAMIC HEDGE FUND MODE (Kelly Compounding)")
        print("Note: Dynamic wins by betting bigger when the price is cheap ($0.85).")
    else:
        print("\n🏆 WINNER: STATIC MODE (Safety First)")

if __name__ == "__main__":
    run_simulation()
