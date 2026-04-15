"""
Prediction Market Arbitrage Hunt - Multi-Market Edition
=====================================================

Backtesting arbitrage strategies on synthetic prediction market data.
Simulates inefficiencies described in research across MULTIPLE markets.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_single_market(days=90, steps_per_day=24, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n_steps = days * steps_per_day
    
    # 1. Underlying True Probability
    true_prob = np.zeros(n_steps)
    true_prob[0] = np.random.uniform(0.3, 0.7) # Randomize start prob
    
    for i in range(1, n_steps):
        drift = np.random.normal(0, 0.01)
        true_prob[i] = np.clip(true_prob[i-1] + drift, 0.01, 0.99)
        
    dates = pd.date_range(start='2025-01-01', periods=n_steps, freq='h')
    
    poly_yes = true_prob + np.random.normal(0, 0.005, n_steps)
    kalshi_yes = true_prob + np.random.normal(0, 0.01, n_steps)
    
    poly_ask_yes = poly_yes + 0.01
    poly_ask_no = (1 - poly_yes) + 0.01 
    kalshi_ask_yes = kalshi_yes + 0.02
    kalshi_ask_no = (1 - kalshi_yes) + 0.02
    
    # Inject Inefficiencies
    for i in range(n_steps):
        if np.random.random() < 0.03: # 3% chance (slightly rarer per market)
            shock = np.random.uniform(0.02, 0.05)
            if np.random.random() < 0.5: poly_ask_yes[i] -= shock
            else: poly_ask_no[i] -= shock
                
        if np.random.random() < 0.03:
            shock = np.random.uniform(0.03, 0.08)
            if np.random.random() < 0.5: kalshi_ask_yes[i] += shock
            else: kalshi_ask_yes[i] -= shock

    poly_ask_yes = np.clip(poly_ask_yes, 0.01, 0.99)
    poly_ask_no = np.clip(poly_ask_no, 0.01, 0.99)
    kalshi_ask_yes = np.clip(kalshi_ask_yes, 0.01, 0.99)
    kalshi_ask_no = np.clip(kalshi_ask_no, 0.01, 0.99)

    return pd.DataFrame({
        'Date': dates,
        'Poly_Ask_Yes': poly_ask_yes,
        'Poly_Ask_No': poly_ask_no,
        'Kalshi_Ask_Yes': kalshi_ask_yes,
        'Kalshi_Ask_No': kalshi_ask_no
    })

def generate_multi_market_universe(num_markets=50):
    universe = {}
    print(f"   Generating {num_markets} synthetic markets...")
    for i in range(num_markets):
        universe[f"Mkt_{i+1}"] = generate_single_market(seed=42+i)
    return universe

# =============================================================================
# STRATEGIES
# =============================================================================

def run_multi_market_dutch_book(universe, capital=10000):
    balance = capital
    trades = 0
    max_trade_size = 1000 
    
    # We need to iterate chronologically across ALL markets
    # Assume all align on index (they do)
    
    market_ids = list(universe.keys())
    n_steps = len(universe[market_ids[0]])
    
    # To save memory, we won't deep copy everything. We'll just track PnL.
    
    equity_curve = [balance]
    
    for t in range(n_steps):
        
        # Check every market for an opportunity at this hour
        # (In reality, bots check ms by ms. We simulate 'batch' checking)
        
        for m_id in market_ids:
            row = universe[m_id].iloc[t]
            
            cost = row['Poly_Ask_Yes'] + row['Poly_Ask_No']
            
            if cost < 0.99:
                # Opp found
                invest = min(balance, max_trade_size)
                
                # If we are out of cash, we can't trade.
                # Simplified: instant settlement for next opp? 
                # Ideally, capital is tied up until resolution.
                # BUT, Dutch Book arbs are often instant-realizable if you sell back.
                # Here, we treat it as "Instant Profit" (Arb & Liquidation).
                
                if invest > 100: # Min trade
                    profit = (invest / cost * 1.0) - invest
                    balance += profit
                    trades += 1
        
        equity_curve.append(balance)
            
    return balance, trades, equity_curve

def run_multi_market_cross_arb(universe, capital=10000):
    balance = capital
    trades = 0
    max_trade_size = 1000
    fee_drag = 0.02
    
    market_ids = list(universe.keys())
    n_steps = len(universe[market_ids[0]])
    equity_curve = [balance]
    
    for t in range(n_steps):
        for m_id in market_ids:
            row = universe[m_id].iloc[t]
            
            bundle_1 = row['Poly_Ask_Yes'] + row['Kalshi_Ask_No']
            bundle_2 = row['Kalshi_Ask_Yes'] + row['Poly_Ask_No']
            best = min(bundle_1, bundle_2)
            
            if best < (1.0 - fee_drag):
                invest = min(balance, max_trade_size)
                if invest > 100:
                    profit = (invest / best * 1.0) - invest - (invest * fee_drag)
                    balance += profit
                    trades += 1
        
        equity_curve.append(balance)

    return balance, trades, equity_curve

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(start_cap, end_cap, trades, name="Strategy"):
    total_ret = (end_cap - start_cap) / start_cap
    days = 90
    ann_ret = ((1 + total_ret) ** (365/days)) - 1
    
    with open('results_multi.txt', 'a') as f:
        f.write(f"\n{name} RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Starting Capital: ${start_cap:,.2f}\n")
        f.write(f"Ending Capital:   ${end_cap:,.2f}\n")
        f.write(f"Total Return:     {total_ret*100:.2f}%\n")
        f.write(f"Annualized Ret:   {ann_ret*100:.2f}%\n")
        f.write(f"Total Trades:     {trades}\n")
        f.write(f"Avg Profit/Trade: ${(end_cap-start_cap)/trades if trades > 0 else 0:.2f}\n")
    
    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"Starting Capital: ${start_cap:,.2f}")
    print(f"Ending Capital:   ${end_cap:,.2f}")
    print(f"Total Return:     {total_ret*100:.2f}%")
    print(f"Annualized Ret:   {ann_ret*100:.2f}%")
    print(f"Total Trades:     {trades}")
    print(f"Avg Profit/Trade: ${(end_cap-start_cap)/trades if trades > 0 else 0:.2f}")

    return total_ret

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🔮 GENERATING MULTI-MARKET DATA (50 Markets)...")
    universe = generate_multi_market_universe(50)
    
    print("\n⚔️ RUNNING SCALED ARBITRAGE BACKTESTS...")
    
    if (pd.io.common.file_exists('results_multi.txt')):
        open('results_multi.txt', 'w').close() # Clear file
    
    init_cash = 10000
    
    # 1. Dutch Book
    end_db, trades_db, _ = run_multi_market_dutch_book(universe, init_cash)
    analyze_results(init_cash, end_db, trades_db, "Multi-Market Dutch Book (50 Mkts)")
    
    # 2. Cross Market
    end_cm, trades_cm, _ = run_multi_market_cross_arb(universe, init_cash)
    analyze_results(init_cash, end_cm, trades_cm, "Multi-Market Cross Arb (50 Mkts)")
    
    print("\n✅ DONE.")
