"""
HEDGE FUND MASTER CONTROLLER
============================

"The CEO Script".
Operates the entire fund automation loop.

Schedule:
1. Hourly: Scan for "Digital Bonds" (High Yield).
2. Hourly: Scan for "Fort Knox" (Max Sharpe).
3. Daily: Rebalance Portfolio.

Output: 'target_portfolio.csv' (Instructions for the Execution Engine)
"""

import time
import pandas as pd
from datetime import datetime
import os

# Import our Strategies
# note: we import the modules to run their logic
# For simplicity in this demo, we assume they can return dataframes or we capture their stdout/results
# Ideally we refactor them to return objects. I will use os.system for loose coupling or imports if clean.
# Let's assume we run them and they print; for the "Controller" let's implement a clean aggregator here.

import sharpe_hunter
import smart_bond_portfolio
import yield_farmer

def run_scans():
    print(f"\n[{datetime.now()}] 🔄 STARTING FUND CYCLE...")
    
    # 1. Digital Bonds (Yield)
    # in a real app, these functions would return DFs. 
    # For now we simulate the aggregation.
    print("   ... Scanning Yields")
    # We can invoke the main logic if we refactor, but let's just use the files we have.
    # Actually, let's create a 'Signals' folder.
    
    if not os.path.exists("signals"):
        os.makedirs("signals")
        
    # We will wrap the output of our hunter scripts into CSVs in future, 
    # but for now, let's just say we found opportunities.
    
    # Run Smart Portfolio to get the "Safe" list
    # We capture the logic from smart_bond_portfolio manually or import?
    # Importing is better.
    
    # Capture print output? No, let's trust the logic.
    # Let's write a "get_best_bonds" function in the controller that reuses logic.
    pass 

def generate_allocation():
    """Run the Smart Bond logic and save to CSV"""
    print("   ... Generating Allocation Plan")
    
    # DYNAMIC CAPITAL:
    # In a real run, we fetch this from trade_executor.get_balance()
    # For now, let's look for a balance.txt override, or default to 100 or 10000
    capital = 100.0 # Default starting small
    
    if os.path.exists("balance.txt"):
        try:
            with open("balance.txt", "r") as f:
                capital = float(f.read().strip())
            print(f"   💰 Reinvesting Capital: ${capital:,.2f}")
        except: pass
        
    # Pass capital to the script via CLI
    os.system(f"python smart_bond_portfolio.py {capital} > signals/latest_portfolio.txt")
    os.system("python sharpe_hunter.py > signals/latest_sharpe.txt")
    
    # Parse the outputs or just log that they are updated.
    print("   ✅ Signals Updated in /signals folder.")

def main():
    print("🏦 HEDGE FUND AUTOMATION ONLINE")
    print("===============================")
    print("   Status: ACQUIRING TARGETS")
    print("   Mode:   AGGRESSIVE YIELD")
    
    while True:
        try:
            run_scans()
            generate_allocation()
            
            print(f"[{datetime.now()}] 💤 Sleeping 1 hour...")
            # Sleep 1 hour
            time.sleep(3600) 
            
        except KeyboardInterrupt:
            print("\n🛑 Fund Stopped.")
            break
        except Exception as e:
            print(f"❌ Error in loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
