def get_current_yield(balance):
    """Adjusts yield based on liquidity/slippage as fund grows."""
    if balance < 2000: return 0.05      # 5% weekly (PHASE 1: POLYMARKET ONLY)
    if balance < 50000: return 0.03     # 3% weekly (PHASE 2: GLOBAL ARBITRAGE)
    if balance < 250000: return 0.015   # 1.5% weekly (PHASE 3: INSTITUTIONAL BONDS)
    return 0.0075                       # 0.75% weekly (PHASE 4: MACRO WHALE)

def simulate_to_million(start_cap=200, weekly_add=100):
    balance = start_cap
    week = 0
    history = []
    
    print(f"{'Year':<5} {'Week':<5} {'Balance':<15} {'Yield%'}")
    print("-" * 45)
    
    while balance < 100000000:
        week += 1
        # 1. Add deposit
        balance += weekly_add
        
        # 2. Add Yield
        yield_rate = get_current_yield(balance)
        balance += (balance * yield_rate)
        
        # Print snapshots
        if week % 26 == 0 or week == 1:
            year = week // 52
            print(f"{year:<5} {week:<5} ${balance:<14,.2f} {yield_rate*100:.1f}%")
        
        if week > 52 * 10: break # Safety stop at 10 years
            
    return week, balance

def run_comparison(start_cap=200, weekly_add=100, weeks=52):
    # Scenario 1: Poly Only (Yield only, no arbs)
    poly_rate = 0.04 
    # Scenario 2: Global (Yield + 2% extra from Arbs)
    global_rate = 0.06
    
    p_bal = start_cap
    g_bal = start_cap
    
    print(f"{'Week':<5} | {'Poly-Only ($)':<15} | {'Global-Empire ($)':<18} | {'Alpha Lost'}")
    print("-" * 65)
    
    for w in range(1, weeks + 1):
        p_bal = (p_bal + weekly_add) * (1 + poly_rate)
        g_bal = (g_bal + weekly_add) * (1 + global_rate)
        
        if w % 13 == 0 or w == 52:
            loss = g_bal - p_bal
            print(f"{w:<5} | ${p_bal:<14,.2f} | ${g_bal:<17,.2f} | -${loss:,.2f}")

if __name__ == "__main__":
    print("⚖️ ALPHA LOSS ANALYSIS: THE COST OF SIMPLICITY")
    print("Scenario: $200 Start + $100/Week Deposit\n")
    run_comparison()
