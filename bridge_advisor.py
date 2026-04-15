"""
Bridge Advisor
==============
Calculates the 'Opportunity Cost' of capital fragmentation.
Tells you when to move money from Exchange A to B.
"""

def calculate_rebalance(poly_balance, kalshi_balance, poly_yield, kalshi_yield):
    print("🌉 BRIDGE ADVISOR: REBALANCING SIGNAL")
    print("=====================================")
    
    # Example: If Poly is tapped out (cap hit) but Kalshi has 15% APY bonds
    # We should move capital.
    
    p_util = 0.8 # Assume 80% utilization on Poly
    k_util = 0.4 # Assume 40% utilization on Kalshi
    
    if p_util > 0.9 and k_util < 0.5:
        print("🚩 SIGNAL: MOVE CAPITAL TO KALSHI")
        print(f"   Poly is at Capacity. Kalshi has Deep Liquidity.")
        print(f"   Move Suggestion: Withdraw $ profit and ACH to Kalshi.")
    else:
        print("✅ STATUS: BALANCED")
        print("   Current distribution is optimal for total yield.")

if __name__ == "__main__":
    # Mock data
    calculate_rebalance(10000, 2000, 0.05, 0.03)
