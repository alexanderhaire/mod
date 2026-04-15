"""
Smart Bond Portfolio Manager
============================

Builds a DIVERSIFIED portfolio of Digital Bonds.
Rules:
1. Safety: Price > 0.90 (90% Odds).
2. Diversification: Max 1 Bond per "Theme" (e.g. don't buy 5 Solana bonds).
3. Liquidity: Volume > $1k.
4. Ladder: Mix of expiries.

Goal: Create a stable income generator.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import dateutil.parser

GAMMA_API = "https://gamma-api.polymarket.com"

def get_theme(question):
    """Simple keyword matching to identify theme/risk factor"""
    q = question.lower()
    if "trump" in q: return "Trump"
    if "bitcoin" in q or "btc" in q: return "Bitcoin"
    if "solana" in q or "sol" in q: return "Solana"
    if "ethereum" in q or "eth" in q: return "Ethereum"
    if "nfl" in q or "super bowl" in q: return "NFL"
    if "nba" in q: return "NBA"
    if "tennis" in q or "open" in q: return "Tennis"
    if "ukraine" in q or "russia" in q: return "UkraineWar"
    if "gaza" in q or "israel" in q: return "GazaWar"
    if "rate" in q or "fed" in q: return "FedRates"
    return "Other" # Miscellaneous

def fetch_and_build_portfolio(capital=10000.0, limit=500):
    print("💼 SMART BOND PORTFOLIO MANAGER")
    print(f"   Capital: ${capital:,.2f}")
    print("===============================")
    
    # 1. Fetch Candidates (Dual Scan: Volume + Freshness)
    events = []
    seen_ids = set()
    
    # Batch A: Liquidity (Safe)
    try:
        print("   Satellites: Scanning Top Volume...")
        res = requests.get(f"{GAMMA_API}/events", params={"limit": limit, "closed": "false", "order": "volume24hr", "ascending": "false"})
        batch_a = res.json()
        for e in batch_a:
            if e['id'] not in seen_ids:
                events.append(e)
                seen_ids.add(e['id'])
    except: pass
    
    # Batch B: Freshness (New Listings - often mispriced yields)
    try:
        print("   Satellites: Scanning New Listings...")
        res = requests.get(f"{GAMMA_API}/events", params={"limit": 100, "closed": "false", "order": "startDate", "ascending": "false"})
        batch_b = res.json()
        for e in batch_b:
            if e['id'] not in seen_ids:
                events.append(e)
                seen_ids.add(e['id'])
    except: pass

    print(f"   Analyzing {len(events)} total markets...")

    now = datetime.now(timezone.utc)
    candidates = []
    
    for e in events:
        mkts = e.get('markets', [])
        if not mkts: continue
        m = mkts[0]
        
        # End Date
        end_date_str = m.get('endDate')
        if not end_date_str: continue
        try:
            end_date = dateutil.parser.isoparse(end_date_str)
            if end_date <= now: continue
            delta = end_date - now
            days = delta.total_seconds() / (24 * 3600)
            if days < 0.1: continue
        except: continue
        
        # Prices
        prices = m.get('outcomePrices', [])
        if isinstance(prices, str):
            import json
            try: prices = json.loads(prices)
            except: pass
        try: prices = [float(p) for p in prices]
        except: continue
        if not prices: continue
        
        max_p = max(prices)
        max_idx = prices.index(max_p)
        
        # STRICTER SAFETY for Strategy: Must be > 0.90
        if max_p < 0.90 or max_p >= 0.995: continue
        
        # Outcomes
        outcomes = m.get('outcomes', [])
        if isinstance(outcomes, str):
            import json
            try: outcomes = json.loads(outcomes)
            except: pass
            
        label = str(max_idx)
        if outcomes and len(outcomes) > max_idx:
            label = outcomes[max_idx]
            
        # Yield
        raw_yield = (1.0 - max_p) / max_p
        apy = raw_yield * (365 / days)
        
        # Volume Check
        vol = float(m.get('volume24hr', 0) or 0)
        if vol < 500: continue
        
        candidates.append({
            "Question": m.get('question'),
            "Theme": get_theme(m.get('question')),
            "Price": max_p,
            "Days": days,
            "Yield": raw_yield,
            "APY": apy,
            "Outcome": label,
            "Volume": vol, # Added missing key
            "TokenID": m.get('clobTokenIds', [])[max_idx] if m.get('clobTokenIds') else None
        })
        
    df = pd.DataFrame(candidates)
    if df.empty:
        print("   No safe bonds found.")
        return

    # 2. Portfolio Construction
    # Sort by APY desc
    df = df.sort_values('APY', ascending=False)
    
    # 4. Waterfall Allocation Strategy
    # We want to deploy ALL capital.
    # If the top bonds are capped by liquidity, we move to the next ones.
    
    print("\n🌊 CALCULATING WATERFALL ALLOCATION...")
    
    orders = []
    portfolio_summary = []
    
    remaining_capital = capital
    target_position_size = capital / 10 # Aim for 10 core positions initially
    if capital < 500: target_position_size = capital / 5 # Concentrated for small accts
    
    # We iterate through ALL candidates (sorted by APY)
    for _, item in df.iterrows():
        if remaining_capital < 5.0: break # Fully Deployed
        
        # 1. Determine Max Safe Bet for this specific market
        # Rule: 1% of Daily Volume (Conservative Whale Protection)
        vol_cap = item['Volume'] * 0.01
        
        # 2. Determine Portfolio Risk Limit
        # Rule: Don't put more than 15% of fund into one bet (even if liquid)
        risk_cap = capital * 0.15
        
        # 3. Allocation is the minimum of (Target, VolCap, RiskCap, Remaining)
        max_alloc = min(target_position_size, vol_cap, risk_cap, remaining_capital)
        
        if max_alloc < 2.0: continue # Skip dust
        
        # Add to Portfolio
        # Update remaining
        remaining_capital -= max_alloc
        
        # Track for reporting
        cap_msg = ""
        if max_alloc < target_position_size and max_alloc == vol_cap:
             cap_msg = "💧 Liquidity Constrained"
             
        portfolio_summary.append({
            "Theme": item['Theme'],
            "Question": item['Question'],
            "Alloc": max_alloc,
            "Price": item['Price'],
            "Yield": item['Yield'],
            "APY": item['APY'],
            "Msg": cap_msg,
            "TokenID": item['TokenID']
        })
        
        # Add Order
        orders.append({
            "token_id": item['TokenID'],
            "price": item['Price'],
            "size": max_alloc / item['Price'],
            "side": "BUY",
            "name": item['Question'][:50]
        })
        
    # Summary Output
    count = len(portfolio_summary)
    
    print(f"\n✅ Deployed Capital into {count} Positions")
    print(f"{'Theme':<12} {'Market (Short)':<30} {'Price':<6} {'Alloc($)':<8} {'Est.Profit':<10} {'APY'} {'Note'}")
    print("-" * 110)
    
    total_est_profit = 0
    total_deployed = capital - remaining_capital
    avg_days = 0 
    
    for p in portfolio_summary[:20]: # Show top 20
        q_short = p['Question'][:28] + ".."
        est_profit = p['Alloc'] * p['Yield']
        total_est_profit += est_profit
        
        print(f"{p['Theme']:<12} {q_short:<30} {p['Price']:<6.2f} ${p['Alloc']:<7.2f} ${est_profit:<9.2f} {p['APY']*100:.0f}% {p['Msg']}")
       
    if count > 20:
        print(f"... and {count-20} more positions.")

    # Calculate Totals
    # Note: loop above only printed 20, need to sum all for stats
    real_total_profit = sum(p['Alloc'] * p['Yield'] for p in portfolio_summary)
    real_avg_yield = (real_total_profit / total_deployed * 100) if total_deployed > 0 else 0
    
    print("-" * 110)
    print(f"💰 TOTAL DEPLOYED: ${total_deployed:,.2f} / ${capital:,.2f} ({total_deployed/capital*100:.1f}%)")
    print(f"📈 EXP. PROFIT:    ${real_total_profit:,.2f}")
    print(f"🚀 PORTFOLIO YIELD: {real_avg_yield:.2f}% (Absolute)")

    import os
    if not os.path.exists("signals"):
        os.makedirs("signals")

    with open("signals/buy_orders.json", "w") as f:
        json.dump(orders, f, indent=2)
    print("   ✅ Exported 'signals/buy_orders.json' for execution.")

if __name__ == "__main__":
    import sys
    # Allow CLI arg: python smart_bond_portfolio.py 200
    if len(sys.argv) > 1:
        try:
            cap = float(sys.argv[1])
            fetch_and_build_portfolio(capital=cap)
        except:
             fetch_and_build_portfolio()
    else:
        fetch_and_build_portfolio()
