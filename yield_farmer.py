"""
Yield Farming Scanner (Digital Bonds)
=====================================

Treats deep-in-the-money prediction markets as "Fixed Income" bonds.
Calculates the Annualized Yield (APY) of holding a position to maturity.

Formula:
  Yield = (Payout - Price) / Price
  APY = Yield * (365 / DaysToExpiry)

Target:
  - Price > 0.85 (High Confidence)
  - APY > 15% (Better than Treasury Bills)
  - Volatility < Threshold (Stable)
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import dateutil.parser

GAMMA_API = "https://gamma-api.polymarket.com"

def fetch_yield_opportunities(limit=500):
    print("🌾 YIELD FARMING SCANNER")
    print("=========================")
    print("   Fetching markets...")
    
    try:
        # Fetch active markets. Order by volume to ensure liquidity (exit hatch).
        res = requests.get(f"{GAMMA_API}/events", params={"limit": limit, "closed": "false", "order": "volume24hr", "ascending": "false"})
        events = res.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
    
    rows = []
    now = datetime.now(timezone.utc)
    
    print(f"   Scanning {len(events)} events for Digital Bonds...")
    
    if len(events) > 0:
        print(f"   DEBUG Raw Sample: {events[0].keys()}")
        print(f"   DEBUG Sample MKTS: {events[0].get('markets', [])[0].keys()}")

    for e in events:
        mkts = e.get('markets', [])
        if not mkts: continue
        m = mkts[0]
        
        # Get End Date
        end_date_str = m.get('endDate') # ISO format
        if not end_date_str: continue
        
        try:
            end_date = dateutil.parser.isoparse(end_date_str)
            if end_date <= now: continue # Expired
            
            # Days to Expiry
            delta = end_date - now
            days_to_expiry = delta.total_seconds() / (24 * 3600)
            if days_to_expiry < 0.1: continue # Too short
            
        except:
            continue
            
        # Analyze Outcomes (Binary)
        # We look for the "High Prob" outcome
        outcomes = m.get('outcomes', [])
        prices = m.get('outcomePrices', [])
        
        # Robust Parse
        if isinstance(prices, str):
            import json
            try: prices = json.loads(prices)
            except: pass
            
        try:
            prices = [float(p) for p in prices]
        except:
             continue
             
        if not prices: 
            # print(f"DEBUG: Skipping {m.get('question')} (No Prices)")
            continue
        
        # Find Max Price Outcome
        max_p = max(prices)
        max_idx = prices.index(max_p)
        
        # DEBUG
        # print(f"DEBUG: {m.get('question')} | MaxP: {max_p} | Vol: {m.get('volume24hr')}")
        
        # Filter: Must be "Safe" (> 0.85 cents)
        # Relaxed filter for debugging: > 0.60
        if max_p < 0.60 or max_p >= 0.999: 
            # print(f"DEBUG: Skipped {m.get('question')} (Price {max_p} out of range)")
            continue
            
        # Calculate Yield
        # Payout is $1.00
        raw_yield = (1.0 - max_p) / max_p
        
        # Annualize (APY)
        apy = raw_yield * (365 / days_to_expiry)
        
        # Interpret Label
        label = "Unknown"
        if outcomes and len(outcomes) > max_idx:
            label = outcomes[max_idx]
        elif len(outcomes) == 2:
            label = "Yes" if max_idx == 0 else "No" # Usually
            
        # Volatility Check (Mock - assume safe if volume is high)
        vol_24h = float(m.get('volume24hr', 0) or 0)
        if vol_24h < 1000: continue # Illiquid bond is dangerous
        
        rows.append({
            "Question": m.get('question'),
            "Outcome": label,
            "Price": max_p,
            "DaysLeft": days_to_expiry,
            "Yield%": raw_yield * 100,
            "APY%": apy * 100,
            "Volume": vol_24h
        })
        
    if not rows:
        print("   No valid bonds found matching criteria.")
        return
    
    df = pd.DataFrame(rows)
    # Sort by APY desc, but filter crazy APY (likely bad data)
    df = df[df['APY%'] < 500] 
    df = df.sort_values("APY%", ascending=False)
    
    print(f"\n🏆 TOP DIGITAL BONDS (Yield Farming)")
    print(f"   Found {len(df)} opportunities right now.")
    print(f"{'Market (Short)':<35} {'Outcome':<8} {'Price':<6} {'Days':<6} {'APY%':<6} {'Risk'}")
    print("-" * 100)
    
    for _, row in df.head(15).iterrows():
        q_short = row['Question'][:32] + "..."
        risk = "🟢 Low" if row['Price'] > 0.92 else "🟡 Med"
        
        print(f"{q_short:<35} {row['Outcome']:<8} {row['Price']:<6.2f} {row['DaysLeft']:<6.1f} {row['APY%']:<6.0f}% {risk}")

if __name__ == "__main__":
    fetch_yield_opportunities()
