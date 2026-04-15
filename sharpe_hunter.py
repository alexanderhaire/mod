"""
Sharpe Hunter (The Fort Knox Strategy)
======================================

Finds the "Absolute Highest Sharpe" digital bonds.
Logic: We want to maximize Yield while minimizing variance (probability of loss).

Metric:
  SharpeScore = APY / (1.0 - Price)

  * A bond at 0.99 with 50% APY -> Score 5000.
  * A bond at 0.90 with 50% APY -> Score 500.

This strategy aggressively prefers SAFETY.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import dateutil.parser

GAMMA_API = "https://gamma-api.polymarket.com"

def run_sharpe_hunt(limit=500):
    print("🛡️ SHARPE HUNTER (FORT KNOX)")
    print("============================")
    
    try:
        res = requests.get(f"{GAMMA_API}/events", params={"limit": limit, "closed": "false", "order": "volume24hr", "ascending": "false"})
        events = res.json()
    except:
        return

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
        
        # FILTER: Price must be > 0.90 to even participate in "Bond" class
        if max_p < 0.92: continue
        if max_p >= 0.999: continue # No yield
        
        # Outcomes
        outcomes = m.get('outcomes', [])
        if isinstance(outcomes, str):
            import json
            try: outcomes = json.loads(outcomes)
            except: pass
            
        label = str(max_idx)
        if outcomes and len(outcomes) > max_idx:
            label = outcomes[max_idx]
            
        # Metrics
        raw_yield = (1.0 - max_p) / max_p
        apy = raw_yield * (365 / days)
        
        # Risk (Implied Probability of Failure)
        risk_prob = 1.0 - max_p
        
        # Sharpe Proxy
        # Avoid div by zero
        if risk_prob < 0.001: risk_prob = 0.001
        
        sharpe_score = apy / risk_prob
        
        candidates.append({
            "Question": m.get('question'),
            "Outcome": label,
            "Price": max_p,
            "Days": days,
            "APY": apy,
            "RiskProb": risk_prob,
            "Score": sharpe_score
        })
        
    df = pd.DataFrame(candidates)
    if df.empty:
        print("   No Safe Bonds found.")
        return

    # Sort by Sharpe Score
    df = df.sort_values('Score', ascending=False)
    
    print(f"\n🏆 THE FORT KNOX PORTFOLIO (Top 20 by Risk-Adjusted Return)")
    print(f"{'Market (Short)':<35} {'Outcome':<8} {'Price':<6} {'APY%':<6} {'Risk(%)':<8} {'SharpeScore'}")
    print("-" * 100)
    
    top_20 = df.head(20)
    for _, row in top_20.iterrows():
        q_short = row['Question'][:32] + "..."
        print(f"{q_short:<35} {row['Outcome']:<8} {row['Price']:<6.2f} {row['APY']*100:<6.0f}% {row['RiskProb']*100:<8.1f} {row['Score']:.1f}")
        
    avg_score = top_20['Score'].mean()
    print("-" * 100)
    print(f"✨ Portfolio Avg Sharpe Score: {avg_score:.0f}")
    print(f" This score represents (Return / Risk). A score of 5000+ is legendary.")

if __name__ == "__main__":
    run_sharpe_hunt()
