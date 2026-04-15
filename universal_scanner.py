"""
Universal Market Scanner (Cross-Exchange Arb)
=============================================

Scans multiple prediction exchanges to find mispricings.
Venues:
1. Polymarket (Crypto / Global)
2. Manifold Markets (Sentiment / Play Money)
3. Kalshi (Regulated / US Institutional) - *Simulated Public Data if API restricted*

Goal: Find the SAME event trading at DIFFERENT prices.
"""

import requests
import pandas as pd
from fuzzywuzzy import fuzz
import time

class PolymarketAdapter:
    def fetch(self):
        print("   📡 Polling Polymarket...")
        url = "https://gamma-api.polymarket.com/events?limit=100&closed=false&order=volume24hr&ascending=false"
        try:
            res = requests.get(url).json()
            markets = []
            for e in res:
                mkts = e.get('markets', [])
                if not mkts: continue
                m = mkts[0]
                
                # Get prob
                prices = m.get('outcomePrices', [])
                try: 
                    if isinstance(prices, str): import json; prices = json.loads(prices)
                    prob = float(prices[0]) if prices else 0.5
                except: prob = 0.5
                
                markets.append({
                    "Question": m.get('question'),
                    "Prob": prob,
                    "Source": "Polymarket",
                    "ID": m.get('id')
                })
            return markets
        except Exception as e:
            print(f"   ❌ Poly Error: {e}")
            return []

class ManifoldAdapter:
    def fetch(self):
        print("   🔮 Polling Manifold...")
        url = "https://api.manifold.markets/v0/markets?limit=100"
        try:
            res = requests.get(url).json()
            markets = []
            for m in res:
                if 'probability' not in m: continue
                markets.append({
                    "Question": m.get('question'),
                    "Prob": float(m.get('probability')),
                    "Source": "Manifold",
                    "ID": m.get('id'),
                    "EndDate": m.get('closeTime') # Unix MS
                })
            return markets
        except Exception as e:
            # print(f"   ❌ Manifold Error: {e}")
            return []

class KalshiAdapter:
    def fetch(self):
        print("   🏛️ Polling Kalshi (Public)...")
        url = "https://api.elections.kalshi.com/trade-api/v2/markets?limit=100" 
        try:
            res = requests.get(url, timeout=3)
            if res.status_code != 200: return []
            
            data = res.json()
            markets = []
            for m in data.get('markets', []):
                 # Kalshi often gives 'close_time' or similar
                 markets.append({
                    "Question": m.get('title'),
                    "Prob": m.get('last_price', 50) / 100.0,
                    "Source": "Kalshi",
                    "ID": m.get('ticker'),
                    "EndDate": m.get('close_time') 
                })
            return markets
        except:
            return []

def main():
    print("🌍 UNIVERSAL PREDICTION SCANNER")
    print("==============================")
    
    poly = PolymarketAdapter().fetch()
    mani = ManifoldAdapter().fetch()
    kalshi = KalshiAdapter().fetch()
    
    all_markets = poly + mani + kalshi
    print(f"   Loaded {len(all_markets)} total markets.")
    
    # Matching Logic
    # We take Polymarket as the "Base" and look for matches in others.
    
    print("\n⚔️ CROSS-EXCHANGE ARBITRAGE OPPORTUNITIES")
    print(f"{'Event (Short)':<40} {'Poly%':<6} {'Other%':<6} {'Spread':<6} {'Venue'}")
    print("-" * 100)
    
    matches_found = 0
    
    for p in poly:
        # Compare against Manifold
        for m in mani:
            # Fuzzy match titles
            ratio = fuzz.ratio(p['Question'].lower(), m['Question'].lower())
            if ratio > 85: # High confidence match
                spread = p['Prob'] - m['Prob']
                if abs(spread) > 0.05: # > 5% difference
                    q_short = p['Question'][:38] + ".."
                    print(f"{q_short:<40} {p['Prob']:<6.2f} {m['Prob']:<6.2f} {spread * 100:+.1f}%  Manifold")
                    matches_found += 1
        
        # Compare against Kalshi (if any)
        for k in kalshi:
            ratio = fuzz.ratio(p['Question'].lower(), k['Question'].lower())
            if ratio > 85:
                spread = p['Prob'] - k['Prob']
                if abs(spread) > 0.05:
                    q_short = p['Question'][:38] + ".."
                    print(f"{q_short:<40} {p['Prob']:<6.2f} {k['Prob']:<6.2f} {spread * 100:+.1f}%  Kalshi")
                    matches_found += 1

    if matches_found == 0:
        print("   No spread > 5% found today between exchanges.")
    else:
        print(f"\n   ✅ Found {matches_found} arb opportunities.")

if __name__ == "__main__":
    main()
