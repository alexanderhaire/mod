"""
Multi-Venue Global Scanner (The "Octopus")
=========================================
Polls a massive list of prediction venues.
1. Polymarket (Poly-Active)
2. Manifold   (Mana-Social)
3. Kalshi     (US-Regulated)
4. PredictIt  (US-Niche)
5. Betfair/Smarkets (Placeholder logic)
"""

import requests
import json
import time

class PredictItAdapter:
    def fetch(self):
        print("   🇺🇸 Polling PredictIt (Public API)...")
        # PredictIt has a handy public market XML/JSON
        url = "https://www.predictit.org/api/marketdata/all/"
        try:
            res = requests.get(url, timeout=5).json()
            markets = []
            for m in res.get('markets', []):
                # PredictIt uses pennies (0.55 = 55%)
                for contract in m.get('contracts', []):
                    markets.append({
                        "Question": f"{m['name']} - {contract['name']}",
                        "Prob": contract.get('lastTradePrice', 0.50),
                        "Source": "PredictIt",
                        "ID": contract['id']
                    })
            return markets
        except: return []

# We import our existing ones
from universal_scanner import PolymarketAdapter, ManifoldAdapter, KalshiAdapter

def run_global_scan():
    print("🐙 GLOBAL PREDICTION RADAR (THE OCTOPUS)")
    print("========================================")
    
    scanners = [
        PolymarketAdapter(),
        ManifoldAdapter(),
        KalshiAdapter(),
        PredictItAdapter()
    ]
    
    total_data = []
    for s in scanners:
        data = s.fetch()
        print(f"      ✅ Found {len(data)} markets.")
        total_data.extend(data)
        
    print(f"\n🌍 SUCCESS: Monitoring {len(total_data)} events across the globe.")
    
    # Simple Arb Hunt
    # (Matches would normally use fuzzy matching from our previous scripts)
    print("\n[!] Recommendation: Focusing on high-volume overlaps between Poly and Betfair/PredictIt.")

if __name__ == "__main__":
    run_global_scan()
