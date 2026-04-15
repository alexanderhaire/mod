"""
Manifold Markets Data Fetcher
=============================

Fetches active markets from Manifold Markets (Play Money).
Useful as a 'Crowd Sentiment' signal to compare against Real Money markets.

"""

import requests
import pandas as pd
import os
import time

MANIFOLD_API = "https://api.manifold.markets/v0"

def fetch_manifold_markets(limit=500):
    print("📡 Fetching Manifold markets...")
    url = f"{MANIFOLD_API}/markets"
    try:
        # Get binary markets only for easy comparison
        res = requests.get(url, params={"limit": limit})
        if res.status_code != 200:
            print(f"❌ Error {res.status_code}")
            return []
            
        data = res.json()
        print(f"   Found {len(data)} markets.")
        return data
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def main():
    print("🦄 MANIFOLD MARKETS FETCH")
    print("=======================")
    
    markets = fetch_manifold_markets()
    if not markets:
        return
        
    rows = []
    for m in markets:
        # We focus on Binary (YES/NO) markets for direct comparison
        if m.get('outcomeType') != 'BINARY':
            continue
            
        # Manifold probability is 'probability' field (0-1)
        prob = m.get('probability', 0)
        
        rows.append({
            "ID": m.get('id'),
            "Question": m.get('question'),
            "Probability": prob,
            "Volume24h": m.get('volume24Hours', 0),
            "TotalVolume": m.get('volume', 0),
            "CloseTime": m.get('closeTime')
        })
        
    df = pd.DataFrame(rows)
    
    # Save
    out_dir = "data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path = os.path.join(out_dir, "manifold_markets.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df)} binary markets to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
