"""
Kalshi Data Fetcher
===================

Fetches active markets from Kalshi Public API v2.
No auth required for public market data.

"""

import requests
import pandas as pd
import os
import time

# Public endpoint for market data (no auth needed)
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

def fetch_series(category):
    print(f"   Fetching series for category: {category}...")
    url = f"{KALSHI_API}/series"
    try:
        res = requests.get(url, params={"category": category})
        if res.status_code != 200:
            print(f"   ⚠️ Error fetching series: {res.status_code}")
            return []
        data = res.json()
        return data.get('series', [])
    except Exception as e:
        print(f"   ⚠️ Exception: {e}")
        return []

def fetch_markets_by_series(series_ticker):
    url = f"{KALSHI_API}/markets"
    try:
        res = requests.get(url, params={"series_ticker": series_ticker, "limit": 100, "status": "active"})
        # Note: status might be 'open' or 'active', try 'active' or remove if fails.
        # Let's try minimal params if 400 again.
        if res.status_code == 400:
             res = requests.get(url, params={"series_ticker": series_ticker, "limit": 100})
             
        if res.status_code != 200:
            return []
        data = res.json()
        return data.get('markets', [])
    except:
        return []

def main():
    print("🚀 KALSHI REAL DATA FETCH (POLITICS/ECON)")
    print("="*60)
    
    categories = ["politics", "economics"]
    all_markets = []
    
    for cat in categories:
        series_list = fetch_series(cat)
        print(f"   Found {len(series_list)} series in {cat}.")
        
        for s in series_list:
            ticker = s.get('ticker')
            title = s.get('title')
            # Filter for keywords to save time?
            # Let's get everything for now to ensuring matching.
            
            markets = fetch_markets_by_series(ticker)
            if markets:
                # print(f"      Found {len(markets)} markets for {ticker}")
                all_markets.extend(markets)
            
            time.sleep(0.1) # Rate limit
    
    if not all_markets:
        print("⚠️ No markets found in target categories.")
        return
        
    # Process relevant fields
    rows = []
    for m in all_markets:
        title = m.get('title', '')
        subtitle = m.get('subtitle', '')
        ticker = m.get('ticker', '')
        
        # Prices are usually in cents (1-99). Convert to 0.00-1.00
        yes_bid = m.get('yes_bid', 0) / 100.0
        yes_ask = m.get('yes_ask', 0) / 100.0
        last_price = m.get('last_price', 0) / 100.0
        
        rows.append({
            "Ticker": ticker,
            "Event": title,
            "Subtitle": subtitle,
            "Bid": yes_bid,
            "Ask": yes_ask,
            "Last": last_price,
            "Volume": m.get('volume', 0)
        })
        
    df = pd.DataFrame(rows)
    # Remove duplicates
    df = df.drop_duplicates(subset=['Ticker'])
    
    # Save
    out_dir = "data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path = os.path.join(out_dir, "kalshi_markets.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df)} markets to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
