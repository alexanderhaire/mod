"""
Order Book Imbalance (OBI) Pressure Scanner
===========================================

Scans the "Microstructure" of the market to find coiled springs.
Logic: High Bid Volume + Low Ask Volume = Upward Pressure.

Metric:
OBI = (BidVolume - AskVolume) / (BidVolume + AskVolume)
Range: -1 (Bearish) to +1 (Bullish).

"""

import requests
import pandas as pd
import time

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

def fetch_top_tokens(limit=20):
    """Fetch active tokens to scan order books for"""
    print("📡 Fetching active tokens...")
    try:
        res = requests.get(f"{GAMMA_API}/events", params={"limit": limit, "closed": "false", "active": "true"})
        events = res.json()
        
        tokens = []
        print(f"   DEBUG: Received {len(events)} events from Gamma.")
        if len(events) > 0:
            print(f"   DEBUG Sample Event Keys: {events[0].keys()}")
            
        for e in events:
            # Find binary market
            mkts = e.get('markets', [])
            if not mkts: 
                # Sometimes markets are not nested? Check top level?
                # Actually Gamma API structure is usually event -> markets
                continue
            
            m = mkts[0] # Take first
            clob_id = m.get('clobTokenIds', [])
            
            if not clob_id:
                # Sometimes clobTokenIds is a string in the new API?
                # Or sometimes we need to hit a different endpoint?
                continue
                
            # If clob_id is string, parse it
            if isinstance(clob_id, str):
                import json
                try: clob_id = json.loads(clob_id)
                except: pass
                
            if isinstance(clob_id, list) and len(clob_id) >= 2:
                # [Yes_Token, No_Token]
                # We analyze the YES token orderbook
                tokens.append({
                    "Question": m.get('question'),
                    "TokenID": clob_id[0], 
                    "Slug": e.get('slug')
                })
        return tokens
    except Exception as e:
        print(f"❌ Error fetching tokens: {e}")
        return []

def fetch_orderbook(token_id):
    """Get L2 Orderbook"""
    url = f"{CLOB_API}/book"
    try:
        res = requests.get(url, params={"token_id": token_id})
        if res.status_code != 200: return None
        return res.json() # {"bids": [], "asks": []}
    except:
        return None

def main():
    print("⚖️ ORDER BOOK PRESSURE GAUGE")
    print("==========================")
    
    tokens = fetch_top_tokens(limit=30)
    print(f"   Scanning {len(tokens)} Order Books...")
    
    results = []
    
    print(f"\n{'Market (Short)':<40} {'Bid Vol':<8} {'Ask Vol':<8} {'OBI Score':<8} {'Signal'}")
    print("-" * 100)
    
    for t in tokens:
        book = fetch_orderbook(t['TokenID'])
        if not book: continue
        
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        # Calculate Total Volume (Liquidity Depth)
        # Bids structure: [{"price": "0.60", "size": "100"}]
        
        bid_vol = sum([float(x['size']) for x in bids])
        ask_vol = sum([float(x['size']) for x in asks])
        
        total_vol = bid_vol + ask_vol
        if total_vol < 100: # Filter dust
            continue
            
        # OBI Formula
        obi = (bid_vol - ask_vol) / total_vol
        
        # Interpret
        signal = "⚪ Balanced"
        if obi > 0.5:
            signal = "🚀 PRESSURE UP (Bulls)"
        elif obi < -0.5:
            signal = "📉 PRESSURE DOWN (Bears)"
        elif obi > 0.2:
             signal = "🟢 Lean Bull"
        elif obi < -0.2:
             signal = "🔴 Lean Bear"
             
        # Shorten name
        name = t['Question'][:35] + "..."
        
        print(f"{name:<40} {bid_vol:<8.0f} {ask_vol:<8.0f} {obi:<8.2f} {signal}")
        
        results.append({
            "Question": t['Question'],
            "OBI": obi,
            "TotalLiquidity": total_vol
        })
        
        time.sleep(0.2) # Rate limit
        
    if results:
        # Find the single most pressurized market
        df = pd.DataFrame(results)
        best = df.loc[df['OBI'].idxmax()]
        worst = df.loc[df['OBI'].idxmin()]
        
        print("\n🏆 KEY FINDINGS:")
        print(f"   🔥 Most Bullish Pressure: {best['Question']} (OBI: {best['OBI']:.2f})")
        print(f"   ❄️ Most Bearish Pressure: {worst['Question']} (OBI: {worst['OBI']:.2f})")

if __name__ == "__main__":
    main()
