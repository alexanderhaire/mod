"""
Polymarket Data Fetcher
=======================

Fetches real prediction market data using the Gamma and CLOB APIs.
1. Gets top active events.
2. Gets historical prices for the 'Yes' outcome.
3. Saves to CSV for ML training.

"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def fetch_price_history_debug(token_id, start, end, extra_params):
    url = f"{CLOB_API}/prices-history"
    params = {"market": token_id, "startTs": int(start), "endTs": int(end)}
    params.update(extra_params)
    
    try:
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"        Error {res.status_code}: {res.text[:100]}")
            return []
        data = res.json()
        if 'history' in data and data['history']:
            return data['history']
        return []
    except:
        return []

# Constants
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

def fetch_top_events(limit=10):
    """Fetches top active events from Gamma API."""
    print("📡 Fetching top events...")
    try:
        # We want closed/resolved events for training if possible, but let's start with active for now
        # or mix. Ideally we need resolved events to know the target (Win/Loss).
        # For this demo, let's fetch 'Top' events and assume we can backtest on their history.
        
        # params = {"limit": limit, "active": "true", "closed": "false"} 
        # Actually, let's try to get some that are closed if we want to train a winner model?
        # But closed events might not be easily queryable by 'top'.
        # Let's stick to active and we will use 'current price' as a proxy or just simulate 'next day' prediction.
        
        url = f"{GAMMA_API}/events"
        params = {"limit": limit, "closed": "false"} # Active only
        res = requests.get(url, params=params)
        try:
            res.raise_for_status()
        except Exception:
            print(f"   ⚠️ API Error: {res.text[:200]}")
            raise
            
        events = res.json()
        print(f"   Found {len(events)} events.")
        return events
    except Exception as e:
        print(f"❌ Error fetching events: {e}")
        return []

def fetch_price_history(token_id, start_ts, end_ts):
    """Fetches OHLC price history for a token."""
    url = f"{CLOB_API}/prices-history"
    params = {
        "market": token_id,
        "startTs": int(start_ts), 
        "endTs": int(end_ts),
        "interval": "1h" # granularity
    }
    
    print(f"      Requesting history for: {token_id}")
    try:
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"      ⚠️ CLOB Error {res.status_code}: {res.text[:100]}")
            return []
            
        if not res.content:
            print("      ⚠️ Empty response content.")
            return []
            
        data = res.json()
        if 'history' in data and data['history']:
            return data['history']
            
        print(f"      ⚠️ Valid response but no history. Keys: {list(data.keys())} Msg: {data}")
        return []
    except Exception as e:
        print(f"      ⚠️ JSON/Fetch Error: {e} Raw: {res.text[:100]}")
        return []

def main():
    print("🚀 POLYMARKET REAL DATA FETCH")
    print("="*60)
    
    events = fetch_top_events(limit=500)
    all_data = []
    
    current_time = time.time()
    
    for evt in events:
        title = evt.get('title', 'Unknown')
        slug = evt.get('slug', '')
        markets = evt.get('markets', [])
        
        # Le'ts look for a market with 2 outcomes (Binary)
        target_mkt = None
        for m in markets:
            target_mkt = m
            break
            
        if not target_mkt: continue
        
        question = target_mkt.get('question', title)
        print(f"   Processing: {question[:40]}...")
        
        clob_ids = target_mkt.get('clobTokenIds', [])
        outcomes = target_mkt.get('outcomes', [])
        
        if isinstance(clob_ids, str):
            import json
            try: clob_ids = json.loads(clob_ids)
            except: pass
                
        if isinstance(outcomes, str):
            import json
            try: outcomes = json.loads(outcomes)
            except: pass

        if len(clob_ids) < 2: 
            # print("      Skipping (not enough tokens)")
            continue
            
        yes_token = clob_ids[0]
        outcome_label = outcomes[0] if outcomes else "Outcome_1"
        
        # logic for window
        fetch_end = current_time
        fetch_start = fetch_end - (7 * 24 * 60 * 60) # Last 7 days
        
        # Use dynamic dates
        end_ts = fetch_end
        start_ts = fetch_start
        
        print(f"      Fetch Window: {datetime.fromtimestamp(start_ts).date()} to {datetime.fromtimestamp(end_ts).date()}")

        # Try multiple combos to find working one
        combos = [
            {"label": "Sec/7d/1h", "start": start_ts, "end": end_ts, "params": {"interval": "1h"}},
            {"label": "Sec/7d/1m", "start": end_ts - 7*24*3600, "end": end_ts, "params": {"interval": "1m", "fidelity": 10}}, 
        ]

        history = []
        for c in combos:
            print(f"      Testing {c['label']}...")
            h = fetch_price_history_debug(yes_token, c['start'], c['end'], c['params'])
            if h:
                print(f"      ✅ Success with {c['label']}! Got {len(h)} candles.")
                history = h
                break
        
        if not history:
            print("      No history found (all combos failed).")
            continue
        
        # Parse history
        if history:
             print(f"      DEBUG Sample Candle: {history[0]}")
        
        for candle in history:
            # Candles might be simple dicts
            ts = candle.get('t')
            if not ts: continue
            
            row = {
                'Event': title,
                'Question': question,
                'Outcome': outcome_label,
                'Date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
                'Timestamp': ts,
                'Price': candle.get('p'), # 'p' for Price (checked via debug)
                'Volume': candle.get('v', 0)
            }
            all_data.append(row)
            
        time.sleep(0.5) # Rate limit respect
        
    if not all_data:
        print("⚠️ No data collected.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Save
    out_dir = "data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path = os.path.join(out_dir, "polymarket_real.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Data saved to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
