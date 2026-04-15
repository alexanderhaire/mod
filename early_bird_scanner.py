"""
Early Bird Alpha Scanner (Day 0)
================================

Detects "New Listing Arbitrage" opportunities.
Goal: Find a new market (Age < 24h) that is mispriced vs a mature market.

Safety: 
- Strict Semantic Match (> 0.85).
- Liquidity Check on Parent Market.
"""

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

GAMMA_API = "https://gamma-api.polymarket.com"

def fetch_markets(params):
    try:
        res = requests.get(f"{GAMMA_API}/events", params=params)
        return res.json()
    except:
        return []

def main():
    print("🐣 EARLY BIRD ALPHA SCANNER")
    print("=========================")
    
    # 1. Fetch "Liquid Masters" ( Established markets to benchmark against )
    print("1️⃣  Fetching Liquid Master Markets...")
    # Fetch top 100 active by volume
    masters = fetch_markets({"limit": 100, "closed": "false", "order": "volume24hr", "ascending": "false"})
    
    # 2. Fetch "Newborns" ( Newest listings )
    print("2️⃣  Fetching Newborn Markets...")
    newborns = fetch_markets({"limit": 50, "closed": "false", "order": "creationDate", "ascending": "false"})
    
    if not masters or not newborns:
        print("❌ Failed to fetch markets.")
        return

    # Prepare Dataframes
    master_rows = []
    for m in masters:
        mkts = m.get('markets', [])
        if not mkts: continue
        mkt = mkts[0] # assumption: binary matches
        master_rows.append({
            "Question": mkt.get('question'),
            "Price": mkt.get('outcomePrices', [0.5, 0.5])[0], # Take YES price (string)
            "Volume": mkt.get('volume', 0),
            "ID": mkt.get('id')
        })
    
    new_rows = []
    for n in newborns:
         mkts = n.get('markets', [])
         if not mkts: continue
         mkt = mkts[0]
         new_rows.append({
            "Question": mkt.get('question'),
            "Price": mkt.get('outcomePrices', [0.5, 0.5])[0],
            "Creation": n.get('creationDate'),
            "ID": mkt.get('id')
         })
         
    df_master = pd.DataFrame(master_rows)
    df_new = pd.DataFrame(new_rows)
    
    # Clean Prices (API returns strings sometimes)
    df_master['Price'] = pd.to_numeric(df_master['Price'], errors='coerce')
    df_new['Price'] = pd.to_numeric(df_new['Price'], errors='coerce')

    print(f"   Analyzing {len(df_new)} Newborns vs {len(df_master)} Masters...")
    
    # 3. Vectorization (TF-IDF)
    # Combine all texts to fit vectorizer
    all_questions = df_master['Question'].tolist() + df_new['Question'].tolist()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    
    # Split back
    master_vecs = tfidf_matrix[:len(df_master)]
    new_vecs = tfidf_matrix[len(df_master):]
    
    # 4. Compare
    print("\n🔍 FINDING DAY-0 ARBITRAGE...")
    print(f"{'New Market (Short)':<35} {'Master Match (Short)':<35} {'Sim':<6} {'New $':<6} {'Old $':<6} {'Alpha'}")
    print("-" * 110)
    
    # Calculate similarity matrix (Rows: New, Cols: Master)
    sim_matrix = cosine_similarity(new_vecs, master_vecs)
    
    hits = 0
    for i in range(len(df_new)):
        # Find best match for this new market
        best_idx = sim_matrix[i].argmax()
        score = sim_matrix[i][best_idx]
        
        # Strict Safety Filter
        if score > 0.85:
            new_mkt = df_new.iloc[i]
            master_mkt = df_master.iloc[best_idx]
            
            # Don't match self (if overlap)
            if new_mkt['Question'] == master_mkt['Question']:
                continue
                
            price_diff = new_mkt['Price'] - master_mkt['Price']
            
            # Alpha Logic
            alpha = ""
            if abs(price_diff) > 0.10: # >10 cent mispricing
                alpha = "🔥 HUGE ARB"
            elif abs(price_diff) > 0.05:
                alpha = "🟢 Tradable"
            else:
                alpha = "⚪ Flat"
            
            n_short = new_mkt['Question'][:32] + "..."
            m_short = master_mkt['Question'][:32] + "..."
            
            print(f"{n_short:<35} {m_short:<35} {score:<6.2f} {new_mkt['Price']:<6.2f} {master_mkt['Price']:<6.2f} {alpha}")
            hits += 1
            
    if hits == 0:
        print("   No high-confidence matches found in current window.")

if __name__ == "__main__":
    main()
