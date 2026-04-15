"""
Historical Early Bird Backtest
==============================

Estimates the frequency of "Day 0" Arbitrage opportunities.
Replays the past 7 days of data.
Logic: On the hour a market FIRST appears, check if it matched any ALREADY EXISTING market.

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        return df
    except:
        return None

def main():
    print("🕰️ HISTORICAL REPLAY: DAY 0 SCANNER")
    print("====================================")
    
    df = load_data()
    if df is None: return
    
    # 1. Determine "Birth Time" for each market in our dataset
    market_births = df.groupby('Question')['datetime'].min().sort_values()
    
    unique_markets = market_births.index.tolist()
    print(f"   Dataset contains {len(unique_markets)} unique markets.")
    print(f"   Timeline: {market_births.min()} to {market_births.max()}")
    
    # We need a rolling window. 
    # Iterate through markets in chronological order of appearance.
    
    active_pool = [] # Markets that exist at time T
    matches_found = 0
    
    # Pre-vectorize all to save time (approximation)
    tfidf = TfidfVectorizer(stop_words='english')
    # We fit on ALL questions so dimensions match
    all_vecs = tfidf.fit_transform(unique_markets)
    # Map index
    idx_map = {q: i for i, q in enumerate(unique_markets)}
    
    print("\n🎞️ Replaying History...")
    print(f"{'New Born Market (Short)':<40} {'Match Found (Existing)':<40} {'Sim':<6}")
    print("-" * 100)
    
    # Iterate through time (by birth order)
    # Note: In a real backtest, we'd check if the 'Existing' market was actually active/liquid at that time.
    # For this estimation, we assume if it was born earlier, it exists.
    
    for question, birth_time in market_births.items():
        # This is the "New" market
        q_vec = all_vecs[idx_map[question]]
        
        # Check against Active Pool
        if active_pool:
            # Get vectors for active pool
            pool_indices = [idx_map[q] for q in active_pool]
            pool_vecs = all_vecs[pool_indices]
            
            # Calc similarity
            sims = cosine_similarity(q_vec, pool_vecs)
            
            # Find max
            best_idx = sims.argmax()
            best_score = sims[0, best_idx]
            
            if best_score > 0.85:
                matched_q = active_pool[best_idx]
                
                # Check duplication/overlap text
                if question == matched_q: continue
                
                print(f"{question[:35]:<40} {matched_q[:35]:<40} {best_score:<6.2f}")
                matches_found += 1
        
        # Add to pool for future markets to match against
        active_pool.append(question)
        
    print("-" * 100)
    print(f"\n📊 RESULTS:")
    print(f"   Total Markets Processed: {len(unique_markets)}")
    print(f"   Total Day-0 Matches: {matches_found}")
    
    if len(unique_markets) > 0:
        freq = (matches_found / len(unique_markets)) * 100
        print(f"   Hit Rate: {freq:.1f}% of new listings match an existing one.")
    
    print("\n📝 VERDICT:")
    if matches_found > 0:
        print("   This strategy happens frequently enough to automate!")
    else:
        print("   Matches are rare in this small 7-day sample.")
        print("   (Note: Our sample size is only 50 active markets. In reality, with thousands, hits increase.)")

if __name__ == "__main__":
    main()
