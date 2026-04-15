"""
Market Cluster Scanner (Sector Alpha)
=====================================

Finds "Lagging" markets within a thematic sector.
1. Clusters markets by semantics (e.g. "Trump", "Crypto", "War").
2. Calculates Sector Momentum (Average % Change).
3. Flags markets that are left behind.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def load_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        return df
    except:
        return None

def main():
    print("🧩 MARKET CLUSTER SCANNER")
    print("=======================")
    
    df = load_data()
    if df is None: return
    
    # Get latest snapshot per market
    latest = df.sort_values('Timestamp').groupby('Question').tail(1)
    
    # Calculate 24h Change for each market
    # We need to find the price 24h ago
    now_ts = df['Timestamp'].max()
    day_ago_ts = now_ts - (24 * 3600)
    
    changes = []
    
    print(f"   Analyzing {len(latest)} markets...")
    
    for q in latest['Question'].unique():
        sub = df[df['Question'] == q]
        current_price = sub['Price'].iloc[-1]
        
        # Find price ~24h ago
        geo = sub[sub['Timestamp'] <= day_ago_ts]
        if len(geo) > 0:
            old_price = geo['Price'].iloc[-1]
            pct_change = (current_price - old_price)
        else:
            pct_change = 0.0 # New listing or no history
            
        changes.append({
            "Question": q,
            "Current": current_price,
            "Change24h": pct_change
        })
        
    df_changes = pd.DataFrame(changes)
    
    # CLUSTERING
    # Use TF-IDF to vectorise text
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    matrix = tfidf.fit_transform(df_changes['Question'])
    
    # Auto-detect K (roughly sqrt of N/2 or fixed 10-20)
    k = max(5, int(len(df_changes) / 10)) 
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_changes['Cluster'] = kmeans.fit_predict(matrix)
    
    print(f"   Identified {k} Thematic Clusters.")
    
    # Analyze Clusters
    print("\n📊 SECTOR ANALYSIS")
    print(f"{'Sector Keywords (Approx)':<40} {'Avg Move':<10} {'Signal'}")
    print("-" * 100)
    
    for c_id in range(k):
        subset = df_changes[df_changes['Cluster'] == c_id]
        if len(subset) < 3: continue # Skip tiny clusters
        
        avg_move = subset['Change24h'].mean()
        
        # Extract keywords to name the cluster
        # Heuristic: Most common words in this cluster
        text = " ".join(subset['Question']).lower()
        # Simple split (could use tfidf features but this is fast)
        # Just show the first question as example
        example = subset['Question'].iloc[0][:35] + "..."
        
        # Signal
        trend = "FLAT"
        if avg_move > 0.05: trend = "🚀 BOOMING"
        elif avg_move < -0.05: trend = "📉 CRASHING"
        
        print(f"Cluster {c_id}: {example:<30} {avg_move:<10.2f} {trend}")
        
        # Find Laggards
        # If Sector is BOOMING (>5%) but Stock is FLAT/DOWN
        if avg_move > 0.05:
            laggards = subset[subset['Change24h'] < 0.01]
            for _, row in laggards.iterrows():
                print(f"   🚨 LAGGARD ALERT: {row['Question'][:40]} (Change: {row['Change24h']:.2f})")
                
        # If Sector is CRASHING (<-5%) but Stock is FLAT/UP
        if avg_move < -0.05:
            laggards = subset[subset['Change24h'] > -0.01]
            for _, row in laggards.iterrows():
                print(f"   🚨 RESILIENT ALERT (Short?): {row['Question'][:40]} (Change: {row['Change24h']:.2f})")
                
    print("\n✅ Cluster Scan Complete.")

if __name__ == "__main__":
    main()
