"""
Deep Learning / NLP Alpha Scanner
=================================

Identifies "Alpha" by finding correlations between semantically similar events.
Hypothesis: Events with similar titles (e.g. "Trump in PA", "Trump in MI") should move together.
If one lags (high semantic sim, low immediate price correlation), it's a trade opportunity.

Steps:
1. Load Real Data.
2. Vectorize Titles (TF-IDF).
3. Find High-Similarity Pairs.
4. Analyze Price Correlation (Pearson) and Lead-Lag.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

def load_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        # Ensure numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def main():
    print("🧠 DEEP LEARNING ALPHA SCANNER")
    print("==============================")
    
    df = load_data()
    if df is None: return
    
    # 1. Prepare Text Data
    # Combine Event and Question for richer context
    df['text_feature'] = (df['Event'] + " " + df['Question']).fillna('')
    
    # Get unique events
    events = df.groupby('Question').agg({
        'text_feature': 'first',
        'Event': 'first'
    }).reset_index()
    
    print(f"   Loaded {len(df)} candles for {len(events)} unique events.")
    
    if len(events) < 2:
        print("⚠️ Not enough events to find correlations.")
        return

    # 2. Vectorize (TF-IDF)
    print("\n🧮 Vectorizing Titles (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(events['text_feature'])
    
    # 3. Compute Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # 4. Find Top Correlated Pairs
    pairs = []
    
    # Iterate upper triangle
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            sim_score = cosine_sim[i, j]
            
            if sim_score > 0.4: # Filter for decent semantic similarity
                ev_a = events.iloc[i]
                ev_b = events.iloc[j]
                
                pairs.append({
                    'Event_A': ev_a['Question'],
                    'Event_B': ev_b['Question'],
                    'Semantic_Sim': sim_score,
                    'Idx_A': i,
                    'Idx_B': j
                })
    
    pairs.sort(key=lambda x: x['Semantic_Sim'], reverse=True)
    print(f"   Found {len(pairs)} semantically similar pairs.")
    
    # 5. Analyze Price Correlation for Top Pairs
    print("\n🔬 Analyzing Liquidity & Correlations...")
    print(f"{'Event A vs Event B':<60} {'Sem Sim':<8} {'Corr':<6} {'Vol A':<6} {'Vol B':<6} {'Status'}")
    print("-" * 110)
    
    # Resample all data to hourly for consistent alignment
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Pivot to wide format: Index=Time, Columns=Question, Values=Price
    price_matrix = df.pivot_table(index='datetime', columns='Question', values='Price')
    price_matrix = price_matrix.resample('1h').last().ffill()
    
    # Volume Check (Approx daily volume)
    # We don't have perfect daily volume history, but we have 'Volume' snapshot in CSV?
    # Actually fetcher saves 'Volume' per row. Let's take the max volume seen.
    vol_map = df.groupby('Question')['Volume'].max().to_dict()
    
    count = 0
    for p in pairs[:50]: # Check top 50
        q_a = p['Event_A']
        q_b = p['Event_B']
        
        vol_a = vol_map.get(q_a, 0)
        vol_b = vol_map.get(q_b, 0)
        
        # Liquidity Filter: At least one must be liquid ($10k+) for Lead-Lag
        # Or both for Pair Trade.
        # Let's say we need at least $1000 to even bother checking.
        if vol_a < 1000 and vol_b < 1000:
            continue
            
        if q_a not in price_matrix.columns or q_b not in price_matrix.columns:
            continue
            
        series_a = price_matrix[q_a]
        series_b = price_matrix[q_b]
        
        returns_a = series_a.pct_change().dropna()
        returns_b = series_b.pct_change().dropna()
        
        common_idx = returns_a.index.intersection(returns_b.index)
        if len(common_idx) < 5:
            price_corr = 0
        else:
            price_corr = returns_a.loc[common_idx].corr(returns_b.loc[common_idx])
            
        status = "⚪ Ignore"
        if price_corr > 0.7:
            status = "🟢 Pair Trade"
        elif price_corr > 0.3:
            status = "🟡 Weak Link"
        elif price_corr < 0.1:
            # If correlation is low but semantics are high...
            # And one is Liquid vs Illiquid -> Lead-Lag Opportunity
            if vol_a > 10000 and vol_b < 5000:
                status = "🔥 A Leads B"
            elif vol_b > 10000 and vol_a < 5000:
                status = "🔥 B Leads A"
            else:
                status = "🔴 Disconnected"

        label = f"{q_a[:28]}... vs {q_b[:28]}..."
        print(f"{label:<60} {p['Semantic_Sim']:<8.2f} {price_corr:<6.2f} {vol_a:<6.0f} {vol_b:<6.0f} {status}")
        count += 1

        
    if count == 0:
        print("   No overlapping price history found for pairs.")

if __name__ == "__main__":
    main()
