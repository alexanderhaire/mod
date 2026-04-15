"""
Prediction Market Alpha Report
==============================

Combines Real Market Data (Polymarket) with "Internet Signals" (Agent Research).
Identifies discrepancies between Market Price and Information Consensus.

"""

import pandas as pd
import json
import os

def load_data():
    # Load Real Market Data
    try:
        market_df = pd.read_csv('data/polymarket_real.csv')
    except Exception as e:
        print(f"❌ Could not load market data: {e}")
        return None
        
    # Load Internet Signals
    try:
        with open('data/news_signals.json', 'r') as f:
            signals = json.load(f)
    except Exception as e:
        print(f"❌ Could not load signals: {e}")
        return None
        
    return market_df, signals

def generate_report():
    print("🕵️‍♂️ GENERATING ALPHA REPORT...")
    print("="*60)
    
    market_df, signals = load_data()
    if market_df is None: return

    # Get latest price for each event
    # Group by Question and take the last row (latest timestamp)
    latest_prices = market_df.sort_values('Timestamp').groupby('Question').tail(1)
    
    print(f"{'Event / Question':<50} {'Mkt Price':<10} {'My Model':<10} {'Edge':<10} {'Action'}")
    print("-" * 95)
    
    for _, row in latest_prices.iterrows():
        question = row['Question']
        mkt_price = float(row['Price'])
        outcome = row['Outcome'] # Usually "Yes" or specific, need to handle
        
        # Find matching signal
        matched_sig = None
        for sig in signals:
            if sig['event_substring'].lower() in question.lower():
                matched_sig = sig
                break
        
        if not matched_sig:
            # No alpha for this one
            continue
            
        # Analyze Edge
        # Signal says "NO" with 0.85 confidence.
        # This implies "YES" probability is (1 - 0.85) = 0.15 (Roughly)
        
        my_prob = 0.5
        consensus = matched_sig['internet_consensus']
        confidence = matched_sig['confidence']
        
        if consensus == "YES":
            my_prob = 0.5 + (0.5 * confidence) # Scale 0.5 to 1.0
        elif consensus == "NO":
            my_prob = 0.5 - (0.5 * confidence) # Scale 0.5 to 0.0
            
        # Invert if the token is "NO"? 
        # CSV usually tracks "Yes" price? Let's assume 'Price' is for the Outcome listed?
        # Fetcher was getting 'clob_ids[0]' which is usually YES.
        # So Mkt Price = Price of YES.
        
        edge = my_prob - mkt_price
        
        action = "-"
        if edge > 0.15: action = "🟢 BUY YES"
        elif edge < -0.15: action = "🔴 SELL YES" # or Buy No
        else: action = "⚪ NEUTRAL"
        
        print(f"{question[:48]:<50} {mkt_price:<10.2f} {my_prob:<10.2f} {edge:<10.2f} {action}")
        print(f"   Reasoning: {matched_sig['reasoning']}")
        print(f"   Sources: {', '.join(matched_sig['sources'])}")
        print("-" * 95)

if __name__ == "__main__":
    generate_report()
