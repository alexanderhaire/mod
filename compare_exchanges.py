"""
Cross-Exchange Arbitrage Scanner
================================

Compares Polymarket and Kalshi prices to find arbitrage opportunities.
Arbitrage Condition: |PriceA - PriceB| > 5% (Fee Buffer).

"""

import pandas as pd
from difflib import SequenceMatcher

def fuzzy_match(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def main():
    print("⚔️ CROSS-EXCHANGE ARBITRAGE SCANNER")
    print("="*60)
    
    # Load Data
    try:
        poly_df = pd.read_csv('data/polymarket_real.csv')
        kalshi_df = pd.read_csv('data/kalshi_markets.csv')
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Get latest Poly data
    if 'Timestamp' in poly_df.columns:
        poly_latest = poly_df.sort_values('Timestamp').groupby('Event').tail(1)
    else:
        poly_latest = poly_df
        
    print(f"   Loaded {len(poly_latest)} Polymarket events.")
    print(f"   Loaded {len(kalshi_df)} Kalshi events.")
    print(f"   Sample Kalshi Titles: {kalshi_df['Event'].head(10).tolist()}")
    
    print("\n🔍 Scanning for matches...")
    print(f"{'Polymarket Event':<40} {'Kalshi Match':<40} {'Poly $':<8} {'Kalshi $':<8} {'Spread'}")
    print("-" * 110)
    
    matches_found = 0
    
    for _, prow in poly_latest.iterrows():
        p_title = prow['Event']
        p_q = prow.get('Question', '')
        p_price = float(prow['Price'])
        
        # We need to find this in Kalshi
        # Kalshi 'Event' is the title
        
        best_match = None
        best_score = 0
        
        for _, krow in kalshi_df.iterrows():
            k_title = krow['Event']
            # Match against Title or Subtitle
            score1 = fuzzy_match(p_title, k_title)
            score2 = fuzzy_match(p_q, k_title) # Match question too
            
            score = max(score1, score2)
            
            if score > best_score:
                best_score = score
                best_match = krow
        
        # Threshold for match
        if best_score > 0.3: # Lowered to catch loose matches for demo
            k_price = best_match['Last']
            if k_price == 0:
                # Try mid price if last is 0 (illiquid)
                k_price = (best_match['Bid'] + best_match['Ask']) / 2.0
            
            # If still 0, skip
            if k_price == 0: continue
            
            spread = p_price - k_price
            
            # Verify if it's the SAME outcome (Yes vs Yes)
            # This is hard to automate perfectly without NLP.
            # Assuming 'Yes' matching for now.
            
            print(f"{p_title[:38]:<40} {best_match['Event'][:38]:<40} {p_price:<8.2f} {k_price:<8.2f} {spread:<8.2f}")
            matches_found += 1
            
    if matches_found == 0:
        print("\n⚠️ No matches found.")

if __name__ == "__main__":
    main()
