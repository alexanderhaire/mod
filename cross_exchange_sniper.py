"""
Cross-Exchange Sniper (Poly vs Manifold)
========================================

Compares Real Money (Polymarket) prices against Play Money (Manifold) sentiment.
Goal: Find "Crowd Signal" Alpha. 
If Manifold (Crowd) is significantly different from Polymarket (Money), 
it might indicate a lagging price or a mispriced risk premium.

"""

import pandas as pd
from difflib import SequenceMatcher

def fuzzy_match(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def load_data():
    try:
        poly = pd.read_csv('data/polymarket_real.csv')
        mani = pd.read_csv('data/manifold_markets.csv')
        
        # Get latest Poly snapshot
        if 'Timestamp' in poly.columns:
            poly = poly.sort_values('Timestamp').groupby('Question').tail(1)
            
        return poly, mani
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def main():
    print("⚔️ CROSS-EXCHANGE SIGNAL SCANNER")
    print("================================")
    
    poly_df, mani_df = load_data()
    if poly_df is None: return
    
    print(f"   Loaded {len(poly_df)} Polymarket events.")
    print(f"   Loaded {len(mani_df)} Manifold markets.")
    
    print("\n🔍 Matching Events (Deep Scan)...")
    print(f"{'Event (Short)':<40} {'Poly %':<8} {'Mani %':<8} {'Diff':<8} {'Signal'}")
    print("-" * 100)
    
    matches_found = 0
    
    for _, prow in poly_df.iterrows():
        p_q = prow['Question']
        p_price = float(prow['Price'])
        
        # Find best match in Manifold
        best_match = None
        best_score = 0
        
        # Heuristic: Check volume first to save time? No, need all.
        # Check against Manifold Question
        for _, mrow in mani_df.iterrows():
            m_q = mrow['Question']
            score = fuzzy_match(p_q, m_q)
            
            if score > best_score:
                best_score = score
                best_match = mrow
                
        if best_score > 0.6: # Decent text match
            m_prob = best_match['Probability']
            diff = m_prob - p_price
            
            signal = "⚪ Neutral"
            if abs(diff) > 0.15: # 15% Divergence
                if diff > 0:
                     signal = "🚀 BULLISH Signal (Poly Undervalued)"
                else:
                     signal = "🐻 BEARISH Signal (Poly Overvalued)"
            elif abs(diff) > 0.05:
                signal = "🟡 Weak Signal"
                
            label = f"{p_q[:35]}..."
            print(f"{label:<40} {p_price:<8.2f} {m_prob:<8.2f} {diff:<8.2f} {signal}")
            matches_found += 1
            
    if matches_found == 0:
        print("   No overlapping events found.")
    else:
        print(f"\n✅ Scanned {len(poly_df)} Poly events vs {len(mani_df)} Manifold events.")
        print("   Strategy: If Manifold (Crowd) leads, trade Polymarket in direction of the Signal.")

if __name__ == "__main__":
    main()
