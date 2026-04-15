import logging
import pandas as pd
from db_pool import get_connection
import reorder_math
from constants import PRIMARY_LOCATION

# Setup simple logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze():
    items = [
        'CHEH2SO4', 'CHELIGLIQ', 'EDTAMG', 'GRPASCORBIC', 'GRPFISHEM', 
        'GRPFOLIC', 'GRPFULVIC', 'GRPGABA', 'GRPTRAINER', 'MISCDEFOAM', 
        'NO3FE', 'NO3MG63', 'NO3ZN', 'NPK0030CDI', 'NPKAQUA', 
        'SO4FEDRY19', 'SO4FELIQ', 'SO4MN32'
    ]
    
    print(f"ANALYZING {len(items)} ITEMS FROM SCREENSHOT...\n")
    print(f"{'ITEM':<15} {'DESC':<25} {'URGENCY':<10} {'AVAILABLE':>10} {'ROP':>10} {'SUGGESTED':>10} {'ORDER BY':<12}")
    print("-" * 100)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Analyze using standard logic (lookback 90 days, safety 7 days)
        df = reorder_math.get_reorder_recommendations(
            cursor, 
            item_numbers=items, 
            lookback_days=90, 
            safety_days=7,
            location=PRIMARY_LOCATION,
            only_below_rop=False
        )
        
        if not df.empty:
            # Sort by Urgency (Critical -> Soon -> OK)
            # Custom sort order
            urgency_map = {'Critical': 0, 'Soon': 1, 'OK': 2}
            df['sort'] = df['urgency'].map(urgency_map)
            df = df.sort_values('sort')
            
            for _, row in df.iterrows():
                rec = row['suggested_order_qty']
                if rec == 0:
                    rec_str = "-"
                else:
                    rec_str = f"{rec:,.0f}"
                    
                print(f"{row['item_number']:<15} {row['item_description'][:25]:<25} {row['urgency']:<10} "
                      f"{row['qty_available']:>10,.0f} {row['calculated_rop']:>10,.0f} {rec_str:>10} {row['must_order_by']}")
                      
            print("-" * 100)
            
            # Summary for User
            critical = df[df['urgency'] == 'Critical']
            if not critical.empty:
                print("\n🚨 CRITICAL ITEMS (Order Immediately):")
                for _, row in critical.iterrows():
                    print(f"- {row['item_number']} ({row['item_description']}): Need {row['suggested_order_qty']:,.0f}")
            else:
                print("\n✅ No critical items found based on *Calculated Usage*.")

            soon = df[df['urgency'] == 'Soon']
            if not soon.empty:
                print("\n⚠️ SOON (Watch List):")
                for _, row in soon.iterrows():
                    print(f"- {row['item_number']}")

if __name__ == "__main__":
    analyze()
