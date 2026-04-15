import pandas as pd
from db_pool import get_connection
from reorder_math import get_reorder_recommendations

# Items from the user's screenshot
ITEMS_TO_CHECK = [
    'CHEH2SO4',
    'CHELIGLIQ',
    'EDTACU',
    'EDTAMG',
    'GRPASCORBIC',
    'GRPFISHEM',
    'GRPFOLIC',
    'GRPFULVIC',
    'GRPGABA',
    'GRPTRAINER',
    'MISCDEFOAM',
    'NO3FE',
    'NO3MG63',
    'NO3ZN',
    'NPK0030CDI',
    'SO4FEDRY19',
    'SO4FELIQ'
]

def analyze_items():
    print(f"Analyzing reorder requirements for {len(ITEMS_TO_CHECK)} items...")
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get recommendations based on usage (90 day lookback default)
            df = get_reorder_recommendations(
                cursor=cursor,
                item_numbers=ITEMS_TO_CHECK,
                lookback_days=90,
                safety_days=14, # 2 weeks safety stock
                location='MAIN'
            )
            
            if df.empty:
                print("No data found for these items.")
                return

            # Filter/Format for display
            print("\nAnalysis Results (Based on 90-day Usage & 14-day Safety Stock):")
            print("-" * 120)
            
            # Columns to display
            cols = [
                'item_number', 'qty_available', 'avg_daily_usage', 
                'days_of_coverage', 'lead_time_days', 'gp_order_point', 
                'calculated_rop', 'suggested_order_qty', 'must_order_by', 'urgency'
            ]
            
            # Format output
            for _, row in df.iterrows():
                item = row['item_number']
                avail = f"{row['qty_available']:,.1f}"
                usage = f"{row['avg_daily_usage']:,.2f}/day"
                coverage = f"{row['days_of_coverage']:.1f} days"
                rop = f"{row['calculated_rop']:,.1f}"
                gp_rop = f"{row['gp_order_point']:,.0f}"
                suggested = f"{row['suggested_order_qty']:,.0f}"
                date = row['must_order_by'].strftime('%Y-%m-%d')
                urgency = row['urgency']
                
                print(f"{item:<15} Avail: {avail:>10} | Usage: {usage:>10} | Coverage: {coverage:>10} | GP ROP: {gp_rop:>8} | Calc ROP: {rop:>8} | Order: {suggested:>8} | By: {date} | {urgency}")

            print("-" * 120)
            print("\nSummary Recommendation:")
            critical = df[df['urgency'] == 'Critical']
            soon = df[df['urgency'] == 'Soon']
            
            if not critical.empty:
                print(f"\nCRITICAL (Order Immediately):")
                for _, row in critical.iterrows():
                    print(f"- {row['item_number']} (Must order by {row['must_order_by']})")
                    
            if not soon.empty:
                print(f"\nSOON (Plan to order):")
                for _, row in soon.iterrows():
                    print(f"- {row['item_number']} (Must order by {row['must_order_by']})")

    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_items()
