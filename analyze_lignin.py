import pyodbc
from db_pool import get_connection
from procurement_ml import ProcurementFeatureBuilder
import pandas as pd
import datetime

def analyze_lignin_usage():
    print("--- Analyzing Lignin Usage ---")
    
    items = ['CHELIGDRY', 'CHELIGLIQ']
    
    with get_connection() as conn:
        cursor = conn.cursor()
        builder = ProcurementFeatureBuilder(cursor)
        
        for item in items:
            print(f"\nAnalyzing Item: {item}")
            
            # 1. Get Inventory
            inv = builder._get_inventory_status(item)
            if not inv:
                print(f"  No inventory record found.")
                continue
                
            # Get On Hand
            on_hand = inv.get('on_hand', 0)
            print(f"  On Hand: {on_hand:.2f}")

            # Get Allocation (Try ATYALLOC first, then ignore if fails)
            allocated = 0.0
            try:
                # Try ATYALLOC (Allocated to Mfg commonly) or QTYALLOC
                # Just catch the error if column doesn't exist
                cursor.execute("SELECT SUM(ATYALLOC) FROM IV00102 WHERE ITEMNMBR = ?", item)
                alloc_row = cursor.fetchone()
                if alloc_row and alloc_row[0]:
                    allocated = float(alloc_row[0])
            except Exception:
                pass
                
            print(f"  Allocated (Mfg): {allocated:.2f}")
            
            free_stock = on_hand - allocated
            print(f"  Free Stock: {free_stock:.2f}")

            # 2. Get Usage (Last 12 Months)
            today = datetime.date.today()
            usage_df = builder._get_usage_history(item, today)
            
            if usage_df.empty:
                print("  No usage history found.")
                continue
                
            # Filter to last 12 months
            usage_df = usage_df[usage_df['TransactionDate'] >= pd.Timestamp(today - datetime.timedelta(days=365))]
            
            # Monthly Breakdown
            print("\n  Monthly Usage:")
            if not usage_df.empty:
                monthly = usage_df.groupby(usage_df['TransactionDate'].dt.to_period('M'))['Quantity'].sum()
                for period, qty in monthly.items():
                    print(f"    {period}: {qty:,.2f}")
            
            total_usage = usage_df['Quantity'].sum()
            avg_daily_usage = total_usage / 365
            print(f"\n  Total Usage (12mo): {total_usage:,.2f}")
            print(f"  Avg Daily Usage:    {avg_daily_usage:,.2f}")
            
            if avg_daily_usage > 0:
                days_supply = free_stock / avg_daily_usage
                print(f"  Days of Supply (Free): {days_supply:.1f} days")
                
                # Recommendation
                suggest_1_mo = avg_daily_usage * 30
                suggest_3_mo = avg_daily_usage * 90
                
                print(f"  Recommended Trial (1 Month): {suggest_1_mo:.2f}")
                print(f"  Recommended Full (3 Months): {suggest_3_mo:.2f}")
            else:
                 print("  Item is inactive (0 usage).")

if __name__ == "__main__":
    analyze_lignin_usage()
