"""
Seasonal Needs Analysis Script
------------------------------
Analyzes historical usage for the upcoming season (Feb - Apr) across previous years
to determine purchasing needs, rather than relying on recent short-term history.
"""

import datetime
import logging
import pandas as pd
from tabulate import tabulate

from db_pool import get_connection
from constants import RAW_MATERIAL_CLASS_CODES, PRIMARY_LOCATION

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOGGER = logging.getLogger(__name__)

def get_seasonal_usage(cursor, item_number, start_month=2, end_month=4, years=[2023, 2024, 2025]):
    """
    Calculate average seasonal usage for a specific item over specified years.
    """
    total_usage = 0.0
    years_with_data = 0
    
    for year in years:
        # Construct date range for that year
        start_date = datetime.date(year, start_month, 1)
        
        # Handle end date logic (accounting for month length)
        if end_month == 12:
            end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(year, end_month + 1, 1) - datetime.timedelta(days=1)
            
        query = """
        SELECT SUM(ABS(TRXQTY)) as Usage
        FROM IV30300
        WHERE ITEMNMBR = ?
          AND TRXLOCTN = ?
          AND TRXQTY < 0
          AND DOCDATE BETWEEN ? AND ?
        """
        
        try:
            cursor.execute(query, (item_number, PRIMARY_LOCATION, start_date, end_date))
            row = cursor.fetchone()
            if row and row.Usage:
                usage = float(row.Usage)
                total_usage += usage
                years_with_data += 1
                # LOGGER.info(f"  {year} Usage: {usage:,.1f}")
        except Exception as e:
            LOGGER.warning(f"Error fetching usage for {item_number} in {year}: {e}")
            
    if years_with_data > 0:
        return total_usage / years_with_data
    return 0.0

def analyze_seasonal_needs():
    print("🚀 Starting Seasonal Needs Analysis (Feb-Apr)...")
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Get all Active Raw Materials
        print("📋 Fetching active raw materials...")
        placeholders = ", ".join("?" for _ in RAW_MATERIAL_CLASS_CODES)
        query = f"""
        SELECT 
            i.ITEMNMBR, 
            i.ITEMDESC, 
            i.ITMCLSCD,
            COALESCE(loc.QTYONHND, 0) as QtyOnHand,
            COALESCE(loc.QTYONORD, 0) as QtyOnOrder,
            COALESCE(pv.PLANNINGLEADTIME, 14) as LeadTime
        FROM IV00101 i
        LEFT JOIN IV00102 loc ON i.ITEMNMBR = loc.ITEMNMBR AND loc.LOCNCODE = ?
        CROSS APPLY (
            SELECT TOP 1 iv.PLANNINGLEADTIME
            FROM IV00103 iv
            WHERE iv.ITEMNMBR = i.ITEMNMBR
            ORDER BY iv.LSTORDDT DESC
        ) pv
        WHERE i.ITMCLSCD IN ({placeholders})
          AND i.INACTIVE = 0
        """
        
        cursor.execute(query, [PRIMARY_LOCATION, *RAW_MATERIAL_CLASS_CODES])
        items = cursor.fetchall()
        
        results = []
        
        current_year = datetime.date.today().year
        # Look back at previous 3 years
        # If we are in 2026, we look at 2023, 2024, 2025
        years_to_analyze = [current_year - 3, current_year - 2, current_year - 1]
        
        print(f"🔍 Analyzing usage for Feb-Apr in years: {years_to_analyze}")
        print(f"Total items to scan: {len(items)}")
        
        for item in items:
            item_number = item.ITEMNMBR.strip()
            qty_avail = float(item.QtyOnHand) + float(item.QtyOnOrder)
            
            # Calculate Average Seasonal Demand
            avg_seasonal_usage = get_seasonal_usage(cursor, item_number, start_month=2, end_month=4, years=years_to_analyze)
            
            # Identify Deficit
            # We need enough to cover the season + Safety Stock (let's say 2 weeks of avg seasonal usage)
            # Season is ~90 days. Avg daily = avg_seasonal_usage / 90
            if avg_seasonal_usage > 0:
                daily_usage = avg_seasonal_usage / 90.0
                safety_stock = daily_usage * 14
                
                target_qty = avg_seasonal_usage + safety_stock
                deficit = target_qty - qty_avail
                
                if deficit > 0:
                    results.append({
                        "Item": item_number,
                        "Description": item.ITEMDESC.strip(),
                        "Avg Seasonal Usage": avg_seasonal_usage,
                        "Current Avail": qty_avail,
                        "Target Qty": target_qty,
                        "Projected Deficit": deficit,
                        "Status": "CRITICAL" if qty_avail < (avg_seasonal_usage * 0.5) else "Warning"
                    })
        
        # Sort by Deficit Value? Or just Deficit Amount? 
        # Let's sort by Deficit Amount descending
        results.sort(key=lambda x: x["Projected Deficit"], reverse=True)
        
        # Create DataFrame for nice display
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Format columns
            df["Avg Seasonal Usage"] = df["Avg Seasonal Usage"].map('{:,.1f}'.format)
            df["Current Avail"] = df["Current Avail"].map('{:,.1f}'.format)
            df["Target Qty"] = df["Target Qty"].map('{:,.1f}'.format)
            df["Projected Deficit"] = df["Projected Deficit"].map('{:,.1f}'.format)
            
            print("\n📊 SEASONAL PURCHASING RECOMMENDATIONS (FEB-APR)")
            print("==================================================")
            print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
            
            # Save to CSV
            csv_filename = "seasonal_purchasing_plan.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\n💾 Saved detailed report to {csv_filename}")
        else:
            print("\n✅ No seasonal deficits found! You are well stocked.")

if __name__ == "__main__":
    analyze_seasonal_needs()
