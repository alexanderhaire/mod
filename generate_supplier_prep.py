"""
Supplier Meeting Prep Generator
-------------------------------
Generates a comprehensive Excel report for supplier meetings (e.g. Brenntag).
Combines usage history, spending analysis, seasonal forecasts, and supplier performance.
"""

import datetime
import logging
import pandas as pd
import pyodbc 
from decimal import Decimal

from db_pool import get_connection
from constants import RAW_MATERIAL_CLASS_CODES, PRIMARY_LOCATION
import market_insights
import analyze_seasonal_needs
import reorder_math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOGGER = logging.getLogger(__name__)

def get_item_on_time_performance(cursor, item_number, lookback_years=1):
    """
    Calculate on-time delivery percentage for a specific item over the last year.
    Returns: (on_time_pct, total_orders)
    """
    try:
        # Try joining via POP10500 (Receipt Line Quantities) which bridges Receipt Line to PO Line
        # This gives us access to the Promised Date on the PO Line (POP30110)
        query = """
        SELECT 
            COUNT(r.POPRCTNM) as TotalLines,
            SUM(CASE WHEN h.RECEIPTDATE <= pl.PRMSHPDTE THEN 1.0 ELSE 0.0 END) as OnTimeLines
        FROM POP30310 r
        JOIN POP30300 h ON r.POPRCTNM = h.POPRCTNM
        JOIN POP10500 m ON r.POPRCTNM = m.POPRCTNM AND r.RCPTLNNM = m.RCPTLNNM
        JOIN POP30110 pl ON m.PONUMBER = pl.PONUMBER AND m.ORD = pl.ORD
        WHERE r.ITEMNMBR = ?
          AND h.RECEIPTDATE >= DATEADD(year, -?, GETDATE())
          AND h.POPTYPE NOT IN (2, 4, 5)
        """
        cursor.execute(query, (item_number, lookback_years))
        row = cursor.fetchone()
        
        if row and row.TotalLines and row.TotalLines > 0:
            return (row.OnTimeLines / row.TotalLines) * 100.0, row.TotalLines

        # Fallback: Join POP30110 directly on PONUMBER + ITEMNMBR (less accurate if multiple lines same item)
        # Only if strict join returned nothing (likely POP10500 purged)
        query_fallback = """
        SELECT 
            COUNT(r.POPRCTNM) as TotalLines,
            SUM(CASE WHEN h.RECEIPTDATE <= pl.PRMSHPDTE THEN 1.0 ELSE 0.0 END) as OnTimeLines
        FROM POP30310 r
        JOIN POP30300 h ON r.POPRCTNM = h.POPRCTNM
        JOIN POP30110 pl ON r.PONUMBER = pl.PONUMBER AND r.ITEMNMBR = pl.ITEMNMBR
        WHERE r.ITEMNMBR = ?
          AND h.RECEIPTDATE >= DATEADD(year, -?, GETDATE())
          AND h.POPTYPE NOT IN (2, 4, 5)
          AND r.PONUMBER <> ''
        """
        cursor.execute(query_fallback, (item_number, lookback_years))
        row = cursor.fetchone()
        
        if row and row.TotalLines and row.TotalLines > 0:
             return (row.OnTimeLines / row.TotalLines) * 100.0, row.TotalLines
             
        return None, 0
    except Exception as e:
        # Don't spam logs for every item, just return None
        # LOGGER.warning(f"Error checking on-time perf for {item_number}: {e}")
        return None, 0

def get_monthly_averages(cursor, item_number, years=3):
    """
    Calculate average usage quantity per month over the last N years.
    Returns a dict {month_in (1-12): avg_qty}
    """
    try:
        query = """
        SELECT 
            MONTH(DOCDATE) as MonthNum,
            SUM(ABS(TRXQTY)) as TotalUsage
        FROM IV30300
        WHERE ITEMNMBR = ?
          AND TRXQTY < 0
          AND DOCDATE >= DATEADD(year, -?, GETDATE())
        GROUP BY MONTH(DOCDATE)
        """
        cursor.execute(query, (item_number, years))
        rows = cursor.fetchall()
        
        avgs = {}
        for row in rows:
            # We divide by 'years' to get the annual average for that month
            # Note: This is a simple average. If an item was new last year, this might under-report.
            # But for "Supplier Prep" usually looking at established items.
            avgs[row.MonthNum] = float(row.TotalUsage) / years
            
        return avgs
    except Exception as e:
        LOGGER.warning(f"Error calculating monthly avgs for {item_number}: {e}")
        return {}

def generate_report():
    print("🚀 Starting Supplier Meeting Prep Report...")
    
    today = datetime.date.today()
    results = []

    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Get Base Inventory Data (reusing reorder logic for efficiency)
        print("📦 Fetching inventory status...")
        # Get all raw materials
        inventory_df = reorder_math.get_reorder_recommendations(
            cursor, 
            include_classes=RAW_MATERIAL_CLASS_CODES,
            lookback_days=365, # Look at full year for general usage stats
            safety_days=14,
            location=PRIMARY_LOCATION,
            only_below_rop=False # Get EVERYTHING
        )
        
        if inventory_df.empty:
            print("❌ No inventory data found.")
            return

        # Filter for specific items as requested
        target_items = [
            "Hydrogen Peroxide", 
            "Phosphorous Acid", 
            "Citric Acid", 
            "Potassium Acetate", 
            "Sulfuric Acid"
        ]
        # Use simple string matching on description
        mask = inventory_df['item_description'].apply(
            lambda desc: any(target.lower() in str(desc).lower() for target in target_items)
        )
        inventory_df = inventory_df[mask]

        if inventory_df.empty:
            print("❌ No matching items found after filtering.")
            return

        print(f"🔍 Analyzing {len(inventory_df)} raw materials (filtered subset)...")
        
        # Iterate through items and enrich
        for _, row in inventory_df.iterrows():
            item_number = row['item_number']
            
            # --- A. Annual Usage & Spend ---
            # Reorder math gives avg_daily over 365 days (since we passed lookback_days=365)
            annual_usage_qty = row['avg_daily_usage'] * 365
            
            # Get current cost (approximate from price history or reorder row if available, 
            # but reorder row doesn't have cost. Let's fetch current cost from IV00101)
            # Actually market_insights.get_product_details has it, but that's heavy.
            # Let's just do a quick lookup or assume we can get it data-mined.
            # Efficient way: The reorder_math query didn't pull CURRCOST.
            # Let's fetch unit cost specifically.
            try:
                cursor.execute("SELECT CURRCOST, STNDCOST FROM IV00101 WHERE ITEMNMBR = ?", item_number)
                cost_row = cursor.fetchone()
                unit_cost = float(cost_row.CURRCOST) if cost_row and cost_row.CURRCOST > 0 else (float(cost_row.STNDCOST) if cost_row else 0)
            except:
                unit_cost = 0
                
            annual_spend = annual_usage_qty * unit_cost
            
            # --- B. Seasonal Forecast (Feb - Apr) ---
            # Using 3-year avg for this season
            current_year = today.year
            years_to_analyze = [current_year - 3, current_year - 2, current_year - 1]
            seasonal_forecast_qty = analyze_seasonal_needs.get_seasonal_usage(
                cursor, item_number, start_month=2, end_month=4, years=years_to_analyze
            )
            
            # --- C. Price Intelligence ---
            # Is now a good time to buy? 
            # market_insights.calculate_buying_signals is perfect here.
            # usage of fetch_product_price_history inside it is cached/fast enough? 
            # It runs a query per item. Might be slow for 500 items but acceptable for a one-off report.
            buying_signal = market_insights.calculate_buying_signals(cursor, item_number)
            price_score = buying_signal.get('subscores', {}).get('price', 50)
            price_percentile = buying_signal.get('percentile', 50)
            signal_reason = buying_signal.get('reason', '')

            # --- D. Supplier Performance ---
            on_time_pct, order_count = get_item_on_time_performance(cursor, item_number)
            
             # --- E. Monthly Average Spend ---
            monthly_avgs_qty = get_monthly_averages(cursor, item_number)
            monthly_spend_data = {}
            months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            for i, month_name in enumerate(months, start=1):
                avg_qty = monthly_avgs_qty.get(i, 0.0)
                monthly_spend_data[month_name] = avg_qty * unit_cost

            # Compile Row
            row_data = {
                "Item Number": item_number,
                "Description": row['item_description'],
                "Category": row['item_class'],
                "Primary Vendor": row['vendor_name'],
                "Vendor ID": row['vendor_id'],
                
                # Volume & Spend
                "Annual Usage (Qty)": annual_usage_qty,
                "Unit Cost ($)": unit_cost,
                "Annual Spend ($)": annual_spend,
                
                # Forecasting
                "Feb-Apr Forecast (Qty)": seasonal_forecast_qty,
                "Current On Hand": row['qty_on_hand'],
                "Days on Hand": row['days_of_coverage'],
                
                # Buying Intelligence
                "Price Percentile": price_percentile, # Low is good
                "Buy Score (0-100)": buying_signal.get('score', 0),
                "Recommendation": buying_signal.get('signal', 'N/A'),
                
                # Supplier Performance
                "On-Time Delivery %": on_time_pct if on_time_pct is not None else "N/A",
                "Orders Analyzed": order_count
            }
            
            # Add Monthly Columns
            row_data.update(monthly_spend_data)
            
            results.append(row_data)
            
            if len(results) % 10 == 0:
                print(f"   Processed {len(results)} items...")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by Annual Spend (High to Low)
    df.sort_values(by="Annual Spend ($)", ascending=False, inplace=True)
    
    # Save to Excel
    filename = f"Brenntag_Meeting_Prep_{today.strftime('%Y-%m-%d')}.xlsx"
    print(f"💾 Saving report to {filename}...")
    
    try:
        # Use openpyxl (standard in this env)
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Supplier Report', index=False)
            
            # Simple Column Adjustments using openpyxl
            worksheet = writer.sheets['Supplier Report']
            for column_cells in worksheet.columns:
                length = max(len(str(cell.value) or "") for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 50)
            
        print("✅ Report generated successfully!")
        
    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")

if __name__ == "__main__":
    generate_report()
