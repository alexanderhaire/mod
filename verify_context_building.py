import pyodbc
import pandas as pd
import datetime as dt
import calendar
from secrets_loader import build_connection_string
from market_insights import get_product_details
from context_utils import summarize_sql_context

def verify_context():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'NO3CA12'
        print(f"Fetching details for {item}...")
        
        # 1. Fetch Data (Mocking app.py)
        details = get_product_details(cursor, item)
        category = details.get('category', 'default')
        inventory = details.get('inventory_status', {})
        price_hist = details.get('price_history', [])
        usage_hist = details.get('usage_history', [])
        
        print(f"Raw Usage History Entries: {len(usage_hist)}")
        if usage_hist:
            print(f"Sample Entry: {usage_hist[0]}")
        else:
             print("Usage History is EMPTY.")
        
        # 2. Build Context (Copying logic from app.py)
        
        # Price Summary
        price_summary = {}
        if price_hist:
            hist_df = pd.DataFrame(price_hist)
            if not hist_df.empty and 'AvgCost' in hist_df.columns:
                hist_df['TransactionDate'] = pd.to_datetime(hist_df['TransactionDate'])
                price_summary = {
                    'avg_cost': float(hist_df['AvgCost'].mean()),
                    'monthly_spend_current_year': {}
                }
                # Monthly Spend
                current_year = dt.date.today().year
                year_df = hist_df[hist_df['TransactionDate'].dt.year == current_year]
                if not year_df.empty:
                    monthly_spend = year_df.groupby(year_df['TransactionDate'].dt.month)['ExtendedCost'].sum().to_dict()
                    price_summary['monthly_spend_current_year'] = {calendar.month_abbr[m]: v for m, v in monthly_spend.items()}

        # Usage Summary (New Logic)
        usage_summary = {}
        usage_hist = details.get('usage_history', [])
        if usage_hist:
            us_df = pd.DataFrame(usage_hist)
            if not us_df.empty and 'UsageQty' in us_df.columns:
                if 'Year' in us_df.columns and 'Month' in us_df.columns:
                    us_df['Date'] = pd.to_datetime(us_df[['Year', 'Month']].assign(day=1))
                    
                    usage_summary = {
                        'total_usage_qty': float(us_df['UsageQty'].sum()),
                        'avg_usage_qty': float(us_df['UsageQty'].mean()),
                        'usage_trend': 'increasing' if len(us_df) > 1 and us_df.iloc[-1]['UsageQty'] > us_df['UsageQty'].mean() else 'stable'
                    }
                    # Monthly Current
                    current_year = dt.date.today().year
                    us_year_df = us_df[us_df['Date'].dt.year == current_year]
                    if not us_year_df.empty:
                        monthly_usage = us_year_df.groupby(us_year_df['Date'].dt.month)['UsageQty'].sum().to_dict()
                        usage_summary['monthly_usage_current_year'] = {calendar.month_abbr[m]: v for m, v in monthly_usage.items()}
                    
                    # Monthly Last Year
                    last_year = current_year - 1
                    us_last_year_df = us_df[us_df['Date'].dt.year == last_year]
                    if not us_last_year_df.empty:
                        last_year_usage = us_last_year_df.groupby(us_last_year_df['Date'].dt.month)['UsageQty'].sum().to_dict()
                        usage_summary['monthly_usage_last_year'] = {calendar.month_abbr[m]: v for m, v in last_year_usage.items()}
        
        # Build Context Dict
        context = {
            'selected_item': item,
            'item_description': details.get('description', ''),
            'category': category,
            'current_cost': inventory.get('CURRCOST', 0),
            'stock_status': inventory.get('StockStatus', 'Unknown'),
            'on_hand': inventory.get('TotalOnHand', 0),
            'available': inventory.get('Available', 0),
            'on_order': inventory.get('OnOrder', 0),
            'price_history_summary': price_summary,
            'usage_history_summary': usage_summary,
            'demand_forecast': details.get('demand_forecast', {}).get('forecast_next_30days', 'stable')
        }
        
        # 3. Generate Final String
        final_prompt = summarize_sql_context(context)
        print("\n--- Generated AI Context ---")
        print(final_prompt)
        
        # Check for key phrases
        if "monthly_usage_last_year" in final_prompt:
            print("\nSUCCESS: Last Year Usage found in context.")
        else:
            print("\nFAILURE: Last Year Usage NOT found.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_context()
