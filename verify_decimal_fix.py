import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history, fetch_monthly_price_trends
import pandas as pd
import decimal

def verify_fix():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Test Item (use one that exists)
        item_number = 'NO3CA' 
        
        print(f"--- Testing {item_number} ---")
        
        # 1. Price History
        print("Checking Price History...")
        history = fetch_product_price_history(cursor, item_number)
        if history:
            row = history[0]
            avg_cost = row.get('AvgCost')
            print(f"Price History AvgCost Type: {type(avg_cost)}")
            if isinstance(avg_cost, decimal.Decimal):
                print("❌ FAIL: AvgCost is Decimal")
            else:
                print("✅ PASS: AvgCost is not Decimal")
        else:
            print("⚠️ No history found.")

        # 2. Monthly Trends
        print("\nChecking Monthly Trends...")
        df = fetch_monthly_price_trends(cursor, item_number)
        if not df.empty:
             val = df.iloc[0]['AvgCost']
             print(f"Monthly Trends AvgCost Type: {type(val)}")
             if isinstance(val, decimal.Decimal):
                 print("❌ FAIL: AvgCost is Decimal")
             else:
                 print("✅ PASS: AvgCost is not Decimal")
        else:
             print("⚠️ No trends found.")
             
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_fix()
