import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
from market_insights import fetch_monthly_price_trends, fetch_product_price_history

def debug():
    conn_str, _, _, _ = build_connection_string()
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        
        item = 'NPK3011'
        print(f"--- Debugging {item} ---")
        
        # 1. Check Price History (used for signals)
        print("\n1. fetch_product_price_history:")
        history = fetch_product_price_history(cursor, item)
        print(f"Found {len(history)} records")
        if history:
            print(history[0])
            
        # 2. Check Monthly Trends (used for chart)
        print("\n2. fetch_monthly_price_trends:")
        trends = fetch_monthly_price_trends(cursor, item)
        print(f"Found {len(trends)} records")
        if not trends.empty:
            print(trends.head())
        else:
            print("Trends DataFrame is empty.")
            
        # 3. Check Raw Data in SOP30300
        print("\n3. Raw SOP30300 count:")
        cursor.execute("SELECT COUNT(*) FROM SOP30300 WHERE ITEMNMBR = ?", item)
        print(cursor.fetchone()[0])
        
        # 4. Check Raw Data in POP30310 (just in case)
        print("\n4. Raw POP30310 count:")
        cursor.execute("SELECT COUNT(*) FROM POP30310 WHERE ITEMNMBR = ?", item)
        print(cursor.fetchone()[0])

        # 5. Check Raw Data in IV30300 (Inventory Trx)
        print("\n5. Raw IV30300 count:")
        cursor.execute("SELECT COUNT(*) FROM IV30300 WHERE ITEMNMBR = ?", item)
        print(cursor.fetchone()[0])

if __name__ == "__main__":
    debug()
