
import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_monthly_price_trends, fetch_product_price_history

def verify_extended_history():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MISCINTGR80'
        
        print(f"--- 1. Testing Monthly Trends (Should include 2018) ---")
        # Ensure we pass months=120 to cover 10 years
        df = fetch_monthly_price_trends(cursor, item, months=120)
        
        if not df.empty:
            print(f"Data Date Range: {df['Date'].min()} to {df['Date'].max()}")
            if df['Date'].min().year <= 2018:
                print("SUCCESS: Data goes back to 2018.")
            else:
                print(f"FAILURE: Data starts in {df['Date'].min().year}")
        else:
            print("No data returned for monthly trends.")

        print(f"\n--- 2. Testing Price History List (Default 10 years) ---")
        # Default now 3650 days
        history = fetch_product_price_history(cursor, item) # Uses default days=3650
        
        if history:
            oldest = min(h['TransactionDate'] for h in history)
            print(f"Oldest Transaction: {oldest}")
            if oldest.year <= 2018:
                print("SUCCESS: History list includes 2018.")
            else:
                print("FAILURE: History list does not include 2018.")
        else:
            print("No history returned.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_extended_history()
