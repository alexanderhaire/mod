
import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history

def verify_final():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MISCINTGR80'
        print(f"--- Verifying Nov 27, 2024 Count for {item} ---")
        
        history = fetch_product_price_history(cursor, item) # Uses updated logic
        
        nov27_transactions = [h for h in history if h['TransactionDate'].strftime('%Y-%m-%d') == '2024-11-27']
        
        print(f"Nov 27 Transactions Found: {len(nov27_transactions)}")
        for t in nov27_transactions:
            print(t)
            
        if len(nov27_transactions) == 1:
            print("SUCCESS: Count is 1.")
        else:
            print(f"FAILURE: Count is {len(nov27_transactions)}.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_final()
