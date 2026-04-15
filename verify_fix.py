
import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history

def verify_fix():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = "CHELIGLIQ"
        print(f"Fetching history for {item}...")
        history = fetch_product_price_history(cursor, item)
        
        if not history:
            print("No history returned.")
            return

        first_record = history[0]
        print("Keys in first record:")
        print(list(first_record.keys()))
        
        if 'PONumber' in first_record:
            print("SUCCESS: PONumber key is present.")
        else:
            print("FAILURE: PONumber key is MISSING.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_fix()
