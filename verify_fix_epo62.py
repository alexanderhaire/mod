
import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_monthly_price_trends

def verify_fix():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MISCINTGR80'
        print(f"--- Verifying Purchase History for {item} (Dec 2025) ---")
        
        # Test fetch_monthly_price_trends which we modified
        df = fetch_monthly_price_trends(cursor, item, months=1)
        
        print(df.to_string())
        
        if not df.empty:
            dec_row = df[df['Month'] == 12]
            if not dec_row.empty:
                receipt_count = dec_row.iloc[0]['Receipts']
                print(f"\nResult: Receipt Count for Dec: {receipt_count}")
                if receipt_count == 1:
                    print("SUCCESS: Only 1 receipt counted (fix works).")
                else:
                    print(f"FAILURE: {receipt_count} receipts counted (expected 1).")
            else:
                print("No December data found.")
        else:
            print("No data returned.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_fix()
