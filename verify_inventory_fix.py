import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_inventory_trends

def verify_fix():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'NO3CA12'
        print(f"Fetching inventory trends for {item}...")
        
        data = fetch_product_inventory_trends(cursor, item)
        
        print("\n--- Result ---")
        print(f"On Hand: {data.get('TotalOnHand')}")
        print(f"Allocated: {data.get('TotalAllocated')}")
        print(f"Available: {data.get('Available')}")
        print(f"On Order: {data.get('OnOrder')}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_fix()
