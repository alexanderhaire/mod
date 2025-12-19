import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history
import datetime

def verify_internal_history():
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Pick a Finished Good known to have history (likely 'Omega' related or general FG)
        # Using a dummy or finding one. Let's try 'OMFISHEM02' from previous context or generic.
        # Actually, let's just query one that has negative usage in IV30300 to prove it's ignored.
        
        item_number = 'OMFISHEM02' 
        
        print(f"Fetching history for {item_number}...")
        history = fetch_product_price_history(cursor, item_number, days=3650)
        
        neg_qty_count = 0
        total_qty = 0
        
        for record in history:
            qty = record.get('Quantity', 0)
            if qty < 0:
                neg_qty_count += 1
                print(f"FAIL: Found negative quantity record! Date: {record['TransactionDate']}, Qty: {qty}")
            total_qty += qty
            
        print(f"Total Records: {len(history)}")
        print(f"Negative Qty Records: {neg_qty_count}")
        print(f"Total Positive Qty Sum: {total_qty}")
        
        if neg_qty_count == 0:
            print("PASS: No negative quantities found in history.")
        else:
            print("FAIL: Negative quantities still exist.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_internal_history()
