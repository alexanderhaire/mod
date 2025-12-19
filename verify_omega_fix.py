import pyodbc
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history
import datetime

def verify_fix():
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item_number = 'GRPFISHEM'
        history = fetch_product_price_history(cursor, item_number, days=3650)
        
        # Look for 2019-04-03
        target_date = datetime.date(2019, 4, 3)
        found = False
        for record in history:
            if record['TransactionDate'] == target_date:
                found = True
                landed = record['LandedCost']
                delivered = record['DeliveredCost']
                print(f"Date: {record['TransactionDate']}")
                print(f"Landed Cost: {landed}")
                print(f"Delivered Cost: {delivered}")
                
                if landed > delivered:
                     print("FAIL: Landed Cost > Delivered Cost")
                elif landed == delivered:
                     print("PASS: Landed Cost clamped to Delivered Cost (or equal)")
                else:
                     print("PASS: Landed Cost < Delivered Cost")
                break
        
        if not found:
            print("Date not found in history.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_fix()
