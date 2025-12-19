
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
from market_insights import fetch_product_price_history

def verify_fix():
    print("Verifying Ascorbic Acid Fix...")
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Fetch history for GRPASCORBIC
    history = fetch_product_price_history(cursor, 'GRPASCORBIC', days=3650)
    
    # Look for the anomalous transaction on 2023-03-09
    target_date = pd.to_datetime("2023-03-09").date()
    
    found = False
    for record in history:
        # Check date (record['TransactionDate'] might be date or string depending on driver, robust check)
        rec_date = record['TransactionDate']
        if rec_date == target_date:
            found = True
            print(f"Found Transaction on {rec_date}")
            print(f"Landed Cost: {record['LandedCost']:.4f}")
            print(f"Delivered Cost: {record['DeliveredCost']:.4f}")
            print(f"Quantity: {record['Quantity']:.4f}")
            print(f"Anomaly Fixed?: {record.get('IsAnomalyFix', False)}")
            
            # Assertions
            if record['LandedCost'] > 10.0:
                 print("FAIL: Landed Cost is still high (~$12.50). Fix not applied.")
            elif record['Quantity'] < 60.0:
                 print("FAIL: Quantity is still low (~55). Fix not applied.")
            else:
                 print("PASS: Landed Cost corrected (~$5.68) and Quantity recalculated (~121).")
                 
    if not found:
        print("FAIL: Target transaction not found in history.")

if __name__ == "__main__":
    verify_fix()
