
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from market_insights import fetch_product_price_history
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # We need to find a finished good to test. 
    # Usually items without receipts in POP30310 but with inventory history.
    # Let's just pick an item that IS NOT a raw material.
    # Or force the function to behave as if it's not a raw material by mocking?
    # Easier to just find an FG. 'PHOS 75' was mentioned in conversation history as a product.
    
    item = 'CHEGLUCO' 
    print(f"Fetching history for {item} (expecting FG path)...")
    
    history = fetch_product_price_history(cursor, item)
    
    if history:
        print(f"Found {len(history)} records.")
        first = history[0]
        print("First Record Keys:", first.keys())
        print(f"AvgCost: {first.get('AvgCost')}")
        print(f"LandedCost: {first.get('LandedCost')}")
        print(f"DeliveredCost: {first.get('DeliveredCost')}")
        
        if 'LandedCost' in first and 'DeliveredCost' in first:
            print("SUCCESS: Keys present.")
        else:
            print("FAILURE: Keys missing.")
    else:
        print("No history found for PHOS 75. Trying another...")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
