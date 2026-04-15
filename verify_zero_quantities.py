import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def verify_zero_quantities():
    print("--- VERIFYING CURRENT QUANTITIES IN GP ---")
    
    # 1. Read the CSV to get the list of items
    csv_path = "year_end_adjustments.csv"
    try:
        csv_df = pd.read_csv(csv_path)
        items_to_check = csv_df['Item Number'].unique().tolist()
        print(f"Loaded {len(items_to_check)} items from {csv_path}")
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # 2. Query GP for these items
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Format list for SQL IN clause
        # Handling single quotes in item numbers if necessary, though simpler to pass as params or just iterate if list is huge.
        # Given 600 items, we can fetch all MAIN inventory and filter in python to avoid massive SQL query string issues
        
        print("Querying GP for current QTYONHND...")
        query = """
        SELECT ITEMNMBR, QTYONHND
        FROM IV00102
        WHERE LOCNCODE = 'MAIN'
        """
        
        gp_df = pd.read_sql(query, conn)
        conn.close()
        
        gp_df['ITEMNMBR'] = gp_df['ITEMNMBR'].astype(str).str.strip()
        
    except Exception as e:
        print(f"Failed to query DB: {e}")
        return

    # 3. Check for non-zero quantities for the items in our list
    # Filter gp_df to only items in our list
    relevant_inventory = gp_df[gp_df['ITEMNMBR'].isin(items_to_check)]
    
    # Find items that are NOT zero
    non_zero = relevant_inventory[relevant_inventory['QTYONHND'] != 0]
    
    # Also items that might not be in the returned list (implying 0 or not existing in IV00102 for MAIN)
    # If it's not in IV00102, it effectively has 0 QTY for that location usually, or it's an error.
    
    print(f"\nResults:")
    if non_zero.empty:
        print("SUCCESS: All items from the list have 0 Quantity on Hand in MAIN.")
    else:
        print(f"WARNING: Found {len(non_zero)} items that are NOT zeroed out:")
        print(non_zero[['ITEMNMBR', 'QTYONHND']].to_string())
        
        # Calculate stats
        print(f"\nSummary:")
        print(f"Total Items Checked: {len(items_to_check)}")
        print(f"Items with Non-Zero Qty: {len(non_zero)}")
        
        # Check if they are exactly equal to the adjustment (which would be weird, but let's see)
        # We can merge with CSV to see
        merged = pd.merge(non_zero, csv_df, left_on='ITEMNMBR', right_on='Item Number')
        # print("\nComparison with Adjustment Value from CSV:")
        # print(merged[['Item Number', 'QTYONHND', 'Trx Qty']].head().to_string())

if __name__ == "__main__":
    verify_zero_quantities()
