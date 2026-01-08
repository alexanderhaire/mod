import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
from datetime import date

def verify_cost():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    target_date = date(2024, 1, 1)
    
    print(f"--- Verification for Date: {target_date} ---")
    
    # 1. Get items that have receipts at NON-MAIN locations
    print("Fetching items with receipts at OTHER locations...")
    query_history_items = """
    SELECT TOP 20 ITEMNMBR 
    FROM IV30300 
    WHERE DOCDATE <= ? AND DOCTYPE IN (4, 1) AND TRIM(TRXLOCTN) <> 'MAIN'
    GROUP BY ITEMNMBR
    ORDER BY MAX(DOCDATE) DESC
    """
    history_items_df = pd.read_sql(query_history_items, conn, params=[target_date])
    
    if history_items_df.empty:
        print("No items with history found.")
        return

    item_list = "', '".join(history_items_df['ITEMNMBR'].tolist())
    
    # 2. Get Current Cost for these items
    print(f"Fetching current cost for {len(history_items_df)} items...")
    query_base = f"SELECT ITEMNMBR, CURRCOST FROM IV00101 WHERE ITEMNMBR IN ('{item_list}')"
    base_df = pd.read_sql(query_base, conn)
    
    # 3. Run the new Historical Cost Logic for these items
    print("Running historical cost query...")
    query_cost = f"""
    WITH RatedTransactions AS (
        SELECT 
            ITEMNMBR,
            UNITCOST,
            DOCDATE,
            ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
        FROM IV30300
        WHERE DOCDATE <= ?
          AND TRIM(TRXLOCTN) = 'MAIN'
          AND DOCTYPE IN (4, 1) 
          AND UNITCOST > 0
          AND ITEMNMBR IN ('{item_list}')
    )
    SELECT ITEMNMBR, UNITCOST as LastCost, DOCDATE
    FROM RatedTransactions
    WHERE rn = 1
    """
    params = [target_date]
    cost_df = pd.read_sql(query_cost, conn, params=params)
    
    # 3. Compare
    print("\n--- COMPARISON ---")
    merged = pd.merge(base_df, cost_df, on='ITEMNMBR', how='inner')
    
    if merged.empty:
        print("No overlapping items found between sample and history.")
    else:
        merged['Diff'] = merged['CURRCOST'] - merged['LastCost']
        print(merged[['ITEMNMBR', 'CURRCOST', 'LastCost', 'DOCDATE', 'Diff']])
        
        diffs = merged[merged['Diff'].abs() > 0.001]
        if not diffs.empty:
            print("\nSUCCESS: Found items with historical cost different from current cost:")
            print(diffs[['ITEMNMBR', 'CURRCOST', 'LastCost']])
        else:
            print("\nWARNING: No cost differences found. This might be correct if costs haven't changed, but let's double check.")

    conn.close()

if __name__ == "__main__":
    verify_cost()
