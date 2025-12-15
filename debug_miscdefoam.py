
import pyodbc
from secrets_loader import build_connection_string
import pandas as pd

def inspect_defoam():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    print("--- 1. ITEM MASTER (IV00101) ---")
    query_item = "SELECT ITEMNMBR, ITEMDESC, ITMCLSCD, ITEMTYPE, CURRCOST FROM IV00101 WHERE ITEMNMBR = 'MISCDEFOAM'"
    df_item = pd.read_sql(query_item, conn)
    print(df_item.to_string())
    
    print("\n--- 2. PURCHASE HISTORY (POP30300) ---")
    query_hist = """
    SELECT TOP 5 h.RECEIPTDATE, h.VENDORID, h.VENDNAME
    FROM POP30310 l
    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
    WHERE l.ITEMNMBR = 'MISCDEFOAM'
    ORDER BY h.RECEIPTDATE DESC
    """
    df_hist = pd.read_sql(query_hist, conn)
    print(df_hist.to_string())

    print("\n--- 3. USAGE HISTORY (IV30300) ---")
    query_usage = """
    SELECT COUNT(*) as TrxCount, SUM(ABS(TRXQTY)) as TotalQty
    FROM IV30300
    WHERE ITEMNMBR = 'MISCDEFOAM'
      AND DOCTYPE = 1 -- Adjustment/Sale? (Standard checks usually exclude transfers, but let's see)
      AND DOCDATE >= DATEADD(day, -180, GETDATE())
    """
    df_usage = pd.read_sql(query_usage, conn)
    print(df_usage.to_string())

    print("\n--- 4. RAW MATERIAL CLASS CHECK ---")
    # Check if its class is in the allowed list
    itm_class = df_item['ITMCLSCD'].iloc[0] if not df_item.empty else "N/A"
    print(f"Class Code: '{itm_class}'")

    print("\n--- 5. INVENTORY CHECK (IV00102 - MAIN) ---")
    query_inv = "SELECT QTYONHND, QTYONORD, ATYALLOC FROM IV00102 WHERE ITEMNMBR = 'MISCDEFOAM' AND LOCNCODE = 'MAIN'"
    df_inv = pd.read_sql(query_inv, conn)
    print(df_inv.to_string())

    print("\n--- 6. FULL PRIORITY RANKING CHECK ---")
    from market_insights import get_priority_raw_materials
    
    # Run with same limit as App (75)
    df_prio = get_priority_raw_materials(conn.cursor(), limit=75, require_purchase_history=True)
    
    # Check if MISCDEFOAM is in list
    if not df_prio.empty:
        rank = df_prio[df_prio['ITEMNMBR'].str.strip() == 'MISCDEFOAM']
        if not rank.empty:
            print(f"FOUND! It is in the top 75.")
            print(rank.to_string())
        else:
            print("NOT FOUND in top 75.")
            print(f"Total items returned: {len(df_prio)}")
            print(f"Lowest Spend in list: ${df_prio['EstAnnualSpend'].min():,.2f}")
            
            # Run again with higher limit to find where it is
            print("\nSearching with limit=500...")
            df_all = get_priority_raw_materials(conn.cursor(), limit=500, require_purchase_history=True)
            rank_all = df_all[df_all['ITEMNMBR'].str.strip() == 'MISCDEFOAM']
            if not rank_all.empty:
                 idx = df_all.index[df_all['ITEMNMBR'].str.strip() == 'MISCDEFOAM'].tolist()[0]
                 print(f"Found at Rank #{idx + 1}")
                 print(rank_all.to_string())
            else:
                print("STILL NOT FOUND even with limit=500. Something else is wrong.")
    else:
        print("Error: DataFrame is empty")

if __name__ == "__main__":
    inspect_defoam()
