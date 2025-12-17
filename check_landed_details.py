
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    receipt_num = 'RV0000039639' 
    item_num = 'NPKKNO3'

    print(f"Checking Landed Cost for {receipt_num}...")
    
    # Check POP30390 (Landed Cost History)
    print("--- POP30390 ---")
    cursor.execute(f"SELECT * FROM POP30390 WHERE POPRCTNM = '{receipt_num}'")
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(dict(zip(columns, row)))
    else:
        print("No rows in POP30390.")

    # Check IV10200 (Purchase Receipts Layer) - This holds the inventory cost
    print("\n--- IV10200 ---")
    # Need to match by RCTSEQNM usually, or just item/date
    cursor.execute(f"""
        SELECT RCPTSOLD, UNITCOST, ADJUNITCOST, * 
        FROM IV10200 
        WHERE ITEMNMBR = '{item_num}' AND DATERECD = '2025-12-12'
    """)
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            r = dict(zip(columns, row))
            print(f"Layer Cost: {r['UNITCOST']}, Adj Cost: {r['ADJUNITCOST']}, Qty: {r['QTYRECVD']}")
    else:
        print("No rows in IV10200 for this date.")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
