
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Searching for receipts with multiple lines for same item...")
    cursor.execute("""
        SELECT TOP 5 
            POPRCTNM, ITEMNMBR, COUNT(*) as LineCount
        FROM POP30310
        GROUP BY POPRCTNM, ITEMNMBR
        HAVING COUNT(*) > 1
    """)
    
    rows = cursor.fetchall()
    
    if rows:
        for r in rows:
            print(f"\nReceipt: {r.POPRCTNM}, Item: {r.ITEMNMBR}, Count: {r.LineCount}")
            cursor.execute(f"""
                SELECT RCPTLNNM, LOCNCODE, UMQTYINB, UNITCOST, EXTDCOST 
                FROM POP30310 
                WHERE POPRCTNM = '{r.POPRCTNM}' AND ITEMNMBR = '{r.ITEMNMBR}'
            """)
            lines = cursor.fetchall()
            for l in lines:
                print(f"  Line: {l.RCPTLNNM}, Loc: '{l.LOCNCODE}', Qty: {l.UMQTYINB}, Cost: {l.UNITCOST}, Ext: {l.EXTDCOST}")
                
    conn.close()
except Exception as e:
    print(f"Error: {e}")
