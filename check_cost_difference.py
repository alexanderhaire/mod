
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Checking UNITCOST vs ORUNTCST...")
    cursor.execute("""
        SELECT TOP 10 
            ITEMNMBR, 
            UNITCOST, 
            ORUNTCST
        FROM POP30310 
        WHERE UNITCOST <> ORUNTCST AND UNITCOST > 0
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} rows where UNITCOST <> ORUNTCST:")
        for row in rows:
            print(row)
    else:
        print("No rows found where UNITCOST <> ORUNTCST. They appear identical.")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
