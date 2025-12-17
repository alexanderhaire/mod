
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Checking Landed_Cost values (without join)...")
    cursor.execute("""
        SELECT TOP 10 
            ITEMNMBR, 
            UNITCOST, 
            Landed_Cost,
            EXTDCOST
        FROM POP30310 
        WHERE Landed_Cost <> 0
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} rows with Landed_Cost <> 0:")
        for row in rows:
            print(row)
    else:
        print("No rows found with Landed_Cost <> 0. Landed_Cost column might be unused.")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
