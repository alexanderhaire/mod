
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Finding valid Finished Good item...")
    cursor.execute("SELECT TOP 1 ITEMNMBR FROM IV30300 WHERE DOCTYPE=1") # 1 = Adjustment? Just want any trx
    row = cursor.fetchone()
    if row:
        print(f"Components found for: {row.ITEMNMBR}")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
