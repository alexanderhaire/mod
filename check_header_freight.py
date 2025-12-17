
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Checking POP30300 Freight...")
    cursor.execute("""
        SELECT TOP 10 
            POPRCTNM,
            ORFRTAMT,
            ORMISCAMT
        FROM POP30300 
        WHERE ORFRTAMT > 0
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} rows with Freight:")
        for row in rows:
            print(row)
    else:
        print("No rows found with Freight > 0.")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
