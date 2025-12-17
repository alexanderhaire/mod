
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    receipt = 'RV0000008166'
    print(f"Dumping IV10200 for {receipt}...")
    
    cursor.execute(f"SELECT * FROM IV10200 WHERE RCPTNMBR='{receipt}'")
    
    columns = [column[0] for column in cursor.description]
    print(f"Columns: {columns}")
    
    rows = cursor.fetchall()
    for row in rows:
        r = dict(zip(columns, row))
        print(r)
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
