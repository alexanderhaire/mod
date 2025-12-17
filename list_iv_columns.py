
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    table_name = "IV10200"
    print(f"\nInspecting columns for {table_name}...")
    
    cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
    columns = [column[0] for column in cursor.description]
    
    for col in columns:
        print(f"- {col}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
