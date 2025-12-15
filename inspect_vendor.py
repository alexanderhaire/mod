
import pyodbc
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    print(f"Connecting with: {conn_str}")
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    table_name = "PM00200"
    print(f"\nInspecting columns for {table_name}...")
    
    cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
    columns = [column[0] for column in cursor.description]
    print(f"Columns: {columns}")
    
    required = ["ADDRESS1", "ADDRESS2", "CITY", "STATE", "ZIPCODE"]
    found = [c for c in columns if c in required]
    print(f"Found required: {found}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
