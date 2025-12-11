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
    
    
    # Check both tables
    for table_name in ["POP30310", "POP10500"]:
        print(f"\nInspecting columns for {table_name}...")
        
        cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
        columns = [column[0] for column in cursor.description]
        
        # Check if QTYSHPPD exists
        if "QTYSHPPD" in columns:
            print(f"  ✓ QTYSHPPD found in {table_name}")
            idx = columns.index("QTYSHPPD")
            row = cursor.fetchone()
            if row:
                print(f"  Value: {row[idx]} (Type: {type(row[idx])})")
        else:
            print(f"  ✗ QTYSHPPD NOT in {table_name}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
