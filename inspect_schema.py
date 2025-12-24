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
    for table_name in ["POP30310", "POP10500", "POP10110", "POP30110"]:
        print(f"\nInspecting columns for {table_name}...")
        
        try:
            cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
            columns = [column[0] for column in cursor.description]
            print(f"  Columns: {', '.join(columns)}")
            
            # Specifically check for QTYINVCD
            if "QTYINVCD" in columns:
                print(f"  ✓ QTYINVCD found in {table_name}!")
        except Exception as table_err:
            print(f"  ✗ Could not query {table_name}: {table_err}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
