import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_bom_raw():
    print("--- DIAGNOSTIC: BOM Raw Data ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("Fetching BM00101 (Top 10)...")
        headers = pd.read_sql("SELECT TOP 10 * FROM BM00101", conn)
        print(headers.to_string())
        
        print("\nFetching BM00111 (Top 10)...")
        components = pd.read_sql("SELECT TOP 10 * FROM BM00111", conn)
        print(components.to_string())
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_bom_raw()
