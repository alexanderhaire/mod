import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_iv00103_v2():
    print("--- DIAGNOSTIC: Item Vendor Master (IV00103) - Fixed ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Check SCHEMA
        print("Checking basic schema...")
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 * FROM IV00103")
        cols = [column[0] for column in cursor.description]
        print(f"Columns: {cols}")
        cursor.close() # Close cursor to free connection for pandas
        
        # Check DATA
        print("\nFetching Sample Vendor Pricing...")
        query = """
        SELECT TOP 20 
            ITEMNMBR, 
            VENDORID, 
            Last_Originating_Cost, 
            PRCHSUOM
        FROM IV00103
        WHERE Last_Originating_Cost > 0
        ORDER BY Last_Originating_Cost DESC
        """
        data = pd.read_sql(query, conn)
        print(data.to_string())
        
        if not data.empty:
            print("\nFound vendor pricing data! This is our 'Offer Sheet' source.")
        else:
            print("\nNo pricing data found in IV00103.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_iv00103_v2()
