import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_pop_joins():
    print("--- DIAGNOSTIC: POP Table Inspection (No Joins) ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Check POP30310 (Lines) Raw
        print("Fetching Top 5 Lines from POP30310...")
        query_lines = """
        SELECT TOP 5 
            POPRCTNM, 
            ITEMNMBR, 
            UNITCOST, 
            EXTDCOST, 
            POPTYPE 
        FROM POP30310
        """
        print(pd.read_sql(query_lines, conn).to_string())
        
        # 2. Check POP30300 (Header) Raw for matching number
        print("\nFetching Matching Header from POP30300...")
        # Get one Receipt Number from above
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 POPRCTNM FROM POP30310")
        rct_num = cursor.fetchone()[0]
        
        query_header = f"SELECT * FROM POP30300 WHERE POPRCTNM = '{rct_num}'"
        print(pd.read_sql(query_header, conn).to_string())
        
        # 3. Check POP10500 (Receipt Line Items - Open?)
        # Maybe data is in Open tables, not History?
        print("\nChecking POP10500 (Open Receipts)...")
        try:
            query_open = "SELECT TOP 5 POPRCTNM, ITEMNMBR FROM POP10500"
            print(pd.read_sql(query_open, conn).to_string())
        except Exception as e:
            print(f"POP10500 Error: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_pop_joins()
