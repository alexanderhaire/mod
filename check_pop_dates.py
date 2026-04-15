import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def check_pop_dates():
    print("--- DIAGNOSTIC: POP Receipts Dates ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Check MIN/MAX date in POP30300 (History Header)
        print("Checking POP30300 Date Range...")
        query = """
        SELECT 
            MIN(receiptdate) as FirstReceipt,
            MAX(receiptdate) as LastReceipt,
            COUNT(*) as TotalReceipts
        FROM POP30300
        """
        print(pd.read_sql(query, conn).to_string())
        
        # Check if we have lines for these receipts
        print("\nChecking POP30310 Line Count...")
        query_lines = "SELECT COUNT(*) as TotalLines FROM POP30310"
        print(pd.read_sql(query_lines, conn).to_string())
        
        # Check Sample
        print("\nSample Receipt Header:")
        query_sample = "SELECT TOP 1 POPRCTNM, receiptdate, VENDORID FROM POP30300 ORDER BY receiptdate DESC"
        print(pd.read_sql(query_sample, conn).to_string())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    check_pop_dates()
