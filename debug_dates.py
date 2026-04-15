import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_dates():
    print("--- DIAGNOSTIC: Manufacturing Dates ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("Checking MOP1000 Date Range...")
        query_mop = """
        SELECT 
            MIN(DATERECD) as FirstReceipt,
            MAX(DATERECD) as LastReceipt,
            COUNT(*) as TotalOrders,
            COUNT(CASE WHEN DATERECD > '1900-01-01' THEN 1 END) as OrdersWithReceipts
        FROM MOP1000
        """
        print(pd.read_sql(query_mop, conn).to_string())
        
        print("\nChecking PK010033 Activity...")
        query_pk = """
        SELECT TOP 10
            QTY_ISSUED_I,
            ITEMNMBR
        FROM PK010033
        WHERE QTY_ISSUED_I > 0
        ORDER BY DEX_ROW_ID DESC
        """
        print(pd.read_sql(query_pk, conn).to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_dates()
