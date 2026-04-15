import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def check():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Base Item Info
        print("Fetching BIAMZ02...")
        query = """
        SELECT 
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.CURRCOST,
            T2.QTYONHND as CurrentQty,
            T2.LOCNCODE
        FROM IV00101 T1
        JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        WHERE T1.ITEMNMBR = 'BIAMZ02'
        """
        df = pd.read_sql(query, conn)
        print(df)
        
        # 2. Get Last Receipt Cost
        query_cost = """
        SELECT TOP 5
            ITEMNMBR,
            UNITCOST,
            DOCDATE,
            DOCTYPE
        FROM IV30300
        WHERE ITEMNMBR = 'BIAMZ02'
          AND DOCTYPE = 4 -- Receipt
        ORDER BY DOCDATE DESC
        """
        cost_df = pd.read_sql(query_cost, conn)
        print("\nRecent Receipts:")
        print(cost_df)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check()
