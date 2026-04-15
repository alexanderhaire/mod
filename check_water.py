import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def check_water():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("--- HISTORY FOR H2OCOLD ---")
        query = """
        SELECT TOP 20
            DOCDATE,
            DOCTYPE,
            TRXQTY,
            UNITCOST,
            TRXLOCTN
        FROM IV30300
        WHERE ITEMNMBR = 'H2OCOLD'
        ORDER BY DOCDATE DESC, DEX_ROW_ID DESC
        """
        df = pd.read_sql(query, conn)
        print(df.to_string())
        
        print("\n--- TOTAL RECEIPTS VS USAGE ---")
        query_sum = """
        SELECT 
            DOCTYPE,
            SUM(TRXQTY) as TotalQty
        FROM IV30300
        WHERE ITEMNMBR = 'H2OCOLD'
        GROUP BY DOCTYPE
        """
        df_sum = pd.read_sql(query_sum, conn)
        print(df_sum.to_string())
        
        # Check Item Master for UofM
        query_master = """
        SELECT ITEMNMBR, ITEMDESC, UOMSCHDL, DECPLQTY
        FROM IV00101
        WHERE ITEMNMBR = 'H2OCOLD'
        """
        print("\n--- ITEM MASTER ---")
        print(pd.read_sql(query_master, conn).to_string())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_water()
