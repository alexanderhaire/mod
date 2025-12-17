
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def debug_investigation():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MISCINTGR80'
        
        print(f"--- 1. Investigating Nov 27, 2024 (Double Counting?) ---")
        # Check raw lines for this date
        query_date = """
        SELECT 
            h.POPRCTNM,
            h.POPTYPE,
            h.VNDDOCNM,
            h.RECEIPTDATE,
            l.ITEMNMBR,
            l.ACTLSHIP,
            l.UNITCOST
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.ITEMNMBR = ?
          AND h.RECEIPTDATE = '2024-11-27'
        """
        df_date = pd.read_sql(query_date, conn, params=[item])
        print(df_date.to_string())
        
        print(f"\n--- 2. Checking Old History (2018) ---")
        # Check if we have data from 2018
        query_hist = """
        SELECT TOP 10
            h.RECEIPTDATE,
            l.ACTLSHIP,
            h.POPTYPE
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.ITEMNMBR = ?
          AND h.RECEIPTDATE BETWEEN '2018-01-01' AND '2018-12-31'
        ORDER BY h.RECEIPTDATE
        """
        df_hist = pd.read_sql(query_hist, conn, params=[item])
        print(df_hist.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_investigation()
