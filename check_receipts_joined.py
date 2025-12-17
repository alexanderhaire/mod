
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def check_receipts_joined():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print(f"--- Receipts for MISCINTGR80 in Dec 2025 ---")
        
        query = """
        SELECT 
            h.POPRCTNM,
            h.VNDDOCNM,
            h.RECEIPTDATE,
            h.VOIDSTTS,
            h.TRXSORCE,
            l.ITEMNMBR,
            l.ACTLSHIP,
            l.UNITCOST
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.ITEMNMBR = 'MISCINTGR80'
          AND h.RECEIPTDATE >= '2025-12-01'
        ORDER BY h.RECEIPTDATE
        """
        
        df = pd.read_sql(query, conn)
        print(df.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_receipts_joined()
