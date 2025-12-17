
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def inspect_misc_history():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MISCINTGR80'
        
        print(f"--- POP30310 Receipts for {item} (Dec 2025 onwards) ---")
        query = """
        SELECT 
            h.POPRCTNM,
            h.RECEIPTDATE,
            h.VENDORID,
            l.RCPTLNNM,
            l.ITEMNMBR,
            l.UNITCOST,
            l.EXTDCOST,
            l.ACTLSHIP
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.ITEMNMBR = ?
          AND h.RECEIPTDATE >= '2025-12-01'
        ORDER BY h.RECEIPTDATE
        """
        
        df = pd.read_sql(query, conn, params=[item])
        print(df.to_string())
        
        print(f"\n--- IV30300 Transactions for {item} (Dec 2025 onwards) ---")
        query_iv = """
        SELECT 
            h.DOCNUMBR,
            h.DOCDATE,
            t.TRXQTY,
            t.UNITCOST,
            t.EXTDCOST,
            h.IVDOCTYP
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        WHERE t.ITEMNMBR = ?
          AND h.DOCDATE >= '2025-12-01'
        ORDER BY h.DOCDATE
        """
        df_iv = pd.read_sql(query_iv, conn, params=[item])
        print(df_iv.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_misc_history()
