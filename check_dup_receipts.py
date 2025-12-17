
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def check_duplicates_nov27():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        receipts = ['RV0000038118', 'RV0000038144']
        print(f"--- Checking Receipts: {receipts} ---")
        
        placeholders = ','.join(['?'] * len(receipts))
        query = f"""
        SELECT 
            h.POPRCTNM,
            h.RECEIPTDATE,
            h.POPTYPE,
            h.VOIDSTTS,
            h.TRXSORCE,
            h.VNDDOCNM,
            h.BACHNUMB,
            l.ITEMNMBR,
            l.ACTLSHIP,
            l.EXTDCOST
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE h.POPRCTNM IN ({placeholders})
        """
        
        df = pd.read_sql(query, conn, params=receipts)
        print(df.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_duplicates_nov27()
