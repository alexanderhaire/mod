import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def spot_check():
    items = ['ZZ2.5GALF', 'GOLDFE02', 'ZZ55GAL', 'NPKKTS', 'ZZ30GAL']
    cutoff = date(2025, 9, 30)
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print(f"Checking items: {items}")
        print(f"Cutoff Date: {cutoff}")
        
        # 1. Current Qty
        print("\n--- Current IV00102 (MAIN) ---")
        q1 = f"SELECT ITEMNMBR, QTYONHND FROM IV00102 WHERE ITEMNMBR IN ({','.join(['?']*len(items))}) AND LOCNCODE='MAIN'"
        df1 = pd.read_sql(q1, conn, params=items)
        print(df1.to_string())
        
        # 2. History Sum
        print("\n--- History IV30300 (> 9/30) ---")
        q2 = f"SELECT ITEMNMBR, SUM(TRXQTY) as TrxSum FROM IV30300 WHERE ITEMNMBR IN ({','.join(['?']*len(items))}) AND DOCDATE > ? AND TRXLOCTN='MAIN' GROUP BY ITEMNMBR"
        params2 = items + [cutoff] # Note: SQL IN clause parameters come first
        # Wait, the params order depends on the query string.
        # "WHERE ITEMNMBR IN (?,?,?,?,?) AND DOCDATE > ?"
        # So items first, then date.
        df2 = pd.read_sql(q2, conn, params=items + [cutoff])
        print(df2.to_string())
        
        conn.close()
        
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    spot_check()
