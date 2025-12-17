
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def check_inventory_integration():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        receipts = ['RV0000038118', 'RV0000038144']
        print(f"--- Checking Inventory (IV30300) for Receipts: {receipts} ---")
        
        placeholders = ','.join(['?'] * len(receipts))
        
        # Check IV30300
        query_iv = f"""
        SELECT 
            DOCNUMBR,
            DOCDATE,
            ITEMNMBR,
            TRXQTY
        FROM IV30300
        WHERE DOCNUMBR IN ({placeholders})
        """
        df_iv = pd.read_sql(query_iv, conn, params=receipts)
        print("IV30300 Results:")
        print(df_iv.to_string())
        
        print(f"\n--- Checking POP30310 Flags ---")
        query_pop = f"""
        SELECT 
            POPRCTNM,
            ITEMNMBR,
            NONINVEN,
            LOCNCODE,
            ACTLSHIP,
            EXTDCOST
        FROM POP30310
        WHERE POPRCTNM IN ({placeholders})
        """
        df_pop = pd.read_sql(query_pop, conn, params=receipts)
        print("POP30310 Flags:")
        print(df_pop.to_string())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_inventory_integration()
