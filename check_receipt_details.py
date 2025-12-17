
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def check_receipt_details():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        receipts = ['RCT1239922', 'RCT1240228']
        
        print(f"--- Checking details for Receipts: {receipts} ---")
        
        placeholders = ','.join(['?'] * len(receipts))
        query = f"""
        SELECT 
            POPRCTNM,
            VNDDOCNM,
            RECEIPTDATE,
            BACHNUMB,
            VOIDSTTS,
            TRXSORCE
        FROM POP30300
        WHERE POPRCTNM IN ({placeholders})
        """
        
        df = pd.read_sql(query, conn, params=receipts)
        print(df.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_receipt_details()
