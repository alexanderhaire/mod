import pyodbc
from secrets_loader import build_connection_string
import pandas as pd

def check_item():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        item = 'CHEACETIC'
        
        # Check IV00101
        print(f"Checking {item} in IV00101...")
        df = pd.read_sql("SELECT * FROM IV00101 WHERE ITEMNMBR = ?", conn, params=[item])
        if not df.empty:
            print("Found in IV00101!")
            print(df.iloc[0])
        else:
            print("NOT Found in IV00101.")
            
            # Fuzzy search
            print("Searching for similar...")
            df_like = pd.read_sql("SELECT ITEMNMBR FROM IV00101 WHERE ITEMNMBR LIKE ?", conn, params=['%CHE%'])
            print("Items matching %CHE%:")
            print(df_like.head(10))

        # Check IV00102 (Any Location)
        print(f"\nChecking {item} using IV00102 (All Locations)...")
        df_loc = pd.read_sql("SELECT LOCNCODE, QTYONHND FROM IV00102 WHERE ITEMNMBR = ?", conn, params=[item])
        print(df_loc)

        print(f"\nChecking {item} in IV30300 (Any Location)...")
        df_trx = pd.read_sql("SELECT TOP 5 TRXLOCTN, TRXQTY, DOCDATE FROM IV30300 WHERE ITEMNMBR = ? ORDER BY DOCDATE DESC", conn, params=[item])
        print(df_trx)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_item()
