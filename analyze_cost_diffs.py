
import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def analyze_cost_differences():
    target_date = date(2025, 9, 30)
    items_to_check = ['MISCGLYCEROL', 'SO4BORIC', 'NPKPHOS85', 'NPKU32', 'GRPAG50Y']
    
    print(f"Analyzing items: {items_to_check}")
    print(f"As of Date: {target_date}")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Master Record Info (Current Cost, Standard Cost)
        query_master = f"""
        SELECT 
            ITEMNMBR, 
            ITEMDESC, 
            CURRCOST as Master_CurrCost, 
            STNDCOST as Master_StndCost
        FROM IV00101
        WHERE ITEMNMBR IN ({','.join(['?']*len(items_to_check))})
        """
        master_df = pd.read_sql(query_master, conn, params=items_to_check)
        
        print("\n--- MASTER RECORD (IV00101) ---")
        print(master_df.to_string())

        # 2. Get Transaction History (Last 5 transactions before date)
        print("\n--- TRANSACTION HISTORY (IV30300) <= Target Date ---")
        query_hist = """
        SELECT TOP 30
            ITEMNMBR,
            DOCDATE,
            DOCNUMBR,
            TRXQTY,
            UNITCOST as Trx_UnitCost,
            DEX_ROW_ID
        FROM IV30300
        WHERE ITEMNMBR = ? 
          AND DOCDATE <= ?
          AND UNITCOST > 0
        ORDER BY DOCDATE DESC, DEX_ROW_ID DESC
        """
        
        for item in items_to_check:
            print(f"\nHistory for {item}:")
            hist_df = pd.read_sql(query_hist, conn, params=[item, target_date])
            if not hist_df.empty:
                print(hist_df[['DOCDATE', 'DOCNUMBR', 'TRXQTY', 'Trx_UnitCost']].head(5).to_string())
                print(f"-> Logic picking: {hist_df.iloc[0]['Trx_UnitCost']}")
            else:
                print("No history found.")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_cost_differences()
