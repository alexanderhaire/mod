import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def debug_variance():
    target_date = date(2025, 9, 30)
    print(f"--- DIAGNOSTIC RUN: Inventory As Of {target_date} ---")

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Base Item Info + Current Qty (MAIN Location Only)
        print("Fetching Base Inventory...")
        query_base = """
        SELECT 
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.CURRCOST,
            T2.QTYONHND as CurrentQty
        FROM IV00101 T1
        JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        WHERE T2.LOCNCODE = 'MAIN'
        """
        base_df = pd.read_sql(query_base, conn)
        
        # 2. Get Inventory Changes AFTER target date
        print("Fetching Inventory History...")
        query_history = """
        SELECT 
            ITEMNMBR,
            SUM(TRXQTY) as QtyChange
        FROM IV30300
        WHERE DOCDATE > ? 
            AND TRXLOCTN = 'MAIN'
        GROUP BY ITEMNMBR
        """
        history_df = pd.read_sql(query_history, conn, params=[target_date])
        
        # 3. Get Last Cost AS OF target date
        print("Fetching Historical Costs...")
        query_cost = """
        WITH RatedTransactions AS (
            SELECT 
                ITEMNMBR,
                UNITCOST,
                DOCDATE,
                ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
            FROM IV30300
            WHERE DOCDATE <= ?
                AND DOCTYPE IN (4, 1)
                AND UNITCOST > 0
        )
        SELECT ITEMNMBR, UNITCOST as LastCost
        FROM RatedTransactions
        WHERE rn = 1
        """
        cost_df = pd.read_sql(query_cost, conn, params=[target_date])
        
        conn.close()

        # Merge
        df = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df = pd.merge(df, cost_df, on='ITEMNMBR', how='left')
        df['QtyChange'] = df['QtyChange'].fillna(0)
        
        # Calculate As Of Qty
        df['Quantity'] = df['CurrentQty'] - df['QtyChange']
        
        # Calculate Valuation
        df['Unit Cost'] = df['LastCost'].fillna(df['CURRCOST'])
        df['Extended Cost'] = df['Quantity'] * df['Unit Cost']

        print(f"\nTotal Items Processed: {len(df)}")
        print(f"Total Valuation: ${df['Extended Cost'].sum():,.2f}")

        # --- DIAGNOSTICS ---

        print("\n--- CHECK 1: NEGATIVE QUANTITIES (Reduces Total Value) ---")
        negatives = df[df['Quantity'] < 0].sort_values('Extended Cost')
        if not negatives.empty:
            print(negatives[['ITEMNMBR', 'ITEMDESC', 'Quantity', 'Unit Cost', 'Extended Cost']].to_string())
            print(f"Total Negative Impact: ${negatives['Extended Cost'].sum():,.2f}")
        else:
            print("No negative quantities found.")

        print("\n--- CHECK 2: TOP 10 ITEMS BY VALUE ---")
        top_items = df.sort_values('Extended Cost', ascending=False).head(10)
        print(top_items[['ITEMNMBR', 'ITEMDESC', 'Quantity', 'Unit Cost', 'Extended Cost']].to_string())

        print("\n--- CHECK 3: COST ANOMALIES (LastCost vs CurrCost > 50% diff) ---")
        # Filter where we have both costs and they differ significantly
        df['CostDiffPct'] = abs(df['LastCost'] - df['CURRCOST']) / df['CURRCOST']
        anomalies = df[ (df['LastCost'].notna()) & (df['CURRCOST'] > 0) & (df['CostDiffPct'] > 0.5) ]
        
        if not anomalies.empty:
            anomalies = anomalies.sort_values('Extended Cost', ascending=False)
            print(anomalies[['ITEMNMBR', 'ITEMDESC', 'CURRCOST', 'LastCost', 'CostDiffPct']].head(10).to_string())
        else:
            print("No significant cost anomalies found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_variance()
