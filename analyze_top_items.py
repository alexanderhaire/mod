
import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def analyze_top_items():
    target_date = date(2025, 9, 30)
    print(f"Analyzing top items for date: {target_date}")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Base + Filter for the 5 GL Codes
        query_base = """
        SELECT 
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.CURRCOST,
            T2.QTYONHND as CurrentQty,
            RTRIM(GL.ACTNUMST) as GLCode
        FROM IV00101 T1
        JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        LEFT JOIN IV40400 ClassAccts ON T1.ITMCLSCD = ClassAccts.ITMCLSCD
        LEFT JOIN GL00105 GL ON GL.ACTINDX = (
            CASE 
                WHEN T1.IVIVINDX > 0 THEN T1.IVIVINDX 
                ELSE ClassAccts.IVIVINDX 
            END
        )
        WHERE T2.LOCNCODE = 'MAIN'
          AND RTRIM(GL.ACTNUMST) IN ('12100-00-00', '12200-00-00', '12250-00-00', '12300-00-00', '12700-00-00')
        """
        base_df = pd.read_sql(query_base, conn)
        
        # 2. History
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
        
        # 3. Cost
        query_cost = """
        WITH RankedCosts AS (
            SELECT 
                ITEMNMBR,
                UNITCOST,
                ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
            FROM IV30300
            WHERE DOCDATE <= ?
              AND UNITCOST IS NOT NULL
              AND UNITCOST <> 0
        )
        SELECT ITEMNMBR, UNITCOST as LastCost
        FROM RankedCosts
        WHERE rn = 1
        """
        cost_df = pd.read_sql(query_cost, conn, params=[target_date])
        
        conn.close()
        
        # Merge
        df = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df = pd.merge(df, cost_df, on='ITEMNMBR', how='left')
        df['QtyChange'] = df['QtyChange'].fillna(0)
        df['Quantity'] = df['CurrentQty'] - df['QtyChange']
        
        df = df[df['Quantity'] != 0]
        
        df['Unit Cost'] = df['LastCost'].fillna(df['CURRCOST'])
        df['Extended Cost'] = df['Quantity'] * df['Unit Cost']
        
        # Sort and Top 20
        df_sorted = df.sort_values('Extended Cost', ascending=False)
        
        print("\nTop 20 Items by Value:")
        print(df_sorted[['ITEMNMBR', 'GLCode', 'Quantity', 'Unit Cost', 'Extended Cost']].head(20).to_string())
        
        print(f"\nTotal Items: {len(df)}")
        print(f"Total Value: ${df['Extended Cost'].sum():,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_top_items()
