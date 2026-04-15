
import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string
from constants import INVENTORY_GL_CODES, INVENTORY_EXCLUDED_ITEMS

def verify_inventory():
    target_date = date(2025, 9, 30)
    print(f"Verifying inventory for date: {target_date}")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Base Item Info + Current Qty (MAIN Location Only)
        query_base = f"""
        SELECT 
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.CURRCOST,
            T2.QTYONHND as CurrentQty,
            GL.ACTNUMST as GLCode
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
          AND T1.ITEMNMBR NOT IN {tuple(INVENTORY_EXCLUDED_ITEMS)}
          AND RTRIM(GL.ACTNUMST) IN {tuple(INVENTORY_GL_CODES)}
        """
        print("Executing Base Query...")
        base_df = pd.read_sql(query_base, conn)
        print(f"Base rows: {len(base_df)}")
        
        # 2. Get Inventory Changes AFTER target date
        query_history = """
        SELECT 
            ITEMNMBR,
            SUM(TRXQTY) as QtyChange
        FROM IV30300
        WHERE DOCDATE > ? 
          AND TRXLOCTN = 'MAIN'
        GROUP BY ITEMNMBR
        """
        print("Executing History Query...")
        history_df = pd.read_sql(query_history, conn, params=[target_date])
        
        # 3. Get Last Cost AS OF target date
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
        print("Executing Cost Query...")
        cost_df = pd.read_sql(query_cost, conn, params=[target_date])
        
        conn.close()
        
        if base_df.empty:
            print("No base data found.")
            return

        # Merge and Calculate
        df = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df = pd.merge(df, cost_df, on='ITEMNMBR', how='left')
        df['QtyChange'] = df['QtyChange'].fillna(0)
        
        # As Of Qty
        df['Quantity'] = df['CurrentQty'] - df['QtyChange']
        
        # Filter zero quantities
        df = df[df['Quantity'] != 0]
        
        # Extended Cost
        df['Unit Cost'] = df['LastCost'].fillna(df['CURRCOST']).round(5)
        df['Extended Cost'] = (df['Quantity'] * df['Unit Cost']).round(2)
        
        # Formatting
        df_final = df[[
            'ITEMNMBR', 
            'ITEMDESC', 
            'Quantity', 
            'Unit Cost', 
            'Extended Cost', 
            'GLCode'
        ]].copy()
        
        df_final.columns = ['Item Number', 'Description', 'Quantity', 'Unit Cost', 'Extended Cost', 'GL Code']
        df_final = df_final.sort_values('Item Number')

        # Metrics
        total_val = df_final['Extended Cost'].sum()
        total_val_fmt = f"${total_val:,.2f}"
        items_count = len(df_final)
        
        print("\n" + "="*40)
        print("REPORT SUMMARY")
        print("="*40)
        print(f"Total Value: {total_val_fmt}")
        print(f"Items Count: {items_count}")
        print("="*40)
        
        print("\nTop 5 Items by Extended Cost:")
        print(df_final.nlargest(5, 'Extended Cost').to_string())

        # Specific Check for NPKPHOSDRY
        print("\nChecking NPKPHOSDRY:")
        item_check = df_final[df_final['Item Number'] == 'NPKPHOSDRY']
        if not item_check.empty:
            print(item_check.to_string())
        else:
            print("NPKPHOSDRY not found in final list (might have 0 quantity)")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_inventory()
