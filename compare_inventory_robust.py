import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def compare_inventory_robust():
    output = []
    def log(msg):
        print(msg)
        output.append(str(msg))

    log("--- INVENTORY COMPARISON (ROBUST) ---")
    
    # 1. READ EXCEL
    try:
        excel_df = pd.read_excel(
            "c:/Users/alexh/Downloads/mod/10-01-25 Finished Goods and Rawmaterial Count.xls", 
            header=1
        )
        
        # Access by name search
        col_names = excel_df.columns.tolist()
        item_col = next((c for c in col_names if "Item Number" in str(c)), None)
        count_col = next((c for c in col_names if "Count" in str(c) and "Tag" not in str(c)), None)
        
        if not item_col or not count_col:
             excel_df.rename(columns={excel_df.columns[1]: 'Item Number', excel_df.columns[4]: 'Count'}, inplace=True)
             item_col = 'Item Number'
             count_col = 'Count'
        
        excel_data = excel_df[[item_col, count_col]].copy()
        excel_data.columns = ['Item Number', 'PhysicalCount']
        excel_data['Item Number'] = excel_data['Item Number'].astype(str).str.strip()
        excel_data['PhysicalCount'] = pd.to_numeric(excel_data['PhysicalCount'], errors='coerce').fillna(0)
        excel_data = excel_data.groupby('Item Number')['PhysicalCount'].sum().reset_index()
        
    except Exception as e:
        log(f"FAILED to read Excel: {e}")
        return

    # 2. CALCULATE DB INVENTORY
    target_date = date(2025, 10, 1)
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Base
        base_df = pd.read_sql("""
        SELECT T1.ITEMNMBR, T1.ITEMDESC, T1.CURRCOST, T2.QTYONHND as CurrentQty
        FROM IV00101 T1 JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        WHERE T2.LOCNCODE = 'MAIN'
        """, conn)
        
        # History
        history_df = pd.read_sql("""
        SELECT ITEMNMBR, SUM(TRXQTY) as QtyChange
        FROM IV30300 WHERE DOCDATE > ? AND TRXLOCTN = 'MAIN' GROUP BY ITEMNMBR
        """, conn, params=[target_date])
        
        # Cost
        cost_df = pd.read_sql("""
        WITH RatedTransactions AS (
            SELECT ITEMNMBR, UNITCOST, DOCDATE,
                ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
            FROM IV30300 WHERE DOCDATE <= ? AND DOCTYPE IN (4, 1) AND UNITCOST > 0
        )
        SELECT ITEMNMBR, UNITCOST as LastCost FROM RatedTransactions WHERE rn = 1
        """, conn, params=[target_date])
        
        conn.close()
        
        # Merge
        df_db = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df_db = pd.merge(df_db, cost_df, on='ITEMNMBR', how='left')
        df_db['QtyChange'] = df_db['QtyChange'].fillna(0)
        df_db['SystemQty'] = df_db['CurrentQty'] - df_db['QtyChange']
        # Cost Logic
        df_db['UnitCost'] = df_db['LastCost'].fillna(df_db['CURRCOST'])
        df_db['Item Number'] = df_db['ITEMNMBR'].astype(str).str.strip()
        
    except Exception as e:
        log(f"FAILED to query DB: {e}")
        return

    # 3. COMPARE
    merged = pd.merge(df_db, excel_data, on='Item Number', how='outer')
    merged['SystemQty'] = merged['SystemQty'].fillna(0)
    merged['PhysicalCount'] = merged['PhysicalCount'].fillna(0)
    merged['UnitCost'] = merged['UnitCost'].fillna(0)
    
    merged['QtyVariance'] = merged['SystemQty'] - merged['PhysicalCount']
    merged['ValueVariance'] = merged['QtyVariance'] * merged['UnitCost']
    merged['ITEMDESC'] = merged['ITEMDESC'].fillna("Unknown (In Excel only)")
    
    total_val = merged['ValueVariance'].sum()
    log(f"TOTAL NET VARIANCE (System - Physical): ${total_val:,.2f}")
    
    merged['AbsValueVariance'] = merged['ValueVariance'].abs()
    top_drivers = merged.sort_values('AbsValueVariance', ascending=False).head(20)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    log("\n--- TOP 20 VARIANCE DRIVERS ---")
    log(top_drivers[[
        'Item Number', 'ITEMDESC', 
        'SystemQty', 'PhysicalCount', 'QtyVariance', 
        'UnitCost', 'ValueVariance'
    ]].to_string())
    
    with open('variance_report.txt', 'w') as f:
        for line in output:
            f.write(line + "\n")

if __name__ == "__main__":
    compare_inventory_robust()
