import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def compare_inventory():
    print("--- INVENTORY COMPARISON START ---")
    
    # 1. READ EXCEL (Physical Counts)
    print("Reading Excel file...")
    try:
        # Header appears to be on row 1 (0-indexed) based on previous inspection
        excel_df = pd.read_excel(
            "c:/Users/alexh/Downloads/mod/10-01-25 Finished Goods and Rawmaterial Count.xls", 
            header=1
        )
        
        # Clean columns
        excel_df.columns = [str(c).strip() for c in excel_df.columns]
        
        # Select key columns
        # We assume 'Count' is the verified physical count
        excel_data = excel_df[['Item Number', 'Count']].copy()
        excel_data['Item Number'] = excel_data['Item Number'].astype(str).str.strip()
        excel_data['PhysicalCount'] = pd.to_numeric(excel_data['Count'], errors='coerce').fillna(0)
        
        # Filter out empty item numbers
        excel_data = excel_data[excel_data['Item Number'].notna() & (excel_data['Item Number'] != '')]
        
        # Group duplicates if any (summing counts)
        excel_data = excel_data.groupby('Item Number')['PhysicalCount'].sum().reset_index()
        
        print(f"Loaded {len(excel_data)} items from Excel.")
        
    except Exception as e:
        print(f"FAILED to read Excel: {e}")
        return

    # 2. CALCULATE DB INVENTORY (System Quantity as of 2025-10-01)
    target_date = date(2025, 10, 1)
    print(f"Calculating Database Inventory as of {target_date}...")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Base Qty (Current)
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
        
        # History (Back out changes to get to target date)
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
        
        # Historical Cost
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
        
        # Merge DB Data
        df_db = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df_db = pd.merge(df_db, cost_df, on='ITEMNMBR', how='left')
        df_db['QtyChange'] = df_db['QtyChange'].fillna(0)
        df_db['SystemQty'] = df_db['CurrentQty'] - df_db['QtyChange']
        df_db['UnitCost'] = df_db['LastCost'].fillna(df_db['CURRCOST'])
        
        df_db['Item Number'] = df_db['ITEMNMBR'].astype(str).str.strip()
        
    except Exception as e:
        print(f"FAILED to query DB: {e}")
        return

    # 3. COMPARE
    print("Comparing...")
    
    # Merge System (Left) with Physical (Right)
    # We use outer join to catch items in Excel not in DB, and vice versa
    merged = pd.merge(df_db, excel_data, on='Item Number', how='outer')
    
    merged['SystemQty'] = merged['SystemQty'].fillna(0)
    merged['PhysicalCount'] = merged['PhysicalCount'].fillna(0)
    merged['UnitCost'] = merged['UnitCost'].fillna(0)
    
    # Variance = System - Physical
    # (Positive means System thinks we have more than we physically counted -> Inventory Shrink/Loss)
    # (Negative means we have more physically -> Inventory Gain)
    merged['QtyVariance'] = merged['SystemQty'] - merged['PhysicalCount']
    merged['ValueVariance'] = merged['QtyVariance'] * merged['UnitCost']
    
    # Check for NaN Item Desc for Excel-only items
    merged['ITEMDESC'] = merged['ITEMDESC'].fillna("Unknown (In Excel only)")
    
    # 4. REPORTING
    total_variance_value = merged['ValueVariance'].sum()
    print(f"\nTOTAL NET VARIANCE (System - Physical): ${total_variance_value:,.2f}")
    if total_variance_value > 0:
         print("(Positive = Inventory Missing/Shrink)")
    else:
         print("(Negative = Inventory Gain)")

    print("\n--- TOP 10 VARIANCE DRIVERS (Absolute Value) ---")
    merged['AbsValueVariance'] = merged['ValueVariance'].abs()
    top_drivers = merged.sort_values('AbsValueVariance', ascending=False).head(15)
    
    print(top_drivers[[
        'Item Number', 'ITEMDESC', 
        'SystemQty', 'PhysicalCount', 'QtyVariance', 
        'UnitCost', 'ValueVariance'
    ]].to_string())

    # Check that BIAMZ02 item specifically
    print("\n--- SPECIFIC CHECK: BIAMZ02 ---")
    biamz = merged[merged['Item Number'] == 'BIAMZ02']
    if not biamz.empty:
        print(biamz[['Item Number', 'ITEMDESC', 'SystemQty', 'PhysicalCount', 'QtyVariance', 'UnitCost', 'ValueVariance']].to_string())
    else:
        print("BIAMZ02 not found in merged data.")

if __name__ == "__main__":
    compare_inventory()
