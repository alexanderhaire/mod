import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string
from constants import INVENTORY_GL_CODES, INVENTORY_EXCLUDED_ITEMS
import csv

def generate_adjustments_csv():
    print("--- GENERATING YEAR END ADJUSTMENTS CSV ---")
    
    # 1. READ EXCEL (Physical Counts)
    print("Reading Excel file...")
    try:
        excel_df = pd.read_excel(
            "c:/Users/alexh/Downloads/mod/10-01-25 Finished Goods and Rawmaterial Count.xls", 
            header=1
        )
        
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
        print(f"FAILED to read Excel: {e}")
        return

    # 2. CALCULATE DB INVENTORY
    target_date = date(2025, 10, 1)
    print(f"Calculating Database Inventory as of {target_date}...")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Base
        base_df = pd.read_sql(f"""
        SELECT T1.ITEMNMBR, T1.ITEMDESC, T1.CURRCOST, T2.QTYONHND as CurrentQty, T3.ACTNUMST as GLCode
        FROM IV00101 T1 
        JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        LEFT JOIN GL00105 T3 ON T1.IVIVINDX = T3.ACTINDX
        WHERE T2.LOCNCODE = 'MAIN'
          AND RTRIM(T3.ACTNUMST) IN {tuple(INVENTORY_GL_CODES)}
        """, conn)
        
        # History
        history_df = pd.read_sql("""
        SELECT ITEMNMBR, SUM(TRXQTY) as QtyChange
        FROM IV30300 WHERE DOCDATE > ? AND TRXLOCTN = 'MAIN' GROUP BY ITEMNMBR
        """, conn, params=[target_date])
        
        conn.close()
        
        # Merge
        df_db = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
        df_db['QtyChange'] = df_db['QtyChange'].fillna(0)
        df_db['SystemQty'] = df_db['CurrentQty'] - df_db['QtyChange']
        df_db['Item Number'] = df_db['ITEMNMBR'].astype(str).str.strip()
        
    except Exception as e:
        print(f"FAILED to query DB: {e}")
        return

    # 3. COMPARE AND GENERATE ADJUSTMENTS
    merged = pd.merge(df_db, excel_data, on='Item Number', how='outer')
    merged['SystemQty'] = merged['SystemQty'].fillna(0)
    merged['PhysicalCount'] = merged['PhysicalCount'].fillna(0)
    merged['CURRCOST'] = merged['CURRCOST'].fillna(0)
    
    # Calculate Adjustment: We want System + Adj = Physical
    # Adj = Physical - System
    merged['AdjustmentQty'] = merged['PhysicalCount'] - merged['SystemQty']
    
    # Filter for non-zero adjustments
    adjustments = merged[merged['AdjustmentQty'] != 0].copy()
    
    # Filter out Excluded Items (Non-inventory, Consignment, Obsolete)
    print(f"Excluding items: {INVENTORY_EXCLUDED_ITEMS}...")
    adjustments = adjustments[~adjustments['Item Number'].isin(INVENTORY_EXCLUDED_ITEMS)]
    
    print(f"Found {len(adjustments)} items requiring adjustment.")
    
    # Prepare CSV Data
    # Format: Item Number, TRX Date, Adjustment Qty, Unit Cost, Batch ID, Reason Code
    csv_data = adjustments[['Item Number', 'AdjustmentQty', 'CURRCOST']].copy()
    csv_data['TRX Date'] = target_date.strftime('%m/%d/%Y')
    csv_data['Batch ID'] = 'YE_CLOSE_2025'
    csv_data['Reason Code'] = 'YE COUNT'
    csv_data['Location Code'] = 'MAIN'
    
    # Reorder
    final_output = csv_data[['Batch ID', 'Item Number', 'Location Code', 'TRX Date', 'AdjustmentQty', 'CURRCOST', 'Reason Code']]
    final_output.columns = ['Batch ID', 'Item Number', 'Site ID', 'Trx Date', 'Trx Qty', 'Unit Cost', 'Reason Code']
    
    output_file = "year_end_adjustments.csv"
    final_output.to_csv(output_file, index=False)
    
    print(f"Successfully generated {output_file}")
    print("Top 5 Adjustments by Cost Impact:")
    
    adjustments['Impact'] = adjustments['AdjustmentQty'] * adjustments['CURRCOST']
    adjustments['AbsImpact'] = adjustments['Impact'].abs()
    print(adjustments.sort_values('AbsImpact', ascending=False).head(5)[
        ['Item Number', 'SystemQty', 'PhysicalCount', 'AdjustmentQty', 'CURRCOST', 'Impact']
    ].to_string())

if __name__ == "__main__":
    generate_adjustments_csv()
