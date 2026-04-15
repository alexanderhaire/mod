import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def check_930_values():
    target_items = ['ZZ2.5GALF', 'GOLDFE02', 'ZZ55GAL', 'NPKKTS', 'ZZ30GAL']
    cutoff_date = date(2025, 9, 30)
    
    print(f"--- CHECKING QUANTITIES AS OF {cutoff_date} ---")
    
    # 1. READ PHYSICAL COUNT (EXCEL)
    # Using the same logic as generate_adjustments.py to get the "Count"
    excel_path = "c:/Users/alexh/Downloads/mod/10-01-25 Finished Goods and Rawmaterial Count.xls"
    try:
        excel_df = pd.read_excel(excel_path, header=1)
        col_names = excel_df.columns.tolist()
        item_col = next((c for c in col_names if "Item Number" in str(c)), None)
        count_col = next((c for c in col_names if "Count" in str(c) and "Tag" not in str(c)), None)
        
        if not item_col or not count_col:
             # Fallback
             item_col = excel_df.columns[1]
             count_col = excel_df.columns[4]
        
        excel_data = excel_df[[item_col, count_col]].copy()
        excel_data.columns = ['Item Number', 'PhysicalCount']
        excel_data['Item Number'] = excel_data['Item Number'].astype(str).str.strip()
        excel_data['PhysicalCount'] = pd.to_numeric(excel_data['PhysicalCount'], errors='coerce').fillna(0)
        
        # Filter to target items
        physical_counts = excel_data[excel_data['Item Number'].isin(target_items)].groupby('Item Number')['PhysicalCount'].sum()
    except Exception as e:
        print(f"Error reading Excel: {e}")
        physical_counts = pd.Series()

    # 2. CALCULATE SYSTEM QUANTITY
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Get Current Qty
        q_current = f"""
        SELECT ITEMNMBR, QTYONHND 
        FROM IV00102 
        WHERE LOCNCODE = 'MAIN' 
        AND ITEMNMBR IN ({','.join(['?']*len(target_items))})
        """
        current_df = pd.read_sql(q_current, conn, params=target_items)
        
        # Get Transactions > Cutoff
        q_hist = f"""
        SELECT ITEMNMBR, SUM(TRXQTY) as QtyChange
        FROM IV30300 
        WHERE DOCDATE > ? 
        AND TRXLOCTN = 'MAIN'
        AND ITEMNMBR IN ({','.join(['?']*len(target_items))})
        GROUP BY ITEMNMBR
        """
        params = [cutoff_date] + target_items
        hist_df = pd.read_sql(q_hist, conn, params=params)
        
        conn.close()
        
        # Merge
        current_df['ITEMNMBR'] = current_df['ITEMNMBR'].astype(str).str.strip()
        hist_df['ITEMNMBR'] = hist_df['ITEMNMBR'].astype(str).str.strip()
        
        merged = pd.merge(current_df, hist_df, on='ITEMNMBR', how='left')
        merged['QtyChange'] = merged['QtyChange'].fillna(0)
        merged['SystemQty_930'] = merged['QTYONHND'] - merged['QtyChange']
        
        system_map = merged.set_index('ITEMNMBR')['SystemQty_930']
        
    except Exception as e:
        print(f"Error querying DB: {e}")
        system_map = pd.Series()

    # 3. PRINT REPORT
    print(f"\n{'Item Number':<15} | {'System Qty (9/30)':<18} | {'Physical Count (Excel)':<22} | {'Difference':<12}")
    print("-" * 75)
    
    for item in target_items:
        sys_val = system_map.get(item, 0.0)
        phys_val = physical_counts.get(item, 0.0)
        diff = phys_val - sys_val
        print(f"{item:<15} | {sys_val:<18.5f} | {phys_val:<22.5f} | {diff:<12.5f}")

if __name__ == "__main__":
    check_930_values()
