import pyodbc
from constants import RAW_MATERIAL_CLASS_CODES
from secrets_loader import build_connection_string
from market_insights import get_top_movers_raw_materials
import pandas as pd

def test_rm_logic():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        rm_class_list = "', '".join(RAW_MATERIAL_CLASS_CODES)
        print(f"Fetching Top Movers (Classes: {rm_class_list})...")
        df = get_top_movers_raw_materials(cursor, limit=10)
        
        if not df.empty:
            print(f"Returned {len(df)} items.")
            # Check if ITMCLSCD column exists (it's not in the DF columns by default unless I added it? 
            # Wait, I didn't add ITMCLSCD to the SELECT list of get_top_movers_raw_materials, I only added it to WHERE clause!)
            # But I should double check if the items returned are indeed the ones expected.
            print(df[['ITEMNMBR', 'ITEMDESC', 'CurrentCost', 'PctChange']].to_string())
        else:
            print(f"No items returned. Warning: Check if any items have class codes in: {rm_class_list}.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_rm_logic()
