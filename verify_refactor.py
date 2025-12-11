import pyodbc
from constants import RAW_MATERIAL_CLASS_CODES
from secrets_loader import build_connection_string
from market_insights import get_raw_material_time_series
import pandas as pd

def verify_refactor():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        rm_class_list = "', '".join(RAW_MATERIAL_CLASS_CODES)
        print(f"Fetching Raw Material Time Series (Classes: {rm_class_list})...")
        data = get_raw_material_time_series(cursor)
        
        # data is a dict with 'monthly_volume' DataFrame etc.
        # "monthly_cost" implies aggregation.
        # To verify items included, we need `data['cost_index']` as it lists specific items,
        # OR we can inspect the internal query data if we could, but we can't easily.
        
        # Wait, get_raw_material_time_series processes the data into aggregates.
        # However, `cost_index` dataframe has an 'Item' column.
        
        idx_df = data.get('cost_index', pd.DataFrame())
        
        if idx_df.empty:
            print("No data returned in cost_index.")
            # This might happen if no top items found.
            
            # Let's inspect "Monthly Volume" - but that is aggregated.
            # I should probably just run the query that get_raw_material_time_series uses directly to test.
            pass
        else:
             print("Items in Cost Index (Top 5):", idx_df['Item'].unique())

        # Let's run a check query manually to see what WOULD be returned by the new logic
        query = f"""
        SELECT DISTINCT i.ITEMNMBR, i.ITMCLSCD
        FROM IV00101 i
        WHERE i.ITMCLSCD IN ('{rm_class_list}')
          AND i.ITEMNMBR IN ('NO3CA', 'AGBAPMG02')
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        print("\nDirect SQL Verification (Should only show NO3CA):")
        for r in rows:
            print(f"{r.ITEMNMBR} - {r.ITMCLSCD}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_refactor()
