import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
import os

def export_nitrogen_sales():
    print("Connecting to database...")
    conn_str, _, _, _ = build_connection_string()
    
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Items
        item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID = 'NIT&TON'"
        items_df = pd.read_sql(item_query, conn)
        
        if items_df.empty:
            item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID LIKE '%NIT%'"
            items_df = pd.read_sql(item_query, conn)
            
        print(f"Found {len(items_df)} items with Nitrogen flag.")
        
        if items_df.empty:
            return

        nitrogen_items = tuple(items_df['ITEMNMBR'].tolist())
        if len(nitrogen_items) == 1:
            nitrogen_items = f"('{nitrogen_items[0]}')"
        
        # 2. Get Sales History
        sales_query = f"""
        SELECT 
            l.ITEMNMBR,
            MAX(l.ITEMDESC) as Description,
            SUM(l.XTNDPRCE) as TotalSales,
            SUM(l.QUANTITY) as TotalQty
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
        WHERE h.DOCDATE BETWEEN '2025-10-01' AND '2025-12-31'
          AND l.ITEMNMBR IN {nitrogen_items}
          AND h.SOPTYPE = 3 -- Invoices only
          AND h.VOIDSTTS = 0
        GROUP BY l.ITEMNMBR
        ORDER BY TotalSales DESC
        """
        
        sales_df = pd.read_sql(sales_query, conn)
        conn.close()
        
        if sales_df.empty:
            print("No sales data found.")
        else:
            artifact_dir = r"C:\Users\alexh\.gemini\antigravity\brain\bf64b684-56da-46d9-9ef7-e86c9436f449"
            output_csv = os.path.join(artifact_dir, "nitrogen_sales_full_list.csv")
            sales_df.to_csv(output_csv, index=False)
            print(f"Exported {len(sales_df)} rows to {output_csv}")
            
            # Summary stats
            print(f"Total Sales: ${sales_df['TotalSales'].sum():,.2f}")
            print(f"Total Items with Sales: {len(sales_df)}")
            print(f"Total Nitrogen Items (in Master): {len(items_df)}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    export_nitrogen_sales()
