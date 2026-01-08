import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
import os

def export_to_excel():
    print("Connecting to database...")
    conn_str, _, _, _ = build_connection_string()
    
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Nitrogen Items
        item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID = 'NIT&TON'"
        items_df = pd.read_sql(item_query, conn)
        
        if items_df.empty:
            item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID LIKE '%NIT%'"
            items_df = pd.read_sql(item_query, conn)
            
        nitrogen_items = tuple(items_df['ITEMNMBR'].tolist())
        if len(nitrogen_items) == 1:
            nitrogen_items = f"('{nitrogen_items[0]}')"

        # 2. Detailed Sales Data (With Date)
        print("Fetching detailed sales data...")
        detail_query = f"""
        SELECT 
            h.DOCDATE,
            l.ITEMNMBR,
            MAX(l.ITEMDESC) as Description,
            SUM(l.XTNDPRCE) as TotalSales,
            SUM(l.QUANTITY) as TotalQty
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
        WHERE h.DOCDATE BETWEEN '2025-10-01' AND '2025-12-31'
          AND l.ITEMNMBR IN {nitrogen_items}
          AND h.SOPTYPE = 3 
          AND h.VOIDSTTS = 0
        GROUP BY h.DOCDATE, l.ITEMNMBR
        ORDER BY h.DOCDATE
        """
        detail_df = pd.read_sql(detail_query, conn)
        
        # 3. Summary Data (By Item)
        print("Fetching summary sales data...")
        summary_query = f"""
        SELECT 
            l.ITEMNMBR,
            MAX(l.ITEMDESC) as Description,
            SUM(l.XTNDPRCE) as TotalSales,
            SUM(l.QUANTITY) as TotalQty
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
        WHERE h.DOCDATE BETWEEN '2025-10-01' AND '2025-12-31'
          AND l.ITEMNMBR IN {nitrogen_items}
          AND h.SOPTYPE = 3 
          AND h.VOIDSTTS = 0
        GROUP BY l.ITEMNMBR
        ORDER BY TotalSales DESC
        """
        summary_df = pd.read_sql(summary_query, conn)
        
        conn.close()
        
        # 4. Write to Excel
        artifact_dir = r"C:\Users\alexh\.gemini\antigravity\brain\bf64b684-56da-46d9-9ef7-e86c9436f449"
        output_file = os.path.join(artifact_dir, "Nitrogen_Sales_Q4_2025.xlsx")
        
        print(f"Writing to {output_file}...")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary by Item', index=False)
            detail_df.to_excel(writer, sheet_name='Detailed Transactions', index=False)
            
        print("Export complete!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    export_to_excel()
