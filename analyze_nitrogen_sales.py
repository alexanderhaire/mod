import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from secrets_loader import build_connection_string
import os

def analyze_nitrogen_sales():
    print("Connecting to database...")
    conn_str, _, _, _ = build_connection_string()
    
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Get Items with Nitrogen Flag (using Tax Schedule ID as proxy)
        print("Fetching Nitrogen items...")
        item_query = """
        SELECT ITEMNMBR, ITEMDESC
        FROM IV00101
        WHERE ITMTSHID = 'NIT&TON'
        """
        items_df = pd.read_sql(item_query, conn)
        
        if items_df.empty:
            print("No items found with Tax Schedule ID 'NIT&TON'.")
            # Fallback: Try searching for any schedule containing 'NIT'
            item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID LIKE '%NIT%'"
            items_df = pd.read_sql(item_query, conn)
            
        print(f"Found {len(items_df)} items with Nitrogen flag.")
        
        if items_df.empty:
            return

        nitrogen_items = tuple(items_df['ITEMNMBR'].tolist())
        if len(nitrogen_items) == 1:
            nitrogen_items = f"('{nitrogen_items[0]}')"
        
        # 2. Get Sales History with Dates
        print("Fetching sales history with dates for Q4 2025...")
        
        sales_query = f"""
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
          AND h.SOPTYPE = 3 -- Invoices only
          AND h.VOIDSTTS = 0
        GROUP BY h.DOCDATE, l.ITEMNMBR
        ORDER BY h.DOCDATE
        """
        
        sales_df = pd.read_sql(sales_query, conn)
        conn.close()
        
        if sales_df.empty:
            print("No sales data found.")
            return

        # Ensure DOCDATE is datetime
        sales_df['DOCDATE'] = pd.to_datetime(sales_df['DOCDATE'])
        
        # Expert detailed CSV
        artifact_dir = r"C:\Users\alexh\.gemini\antigravity\brain\bf64b684-56da-46d9-9ef7-e86c9436f449"
        output_csv = os.path.join(artifact_dir, "nitrogen_sales_with_dates.csv")
        sales_df.to_csv(output_csv, index=False)
        print(f"Exported detailed sales data to {output_csv}")

        # 3. Generate Chart - Daily Sales Trend
        print("Generating Daily Sales chart...")
        
        # Aggregate by Date
        daily_sales = sales_df.groupby('DOCDATE')['TotalSales'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(daily_sales['DOCDATE'], daily_sales['TotalSales'], marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)
        
        plt.title('Daily Sales of Nitrogen Items (Q4 2025)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Format y-axis currency
        import matplotlib.ticker as mticker
        fmt = '${x:,.0f}'
        tick = mticker.StrMethodFormatter(fmt)
        plt.gca().yaxis.set_major_formatter(tick) 
        
        plt.tight_layout()
        
        output_file = 'nitrogen_sales_daily_trend.png'
        output_path = os.path.join(artifact_dir, output_file)
        
        plt.savefig(output_path)
        print(f"Daily Sales Chart saved to: {output_path}")
        
        # Optional: Top 5 Items Daily Trend (if not too cluttered)
        # Identify Top 5 Items
        top_items = sales_df.groupby('ITEMNMBR')['TotalSales'].sum().nlargest(5).index.tolist()
        
        plt.figure(figsize=(12, 6))
        for item in top_items:
            item_data = sales_df[sales_df['ITEMNMBR'] == item].groupby('DOCDATE')['TotalSales'].sum().reset_index()
            # Resample to ensure all days are present or just plot points? Plotting points is safer for sparse data.
            plt.plot(item_data['DOCDATE'], item_data['TotalSales'], marker='o', linestyle='-', label=item)
            
        plt.title('Daily Sales Verification - Top 5 Nitrogen Items (Q4 2025)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales ($)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(tick)
        plt.tight_layout()
        
        output_path_2 = os.path.join(artifact_dir, 'nitrogen_sales_top5_daily.png')
        plt.savefig(output_path_2)
        print(f"Top 5 Daily Chart saved to: {output_path_2}")


    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_nitrogen_sales()
