
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
from datetime import datetime, timedelta

def get_usage():
    conn_str, server, db, auth = build_connection_string()
    try:
        with pyodbc.connect(conn_str, autocommit=True) as conn:
            
            # 1. Find relevant Item Numbers
            keywords = [
                'humic', 'fulvic', 'seaweed', 'amino', 'eddha', 'potassium fulvate'
            ]
            
            like_clauses = " OR ".join([f"ITEMDESC LIKE '%{k}%'" for k in keywords])
            
            find_items_query = f"""
            SELECT ITEMNMBR, ITEMDESC
            FROM IV00101
            WHERE {like_clauses}
            """
            
            df_items = pd.read_sql(find_items_query, conn)
            
            if df_items.empty:
                print("No items found matching keywords.")
                return

            item_map = dict(zip(df_items['ITEMNMBR'], df_items['ITEMDESC']))
            item_list = "', '".join(df_items['ITEMNMBR'].tolist())
            
            # 2. Query Usage (Negative TRXQTY in IV30300) for last 365 days
            # We filter for DOCTYPE that represents usage (Sales, Issues, Transfers?)
            # Usually checking all negative Quantities is a decent proxy for "Consumption"
            
            usage_query = f"""
            SELECT 
                ITEMNMBR, 
                SUM(TRXQTY) * -1 as TotalUsage,
                MAX(UOFM) as UOM
            FROM IV30300
            WHERE 
                ITEMNMBR IN ('{item_list}') AND
                DOCDATE >= DATEADD(day, -365, GETDATE()) AND
                TRXQTY < 0
            GROUP BY ITEMNMBR
            ORDER BY TotalUsage DESC
            """
            
            df_usage = pd.read_sql(usage_query, conn)
            
            if df_usage.empty:
                print("No usage history found for these items in the last 365 days.")
                # Fallback: Print items found even if no usage
                print("\nItems Match but no usage:")
                print(df_items.to_string())
                return

            # Merge descriptions
            df_usage['Description'] = df_usage['ITEMNMBR'].map(item_map)
            
            print("\n--- Estimated Usage (Last 365 Days) ---")
            print(df_usage[['ITEMNMBR', 'Description', 'TotalUsage', 'UOM']].to_string(index=False))
            
            # Summarize by Keyword
            print("\n--- Summary by Category ---")
            category_stats = []
            for k in keywords:
                # Filter items containing this keyword
                relevant_items = df_items[df_items['ITEMDESC'].str.lower().str.contains(k, na=False)]['ITEMNMBR']
                total_cat_usage = df_usage[df_usage['ITEMNMBR'].isin(relevant_items)]['TotalUsage'].sum()
                category_stats.append({'Category': k, 'TotalUsage': total_cat_usage})
            
            df_cat = pd.DataFrame(category_stats).sort_values('TotalUsage', ascending=False)
            print(df_cat.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_usage()
