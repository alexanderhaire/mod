import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_item_types():
    print("--- DIAGNOSTIC: Item Types ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("Fetching Item Type Counts...")
        query = """
        SELECT ITEMTYPE, COUNT(*) as Count
        FROM IV00101
        GROUP BY ITEMTYPE
        """
        counts = pd.read_sql(query, conn)
        print(counts.to_string())
        
        # Check for Kits (Type 3)
        if 3 in counts['ITEMTYPE'].values:
            print("\nFetching Sample Kits...")
            query_kits = """
            SELECT TOP 5 ITEMNMBR, ITEMDESC, CURRCOST 
            FROM IV00101 
            WHERE ITEMTYPE = 3
            """
            kits = pd.read_sql(query_kits, conn)
            print(kits.to_string())
            
            # Check Kit Components (IV00104)
            print("\nChecking IV00104 (Kit Components)...")
            try:
                # Get schema first just in case
                # But I'll blindly try common columns
                query_kit_comps = """
                SELECT TOP 10 
                    ITEMNMBR as KitItem, 
                    CMPTITNM as Component, 
                    CMPONENTQTY as Qty
                FROM IV00104
                """
                kit_comps = pd.read_sql(query_kit_comps, conn)
                print(kit_comps.to_string())
            except Exception as e:
                print(f"Error querying IV00104: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_item_types()
