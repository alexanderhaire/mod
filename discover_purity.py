import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def discover_purity_data():
    print("--- DIAGNOSTIC: Purity Discovery (Fixed) ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Fetch items with "%" in description
        print("Fetching items with '%' or 'Grade' in description...")
        query = """
        SELECT TOP 100 ITEMNMBR, ITEMDESC, CURRCOST, STNDCOST, PRCHSUOM
        FROM IV00101 
        WHERE ITEMDESC LIKE '%[%]%' 
           OR ITEMDESC LIKE '%Grade%'
           OR ITEMDESC LIKE '%Purity%'
        ORDER BY ITEMDESC
        """
        items = pd.read_sql(query, conn)
        
        if items.empty:
            print("No items found with '%', 'Grade', or 'Purity' in description.")
        else:
            print(f"Found {len(items)} items. Examples:")
            print(items[['ITEMNMBR', 'ITEMDESC', 'CURRCOST']].head(10).to_string())
            
            # Parsing logic
            print("\n--- Purity Parsing Analysis ---")
            
            def extract_purity(desc):
                # pattern: number followed by optional space and %
                # e.g. "99%", "99.5 %", "50%"
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', desc)
                if match:
                    return float(match.group(1))
                return None

            items['ParsedPurity'] = items['ITEMDESC'].apply(extract_purity)
            
            # Filter for rows where we successfully parsed a purity
            # and ignore small percentages (likely not purity, asking for chemical purity usually implies > 5%)
            parsed = items[items['ParsedPurity'].notna() & (items['ParsedPurity'] > 5)].copy()
            
            if not parsed.empty:
                # Calculate Price per 1% Purity
                # Avoid division by zero
                parsed['CostPerPct'] = parsed.apply(
                    lambda x: x['CURRCOST'] / x['ParsedPurity'] if x['ParsedPurity'] > 0 and x['CURRCOST'] > 0 else 0, 
                    axis=1
                )
                
                print(f"\nSuccessfully parsed {len(parsed)} items with purity.")
                print(parsed[['ITEMDESC', 'ParsedPurity', 'CURRCOST', 'CostPerPct']].sort_values('CostPerPct').head(20).to_string())
            else:
                print("Could not parse valid purity percentages from the found items.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    discover_purity_data()
