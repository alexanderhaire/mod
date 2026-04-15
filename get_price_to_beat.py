import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def get_price_to_beat():
    print("--- QUANTITATIVE RESEARCH: Establishing the 'Price to Beat' ---")
    
    targets = [
        '%Potassium Nitrate%', '%Urea Ammonium%', '%Calcium Nitrate%', 
        '%Phosphoric Acid%', '%Zinc Nitrate%', '%Sulfuric Acid%', 
        '%Dyna Cal%', '%Urea%', '%Nitrate%'
    ]
    
    where_clause = " OR ".join([f"ITEMDESC LIKE '{t}'" for t in targets])
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        query = f"""
        SELECT 
            ITEMNMBR, 
            ITEMDESC, 
            CURRCOST, 
            STNDCOST, 
            UOMSCHDL, 
            DECPLQTY
        FROM IV00101
        WHERE ({where_clause})
          AND CURRCOST > 0
        ORDER BY ITEMDESC
        """
        
        df = pd.read_sql(query, conn)
        
        print(f"Found {len(df)} internal items matching targets.")
        print(df.to_string())
        
        # Save to CSV for reference
        df.to_csv('price_to_beat.csv', index=False)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    get_price_to_beat()
