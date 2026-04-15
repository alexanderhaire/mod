import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def compare_pdf_prices():
    print("--- QUANTITATIVE RESEARCH: PDF Price Comparison ---")
    
    # 1. Manual Extraction from PDF Output (Simulated for this script based on log)
    # The log showed:
    # "CN-9 $280" -> likely Calcium Nitrate 9%? or CN9?
    # "22-0-12 $670" -> Fertilizer
    # "21-7-14 $845" -> Fertilizer
    
    pdf_offers = [
        {'ItemHint': 'CN-9', 'OfferPrice': 280.00, 'UOM': 'Ton', 'Source': 'YNA EC Price List'},
        {'ItemHint': '22-0-12', 'OfferPrice': 670.00, 'UOM': 'Ton', 'Source': 'YNA EC Price List'},
        {'ItemHint': '21-7-14', 'OfferPrice': 845.00, 'UOM': 'Ton', 'Source': 'YNA EC Price List'},
        {'ItemHint': 'Glacial Acetic Acid', 'OfferPrice': 0.0, 'UOM': 'Unknown', 'Source': 'Offer Sheet'}, # Placeholder if we can't parse price
        {'ItemHint': 'Hexane', 'OfferPrice': 0.0, 'UOM': 'Unknown', 'Source': 'Offer Sheet'},
    ]
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 2. Search Internal Items matching Hints
        print("Searching for internal matches...")
        
        results = []
        
        for offer in pdf_offers:
            hint = offer['ItemHint']
            # Search IV00101
            query = f"""
            SELECT TOP 5 ITEMNMBR, ITEMDESC, CURRCOST, STNDCOST, UOMSCHDL
            FROM IV00101
            WHERE ITEMDESC LIKE '%{hint}%' OR ITEMNMBR LIKE '%{hint}%'
            """
            matches = pd.read_sql(query, conn)
            
            if not matches.empty:
                for idx, row in matches.iterrows():
                    # Calculate Variance
                    current_cost = row['CURRCOST']
                    
                    # Normalized comparison (assuming Ton vs Lb/Kg is hard, just printing side-by-side)
                    results.append({
                        'PDF Item': hint,
                        'Internal Item': row['ITEMNMBR'],
                        'Internal Desc': row['ITEMDESC'],
                        'PDF Price': offer['OfferPrice'],
                        'PDF UOM': offer['UOM'],
                        'Current Cost': current_cost,
                        'Internal UOM': row['UOMSCHDL']
                    })
            else:
                print(f"No match found for '{hint}'")

        # 3. Report
        if results:
            df = pd.DataFrame(results)
            print("\n--- PDF OFFER VS SYSTEM COST ---")
            print(df.to_string())
            
            df.to_csv('pdf_price_comparison.csv', index=False)
        else:
            print("No matches found to compare.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    compare_pdf_prices()
