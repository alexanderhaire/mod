import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def analyze_ppv_final():
    print("--- QUANTITATIVE RESEARCH: PPV Final Attempt ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Fetch ANY valid receipts
        print("Fetching Valid Receipt Lines (Cost > $0.10)...")
        query = """
        SELECT TOP 10000
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.UNITCOST,
            T1.EXTDCOST,
            T1.UOFM,
            T2.VENDORID,
            T2.receiptdate,
            T2.POPRCTNM
        FROM POP30310 T1
        JOIN POP30300 T2 ON T1.POPRCTNM = T2.POPRCTNM
        WHERE T1.UNITCOST > 0.10
          AND T1.EXTDCOST > 0
        """
        df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} receipt lines.")
        
        if df.empty:
            print("No valid receipt lines found. Data is likely in Open Tables (POP10500) not History.")
            return

        # 2. Analyze Variance
        print("\n--- VARIANCE ANALYSIS ---")
        
        # Clean dates for display (ignore year 4004 for logic, just display)
        # We want items with high variance variance in UNITCOST
        
        stats = df.groupby(['ITEMNMBR', 'UOFM']).agg({
             'ITEMDESC': 'first',
             'UNITCOST': ['min', 'max', 'mean', 'count'],
             'VENDORID': 'nunique'
        })
        stats.columns = ['Description', 'MinCost', 'MaxCost', 'AvgCost', 'TxCount', 'VendorCount']
        stats = stats.reset_index()
        
        # Filter: At least 2 transactions, Spread > 20%
        instability = stats[
            (stats['TxCount'] > 1) & 
            (stats['MaxCost'] > stats['MinCost'] * 1.2)
        ].copy()
        
        instability['SpreadPct'] = ((instability['MaxCost'] - instability['MinCost']) / instability['MinCost']) * 100
        instability = instability.sort_values('SpreadPct', ascending=False)
        
        if not instability.empty:
            print("\nTop Items with Price Instability:")
            print(instability.head(15).to_string(formatters={
                'MinCost': '${:,.2f}'.format,
                'MaxCost': '${:,.2f}'.format,
                'SpreadPct': '{:.0f}%'.format
            }))
            instability.to_csv('ppv_final_report.csv', index=False)
            
            # Deep dive top item
            top_item = instability.iloc[0]['ITEMNMBR']
            print(f"\n--- DEEP DIVE: {top_item} ---")
            history = df[df['ITEMNMBR'] == top_item].sort_values('UNITCOST')
            print(history[['receiptdate', 'VENDORID', 'UNITCOST', 'UOFM']].to_string())
            
        else:
            print("No significant price instability found in this dataset.")
            
        # 3. Vendor Arbitrage (Same item, multiple vendors)
        print("\n--- VENDOR ARBITRAGE ---") 
        multi_vendors = stats[stats['VendorCount'] > 1].copy()
        
        if not multi_vendors.empty:
             print(f"Found {len(multi_vendors)} items bought from multiple vendors.")
             # We can reuse the instability table if it has > 1 vendor
             vendor_arb = instability[instability['VendorCount'] > 1]
             if not vendor_arb.empty:
                 print("\nTop Vendor Arbitrage Opportunities:")
                 print(vendor_arb.head(10).to_string(formatters={
                    'MinCost': '${:,.2f}'.format,
                    'MaxCost': '${:,.2f}'.format,
                    'SpreadPct': '{:.0f}%'.format
                 }))
        else:
            print("No items found with multiple vendors in this sample.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_ppv_final()
