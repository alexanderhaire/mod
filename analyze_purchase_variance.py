import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def analyze_purchase_variance_v3():
    print("--- QUANTITATIVE RESEARCH: Purchase Price Variance (PPV) - Wide Net ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("Fetching Receipt History (Wide Date Range)...")
        # Removing date filter to see what we get, or setting it very wide
        query_receipts = """
        SELECT TOP 5000
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.UNITCOST,
            T1.EXTDCOST,
            T1.ACTLSHIP as QtyReceived, 
            T1.UOFM,
            T2.VENDORID,
            T2.receiptdate as ReceiptDate
        FROM POP30310 T1
        JOIN POP30300 T2 ON T1.POPRCTNM = T2.POPRCTNM
        WHERE T1.UNITCOST > 0.05
          AND T1.ACTLSHIP > 0
        ORDER BY T2.receiptdate DESC
        """
        df = pd.read_sql(query_receipts, conn)
        print(f"Analyzed {len(df)} receipt lines.")
        
        if df.empty:
            print("No receipts found even with wide filter.")
            return

        # 2. Identify High Variance Items
        print("\n--- PRICE VARIANCE ANALYSIS ---")
        
        # Filter out invalid future dates for display, but keep for analysis if valid cost
        df['ReceiptDate'] = pd.to_datetime(df['ReceiptDate'], errors='coerce')
        
        # Group by Item + UOM (Price varies by UOM)
        variance_stats = df.groupby(['ITEMNMBR', 'UOFM']).agg({
             'ITEMDESC': 'first',
             'UNITCOST': ['min', 'max', 'mean', 'count'],
             'EXTDCOST': 'sum'
        })
        
        variance_stats.columns = ['Description', 'MinCost', 'MaxCost', 'AvgCost', 'TxCount', 'TotalSpend']
        variance_stats = variance_stats.reset_index()
        
        # "High Variance": Max > 1.3 * Min
        high_variance = variance_stats[
            (variance_stats['TxCount'] > 2) & 
            (variance_stats['MaxCost'] > variance_stats['MinCost'] * 1.3)
        ].copy()
        
        high_variance['SpreadPct'] = ((high_variance['MaxCost'] - high_variance['MinCost']) / high_variance['MinCost']) * 100
        high_variance = high_variance.sort_values('SpreadPct', ascending=False)
        
        if not high_variance.empty:
            cols = ['ITEMNMBR', 'Description', 'UOFM', 'MinCost', 'MaxCost', 'SpreadPct', 'TxCount']
            print("\nTop 15 Items with Extreme Price Instability:")
            print(high_variance[cols].head(15).to_string(formatters={
                'MinCost': '${:,.4f}'.format,
                'MaxCost': '${:,.4f}'.format,
                'SpreadPct': '{:.0f}%'.format
            }))
            
            high_variance.to_csv('ppv_variance_report.csv', index=False)
        else:
             print("No significant price variance found.")
             
        # 3. Vendor Arbitrage 
        print("\n--- VENDOR ARBITRAGE (Same Item, Diff Vendor) ---")
        
        site_level = df.groupby(['ITEMNMBR', 'UOFM', 'VENDORID'])['UNITCOST'].mean().reset_index()
        
        counts = site_level.groupby(['ITEMNMBR', 'UOFM'])['VENDORID'].count()
        multi_vendor_items = counts[counts > 1].index
        
        multi_df = site_level[site_level.set_index(['ITEMNMBR', 'UOFM']).index.isin(multi_vendor_items)].copy()
        
        spreads = multi_df.groupby(['ITEMNMBR', 'UOFM'])['UNITCOST'].agg(['min', 'max']).reset_index()
        spreads['SpreadPct'] = ((spreads['max'] - spreads['min']) / spreads['min']) * 100
        
        opps = spreads[spreads['SpreadPct'] > 20].sort_values('SpreadPct', ascending=False)
        
        if not opps.empty:
            print("\nItems with Multi-Vendor Price Gaps (>20%):")
            print(opps.head(10).to_string(formatters={
                'min': '${:,.4f}'.format,
                'max': '${:,.4f}'.format,
                'SpreadPct': '{:.0f}%'.format
            }))
            opps.to_csv('vendor_consolidation_report.csv', index=False)
        else:
            print("No multi-vendor arbitrage found.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_purchase_variance_v3()
