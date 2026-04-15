import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def analyze_market_arbitrage_v2():
    print("--- QUANTITATIVE RESEARCH: Market Price Arbitrage (Fixed) ---")
    
    # 1. Load External Pricing (The "Market")
    try:
        market_file = "Brenntag_Meeting_Prep_2026-01-28.xlsx"
        print(f"Loading market data from {market_file}...")
        market_df = pd.read_excel(market_file)
        
        market_df.rename(columns={
            'Item Number': 'ItemNumber', 
            'Unit Cost ($)': 'MarketCost',
            'Description': 'MarketDesc'
        }, inplace=True)
        
        market_df = market_df.dropna(subset=['ItemNumber', 'MarketCost'])
        print(f"Loaded {len(market_df)} market items.")
        
    except Exception as e:
        print(f"Error loading market file: {e}")
        return

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 2. Fetch Internal Cost (IV00101)
        print("Fetching internal standard costs...")
        query_internal = "SELECT ITEMNMBR, ITEMDESC, CURRCOST, STNDCOST FROM IV00101"
        internal_df = pd.read_sql(query_internal, conn)
        
        # 3. Fetch Sales Prices (SOP30300 joined with SOP30200)
        # FIX: Join with Header to get DOCDATE
        print("Fetching sales prices (ASP)...")
        query_sales = """
        SELECT 
            T2.ITEMNMBR, 
            SUM(T2.XTNDPRCE) / NULLIF(SUM(T2.QTYFULFI), 0) as AvgSellingPrice
        FROM SOP30200 T1
        JOIN SOP30300 T2 ON T1.SOPNUMBE = T2.SOPNUMBE AND T1.SOPTYPE = T2.SOPTYPE
        WHERE T1.DOCDATE >= DATEADD(day, -365, GETDATE())
          AND T1.SOPTYPE = 3
          AND T2.QTYFULFI > 0
        GROUP BY T2.ITEMNMBR
        """
        sales_df = pd.read_sql(query_sales, conn)
        
        # 4. MERGE DATASETS
        merged = pd.merge(market_df, internal_df, left_on='ItemNumber', right_on='ITEMNMBR', how='inner')
        merged = pd.merge(merged, sales_df, on='ITEMNMBR', how='left')
        
        print(f"Matched {len(merged)} items.")
        
        # 5. SOURCING ARBITRAGE
        print("\n--- SOURCING ARBITRAGE (System Cost vs Market Offer) ---")
        merged['SourcingVariance'] = merged['CURRCOST'] - merged['MarketCost']
        merged['SourcingPct'] = (merged['SourcingVariance'] / merged['CURRCOST']) * 100
        
        sourcing_opps = merged[merged['SourcingPct'] > 5].sort_values('SourcingPct', ascending=False)
        
        if not sourcing_opps.empty:
            print(sourcing_opps[['ItemNumber', 'MarketDesc', 'MarketCost', 'CURRCOST', 'SourcingPct']].head(10).to_string(formatters={
                'MarketCost': '${:,.4f}'.format,
                'CURRCOST': '${:,.4f}'.format,
                'SourcingPct': '{:.1f}%'.format
            }))
            
        # 6. FLIP ARBITRAGE (Sales Price vs Market Offer)
        print("\n--- FLIP ARBITRAGE (Sales Price vs Market Offer) ---")
        merged['EstMargin'] = merged['AvgSellingPrice'] - merged['MarketCost']
        merged['EstMarginPct'] = (merged['EstMargin'] / merged['AvgSellingPrice']) * 100
        
        flips = merged[merged['EstMarginPct'] > 20].sort_values('EstMarginPct', ascending=False)
        
        if not flips.empty:
            cols = ['ItemNumber', 'MarketDesc', 'MarketCost', 'AvgSellingPrice', 'EstMarginPct']
            print(flips[cols].head(15).to_string(formatters={
                'MarketCost': '${:,.4f}'.format,
                'AvgSellingPrice': '${:,.2f}'.format,
                'EstMarginPct': '{:.1f}%'.format
            }))
            merged.to_csv('market_arbitrage_report.csv', index=False)
        else:
            print("No significant flips found against this specific offer sheet.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_market_arbitrage_v2()
