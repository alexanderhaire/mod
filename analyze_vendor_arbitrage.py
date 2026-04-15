import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def analyze_vendor_arbitrage():
    print("--- QUANTITATIVE RESEARCH: Vendor Offer Sheet Arbitrage ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Fetch Vendor Offers (IV00103)
        print("Fetching Vendor Offers (IV00103)...")
        # We assume Last_Originating_Cost is the most recent "Quote" or "Price Paid"
        query_offers = """
        SELECT 
            T1.ITEMNMBR,
            T1.VENDORID,
            T1.VNDITDSC as VendorDesc,
            T1.Last_Originating_Cost as OfferPrice,
            T1.PRCHSUOM as VendorUOM
        FROM IV00103 T1
        WHERE T1.Last_Originating_Cost > 0
        """
        offers_df = pd.read_sql(query_offers, conn)
        print(f"Loaded {len(offers_df)} vendor offers.")

        # 2. Fetch Internal & Sales Data (IV00101 + SOP)
        print("Fetching Internal Costs & Sales Prices...")
        query_internal = """
        SELECT 
            T1.ITEMNMBR, 
            T1.ITEMDESC, 
            T1.CURRCOST as SystemCost,
            (SELECT SUM(S2.XTNDPRCE) / NULLIF(SUM(S2.QTYFULFI), 0)
             FROM SOP30200 S1
             JOIN SOP30300 S2 ON S1.SOPNUMBE = S2.SOPNUMBE AND S1.SOPTYPE = S2.SOPTYPE
             WHERE S1.DOCDATE >= DATEADD(day, -365, GETDATE())
               AND S1.SOPTYPE = 3
               AND S2.QTYFULFI > 0
               AND S2.ITEMNMBR = T1.ITEMNMBR
            ) as AvgSellingPrice
        FROM IV00101 T1
        WHERE T1.INACTIVE = 0
        """
        internal_df = pd.read_sql(query_internal, conn)
        
        # 3. MERGE
        merged = pd.merge(offers_df, internal_df, on='ITEMNMBR', how='inner')
        print(f"Matched {len(merged)} active offers.")
        
        # 4. FIND SOURCING ARBITRAGE (Offer < System Cost)
        # Is there a vendor offering it cheaper than our current standard cost?
        
        print("\n--- SOURCING ARBITRAGE (Legacy Cost vs Vendor Offer) ---")
        merged['SourcingDiff'] = merged['SystemCost'] - merged['OfferPrice']
        merged['SourcingPct'] = (merged['SourcingDiff'] / merged['SystemCost']) * 100
        
        # Filter: > 10% savings, price > $1 (avoid penny dust)
        sourcing_opps = merged[(merged['SourcingPct'] > 10) & (merged['SystemCost'] > 1)].sort_values('SourcingPct', ascending=False)
        
        if not sourcing_opps.empty:
            print(sourcing_opps[['ITEMNMBR', 'VENDORID', 'OfferPrice', 'SystemCost', 'SourcingPct']].head(15).to_string(formatters={
                'OfferPrice': '${:,.4f}'.format,
                'SystemCost': '${:,.4f}'.format,
                'SourcingPct': '{:.1f}%'.format
            }))
            sourcing_opps.to_csv('vendor_sourcing_report.csv', index=False)
        else:
            print("No significant sourcing arbitrage found.")
            
        # 5. FIND FLIP ARBITRAGE (Sales Price vs Offer)
        # Can we buy from this vendor and flip it immediately?
        print("\n--- FLIP ARBITRAGE (Sales vs Vendor Offer) ---")
        merged['FlipMargin'] = merged['AvgSellingPrice'] - merged['OfferPrice']
        merged['FlipMarginPct'] = (merged['FlipMargin'] / merged['AvgSellingPrice']) * 100
        
        flips = merged[(merged['FlipMarginPct'] > 50) & (merged['AvgSellingPrice'] > 0)].sort_values('FlipMarginPct', ascending=False)
        
        if not flips.empty:
            print(flips[['ITEMNMBR', 'VENDORID', 'OfferPrice', 'AvgSellingPrice', 'FlipMarginPct']].head(15).to_string(formatters={
                'OfferPrice': '${:,.4f}'.format,
                'AvgSellingPrice': '${:,.2f}'.format,
                'FlipMarginPct': '{:.1f}%'.format
            }))
            flips.to_csv('vendor_flip_report.csv', index=False)
            
            top_flip = flips.iloc[0]
            print(f"\n⚡ TOP VENDOR FLIP: {top_flip['ITEMNMBR']}")
            print(f"   Buy from: {top_flip['VENDORID']} @ ${top_flip['OfferPrice']:.2f}")
            print(f"   Sell for: ${top_flip['AvgSellingPrice']:.2f}")
            print(f"   Margin:   {top_flip['FlipMarginPct']:.1f}%")
        else:
            print("No vendor flip opportunities found.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_vendor_arbitrage()
