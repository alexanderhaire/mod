import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def validate_arbitrage_v2():
    print("--- DIAGNOSTIC: Validating Arbitrage Logic & Currency (Fixed) ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        target_items = ['GRPVIT', 'FREIGHT', 'FLO021202', 'ZZ250GAL', 'FLOCACHL00', 'NO3ZN']
        targets_str = "', '".join(target_items)
        
        print(f"Inspecting Top Targets: {target_items}")
        
        # FIX: Join SOP30200 (Header) for DOCDATE
        query_details = f"""
        SELECT 
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.UOMSCHDL,
            -- VENDOR DATA
            V.VENDORID,
            V.Last_Originating_Cost as OfferPrice,
            V.PRCHSUOM as VendorUOM,
            V.LSRCPTDT as LastReceiptDate,
            V.LSTORDDT as LastOrderDate,
            -- SALES DATA
            Sales.AvgSellPrice,
            Sales.LastSoldDate,
            Sales.SoldUOM
        FROM IV00101 T1
        LEFT JOIN IV00103 V ON T1.ITEMNMBR = V.ITEMNMBR
        LEFT JOIN (
            SELECT 
                S2.ITEMNMBR,
                MAX(S1.DOCDATE) as LastSoldDate,
                MAX(S2.UOFM) as SoldUOM,
                SUM(S2.XTNDPRCE) / NULLIF(SUM(S2.QTYFULFI), 0) as AvgSellPrice
            FROM SOP30200 S1
            JOIN SOP30300 S2 ON S1.SOPNUMBE = S2.SOPNUMBE AND S1.SOPTYPE = S2.SOPTYPE
            WHERE S1.DOCDATE >= '2024-01-01'
              AND S1.SOPTYPE = 3 -- Invoices
            GROUP BY S2.ITEMNMBR
        ) Sales ON T1.ITEMNMBR = Sales.ITEMNMBR
        WHERE T1.ITEMNMBR IN ('{targets_str}')
          AND V.Last_Originating_Cost > 0
        ORDER BY T1.ITEMNMBR
        """
        
        df = pd.read_sql(query_details, conn)
        
        print("\n--- VALIDATION REPORT ---")
        
        for idx, row in df.iterrows():
            item = row['ITEMNMBR']
            vendor = row['VENDORID']
            offer = row['OfferPrice']
            uom_v = row['VendorUOM']
            
            # Helper to pick best date
            last_date = row['LastReceiptDate']
            if pd.isna(last_date):
                 last_date = row['LastOrderDate']
            
            sell_price = row['AvgSellPrice']
            uom_s = row['SoldUOM']
            last_sold = row['LastSoldDate']
            
            print(f"\nITEM: {item} ({row['ITEMDESC']})")
            print(f"  VENDOR: {vendor}")
            print(f"  OFFER:  ${offer:,.4f} per {uom_v}")
            print(f"  DATE:   {last_date}")
            
            if sell_price:
                print(f"  SALES:  ${sell_price:,.4f} per {uom_s}")
                print(f"  SOLD:   {last_sold}")
            else:
                print("  SALES:  No sales in 2024+")

            # LOGIC VERDICT
            is_stale = False
            if last_date:
                if str(last_date) < '2022-01-01':
                    is_stale = True
                    print(f"  [FAIL] STALE DATA: Pricing is from {last_date} (Too old).")
                elif str(last_date) < '2024-01-01':
                    print(f"  [WARN] AGING DATA: Pricing is from {last_date} (Verify).")
                else:
                    print(f"  [PASS] RECENT DATA: {last_date}")
            else:
                print("  [WARN] NO DATE: Cannot verify age of vendor offer.")

            if uom_v != uom_s and uom_s is not None:
                print(f"  [WARN] UOM MISMATCH: Vendor {uom_v} vs Sales {uom_s}. Check conversion.")
                # Simple logic check for Vitamin E
                if item == 'GRPVIT' and uom_v == 'KG' and uom_s == 'LB': # Hypothetical
                    pass 

            if not is_stale and sell_price and (sell_price > offer * 2):
                 print(f"  [VERIFIED] ARBITRAGE CONFIRMED: Margin > 50%")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    validate_arbitrage_v2()
