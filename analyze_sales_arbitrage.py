import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def analyze_sales_flips():
    print("--- QUANTITATIVE RESEARCH: Chemical Sales Arbitrage (The Flip) ---")
    
    # 1. Load Purity Arbitrage Data (The "Optimal Cost")
    try:
        arb_df = pd.read_csv('chemical_arbitrage_report.csv')
        print(f"Loaded {len(arb_df)} sourcing opportunities.")
    except Exception as e:
        print(f"Error loading arbitrage report: {e}")
        return

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 2. Fetch Sales History (What we sell it for)
        print("Fetching sales history (Last 12 Months)...")
        # SOP30200 = History Header, SOP30300 = History Line
        query_sales = """
        SELECT 
            T2.ITEMNMBR,
            T2.ITEMDESC,
            SUM(T2.QTYFULFI) as TotalQtySold,
            SUM(T2.XTNDPRCE) as TotalRevenue,
            SUM(T2.XTNDPRCE) / NULLIF(SUM(T2.QTYFULFI), 0) as AvgSellingPrice
        FROM SOP30200 T1
        JOIN SOP30300 T2 ON T1.SOPNUMBE = T2.SOPNUMBE AND T1.SOPTYPE = T2.SOPTYPE
        WHERE T1.DOCDATE >= DATEADD(day, -365, GETDATE())
          AND T1.VOIDSTTS = 0
          AND T2.QTYFULFI > 0
          AND T1.SOPTYPE = 3 -- Invoices
        GROUP BY T2.ITEMNMBR, T2.ITEMDESC
        HAVING SUM(T2.XTNDPRCE) > 1000 -- Filter distinct noise
        ORDER BY TotalRevenue DESC
        """
        sales_df = pd.read_sql(query_sales, conn)
        print(f"Analyzed {len(sales_df)} sold items.")
        
        # 3. Match Sales to Optimal Sources
        flips = []
        
        for idx, row in sales_df.iterrows():
            desc = str(row['ITEMDESC'])
            asp = row['AvgSellingPrice']
            
            # Fuzzy match to family
            matched_family = None
            for _, arb_row in arb_df.iterrows():
                if arb_row['Family'].lower() in desc.lower():
                    matched_family = arb_row
                    break
            
            if matched_family is not None:
                # We found a match. Can we make it cheaper than we sell it?
                
                # Parse purity of SOLD item
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', desc)
                if match:
                    sold_purity = float(match.group(1))
                    
                    # Optimal Cost to Make/Buy this Purity
                    # (BestPricePerPct * SoldPurity)
                    best_unit_val = matched_family['BestUnitVal']
                    optimal_cost = best_unit_val * sold_purity
                    
                    # Margin Calculation
                    gross_margin = asp - optimal_cost
                    margin_pct = (gross_margin / asp) * 100
                    
                    # Only huge wins
                    if margin_pct > 50:
                        flips.append({
                            'ItemSold': desc,
                            'AvgSellingPrice': asp,
                            'SoldPurity': sold_purity,
                            'OptimalSource': matched_family['BestDesc'],
                            'OptimalSourceCost': optimal_cost,
                            'MarginPerUnit': gross_margin,
                            'MarginPct': margin_pct,
                            'TotalRevenue': row['TotalRevenue'],
                            'EstProfit': row['TotalQtyUsed'] * gross_margin if 'TotalQtyUsed' in row else row['TotalQtySold'] * gross_margin
                        })

        # 4. Report
        if flips:
            flip_df = pd.DataFrame(flips).sort_values('EstProfit', ascending=False)
            
            print("\n--- TOP FLIP OPPORTUNITIES ---")
            print("(Items we sell where we can source the active ingredient massively cheaper)")
            
            print(flip_df[['ItemSold', 'OptimalSource', 'AvgSellingPrice', 'OptimalSourceCost', 'MarginPct']].head(20).to_string(formatters={
                'AvgSellingPrice': '${:,.2f}'.format,
                'OptimalSourceCost': '${:,.2f}'.format,
                'MarginPct': '{:.1f}%'.format
            }))
            
            flip_df.to_csv('chemical_flip_report.csv', index=False)
            print("\nSaved report to: chemical_flip_report.csv")
            
            # Identify the "Crown Jewel"
            top_flip = flip_df.iloc[0]
            print(f"\n💎 CROWN JEWEL: {top_flip['ItemSold']}")
            print(f"   Sell for: ${top_flip['AvgSellingPrice']:.2f}")
            print(f"   Make for: ${top_flip['OptimalSourceCost']:.2f} (using {top_flip['OptimalSource'][:30]}...)")
            print(f"   Margin:   {top_flip['MarginPct']:.1f}%")
            
        else:
            print("\nNo huge flip opportunities found (Margins < 50% or no matches).")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_sales_flips()
