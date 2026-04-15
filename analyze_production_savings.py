import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def analyze_production_savings_v3():
    print("--- QUANTITATIVE RESEARCH: Production Efficiency & Savings (Fixed Cost) ---")
    
    # 1. Load Purity Arbitrage Data
    try:
        arb_df = pd.read_csv('chemical_arbitrage_report.csv')
        print(f"Loaded {len(arb_df)} arbitrage opportunities.")
    except Exception as e:
        print(f"Error loading arbitrage report: {e}")
        return

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 2. Fetch usage with fallbacks for cost
        print("Fetching production usage...")
        query_usage = """
        SELECT TOP 200
            T3.ITEMDESC as ComponentDesc,
            T1.ITEMNMBR as ComponentID,
            SUM(T1.QTY_ISSUED_I) as TotalQtyUsed,
            MAX(T3.CURRCOST) as CurrentStdCost, -- Use Item Master Cost
            SUM(T1.QTY_ISSUED_I) * MAX(T3.CURRCOST) as EstTotalSpend
        FROM PK010033 T1
        LEFT JOIN IV00101 T3 ON T1.ITEMNMBR = T3.ITEMNMBR
        WHERE T1.QTY_ISSUED_I > 0
        GROUP BY T3.ITEMDESC, T1.ITEMNMBR
        ORDER BY EstTotalSpend DESC
        """
        usage_df = pd.read_sql(query_usage, conn)
        print(f"Found {len(usage_df)} components.")
        
        savings_opportunities = []
        
        for idx, row in usage_df.iterrows():
            desc = str(row['ComponentDesc'])
            std_cost = row['CurrentStdCost']
            
            # Skip invalid costs
            if pd.isna(std_cost) or std_cost <= 0:
                continue
                
            matched_family = None
            
            # Fuzzy match attempt
            for _, arb_row in arb_df.iterrows():
                fam = arb_row['Family'].lower()
                if fam in desc.lower():
                    matched_family = arb_row
                    break
            
            if matched_family is not None:
                best_unit_val = matched_family['BestUnitVal']
                best_desc = matched_family['BestDesc']
                
                # Try to parse current purity
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', desc)
                if match:
                    current_purity = float(match.group(1))
                    if current_purity < 0.1: continue

                    # Cost per 1% using Standard Cost
                    current_cost_per_pct = std_cost / current_purity
                    
                    # Spread Calculation
                    # Is our current item significantly more expensive than the best option?
                    if current_cost_per_pct > (best_unit_val * 1.1): # 10% buffer
                        
                        # Calculate Savings
                        # Active Purity Used = Qty * Purity
                        # Optimized Cost = Active Purity Used * BestUnitVal (Price per 1% purity)
                        
                        total_purity_units = row['TotalQtyUsed'] * current_purity
                        optimized_spend = total_purity_units * best_unit_val
                        actual_spend = row['EstTotalSpend']
                        
                        savings = actual_spend - optimized_spend
                        
                        if savings > 10: # Lower threshold for demo
                            savings_opportunities.append({
                                'Component': desc,
                                'QtyUsed': row['TotalQtyUsed'],
                                'CurrentPrice': std_cost,
                                'OptimalItem': best_desc,
                                'OptimalPricePerPct': best_unit_val,
                                'EstSpend': actual_spend,
                                'PotentialSavings': savings,
                                'SavingsPct': (savings / actual_spend) * 100
                            })
        
        # 4. Report
        if savings_opportunities:
            res = pd.DataFrame(savings_opportunities).sort_values('PotentialSavings', ascending=False)
            print("\n--- IDENTIFIED SAVINGS ---")
            print(res[['Component', 'OptimalItem', 'EstSpend', 'PotentialSavings', 'SavingsPct']].to_string(formatters={
                'EstSpend': '${:,.2f}'.format,
                'PotentialSavings': '${:,.2f}'.format,
                'SavingsPct': '{:.1f}%'.format
            }))
            res.to_csv('production_savings_fixed.csv', index=False)
        else:
            print("\nNo savings found. (Items matched but prices were already optimal or spread too low)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_production_savings_v3()
