import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def test_historical_cost():
    target_date = date(2025, 9, 30)
    test_items = ['ZZ2.5GALF', 'GOLDFE02', 'NPKKTS', 'ZZ55GAL', 'ZZ30GAL']
    
    print(f"--- Testing Historical Cost Lookup as of {target_date} ---")
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    # 1. Get Current Cost
    q_current = f"""
    SELECT ITEMNMBR, CURRCOST 
    FROM IV00101 
    WHERE ITEMNMBR IN ({','.join(['?']*len(test_items))})
    """
    current_df = pd.read_sql(q_current, conn, params=test_items)
    current_df['ITEMNMBR'] = current_df['ITEMNMBR'].str.strip()
    
    # 2. Get Historical Cost (new logic)
    q_hist = """
    WITH LastCostBefore AS (
        SELECT 
            ITEMNMBR,
            UNITCOST,
            ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
        FROM IV30300
        WHERE DOCDATE <= ?
          AND DOCTYPE IN (1, 3, 4, 5, 7)
          AND UNITCOST IS NOT NULL
    ),
    FirstCostAfter AS (
        SELECT 
            ITEMNMBR,
            UNITCOST,
            ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE ASC, DEX_ROW_ID ASC) as rn
        FROM IV30300
        WHERE DOCDATE > ?
          AND DOCTYPE IN (1, 3, 4, 5, 7)
          AND UNITCOST IS NOT NULL
    )
    SELECT 
        COALESCE(b.ITEMNMBR, a.ITEMNMBR) as ITEMNMBR,
        COALESCE(b.UNITCOST, a.UNITCOST) as HistoricalCost
    FROM LastCostBefore b
    FULL OUTER JOIN FirstCostAfter a ON b.ITEMNMBR = a.ITEMNMBR AND b.rn = 1 AND a.rn = 1
    WHERE (b.rn = 1 OR b.ITEMNMBR IS NULL) AND (a.rn = 1 OR a.ITEMNMBR IS NULL)
    """
    hist_df = pd.read_sql(q_hist, conn, params=[target_date, target_date])
    hist_df['ITEMNMBR'] = hist_df['ITEMNMBR'].str.strip()
    
    conn.close()
    
    # Filter to test items
    hist_df = hist_df[hist_df['ITEMNMBR'].isin(test_items)]
    
    # Merge
    merged = pd.merge(current_df, hist_df, on='ITEMNMBR', how='outer')
    
    print("\nResults:")
    print(f"{'Item':<15} | {'Current Cost':<15} | {'Historical Cost (9/30)':<20} | {'Match?':<8}")
    print("-" * 65)
    
    for _, row in merged.iterrows():
        item = row['ITEMNMBR']
        curr = row.get('CURRCOST', 0) or 0
        hist = row.get('HistoricalCost', 0) or 0
        match = "YES" if abs(curr - hist) < 0.01 else "NO"
        print(f"{item:<15} | ${curr:<14.4f} | ${hist:<19.4f} | {match:<8}")

if __name__ == "__main__":
    test_historical_cost()
