import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
from datetime import date

def analyze_changes():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    start_date = date(2025, 9, 30)
    print(f"--- Cost Analysis: {start_date} vs Today ---")
    
    # 1. Get Current Cost (Today)
    print("Fetching current costs...")
    query_current = "SELECT ITEMNMBR, ITEMDESC, CURRCOST as CostToday FROM IV00101 WHERE CURRCOST > 0"
    df_current = pd.read_sql(query_current, conn)
    
    # 2. Get Historical Cost (as of Start Date)
    print(f"Fetching historical costs as of {start_date}...")
    # Using the same logic as the report: Last receipt/adj on or before date
    query_hist = """
    WITH RatedTransactions AS (
        SELECT 
            ITEMNMBR,
            UNITCOST,
            DOCDATE,
            ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
        FROM IV30300
        WHERE DOCDATE <= ?
          AND DOCTYPE IN (4, 1) -- 4=Receipt, 1=Adjustment (Increase)
          AND UNITCOST > 0
    )
    SELECT ITEMNMBR, UNITCOST as CostThen
    FROM RatedTransactions
    WHERE rn = 1
    """
    df_hist = pd.read_sql(query_hist, conn, params=[start_date])
    
    # 3. Merge
    df = pd.merge(df_current, df_hist, on='ITEMNMBR', how='inner')
    
    # 4. Calculate Difference
    df['Diff'] = df['CostToday'] - df['CostThen']
    df['AbsDiff'] = df['Diff'].abs()
    df['PctChange'] = (df['Diff'] / df['CostThen']) * 100
    
    # Filter for significant changes
    changes = df[df['AbsDiff'] > 0.01].sort_values('AbsDiff', ascending=False)
    
    print(f"\nFound {len(changes)} items with cost changes > $0.01")
    
    if not changes.empty:
        # Top 20 changes by magnitude
        print("\nTop 20 absolute cost changes:")
        print(changes[['ITEMNMBR', 'ITEMDESC', 'CostThen', 'CostToday', 'Diff', 'PctChange']].head(20).to_string(index=False))
        
        # Save to detailed CSV
        changes.to_csv("cost_changes_sep30_vs_today.csv", index=False)
        print(f"\nFull list saved to 'cost_changes_sep30_vs_today.csv'")
    
    conn.close()

if __name__ == "__main__":
    analyze_changes()
