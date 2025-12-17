
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # Receipts that had duplicates: RV0000008166, RV0000008181, RV0000037025
    receipts_to_check = ['RV0000008166', 'RV0000008181', 'RV0000037025']
    
    print("Verifying grouping logic...")
    
    for receipt in receipts_to_check:
        print(f"\nChecking {receipt}...")
        
        # This is the logic we put in market_insights.py
        cursor.execute(f"""
            SELECT 
                l.POPRCTNM,
                MAX(l.ITEMNMBR) as ITEMNMBR,
                COUNT(*) as RawLineCount,
                SUM(l.UMQTYINB) as TotalQty,
                AVG(l.UNITCOST) as AvgUnitCost
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE l.POPRCTNM = '{receipt}'
                AND l.LOCNCODE = 'MAIN'
            GROUP BY l.POPRCTNM, l.ITEMNMBR
        """)
        
        rows = cursor.fetchall()
        for row in rows:
            print(f"  Result Rows: 1 (Aggregated from {row.RawLineCount} lines)")
            print(f"  Total Qty: {row.TotalQty}")
            print(f"  Avg Cost: {row.AvgUnitCost}")
            
    conn.close()
except Exception as e:
    print(f"Error: {e}")
