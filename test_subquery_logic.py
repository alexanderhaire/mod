
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    receipt = 'RV0000039639'
    item = 'NPKKNO3'
    
    print(f"Testing Subquery for {receipt}...")
    
    cursor.execute(f"""
        SELECT 
            p.ITEMNMBR,
            p.POPRCTNM,
            p.UNITCOST as POP_Cost,
            (SELECT TOP 1 UNITCOST FROM IV10200 iv WHERE iv.RCPTNMBR = p.POPRCTNM AND iv.ITEMNMBR = p.ITEMNMBR) as IV_Cost
        FROM POP30310 p
        WHERE p.POPRCTNM = '{receipt}' AND p.ITEMNMBR = '{item}'
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} rows (Expected 1):")
        for row in rows:
            print(f"POP Cost: {row.POP_Cost}, IV Cost: {row.IV_Cost}")
    else:
        print("No results.")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
