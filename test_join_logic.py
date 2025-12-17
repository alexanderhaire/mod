
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
    
    print(f"Testing Join for {receipt}...")
    
    cursor.execute(f"""
        SELECT 
            p.ITEMNMBR,
            p.POPRCTNM,
            p.UNITCOST as POP_Cost,
            iv.UNITCOST as IV_Cost,
            iv.ADJUNITCOST as IV_AdjCost,
            iv.RCPTNMBR as IV_Receipt,
            iv.TRXLOCTN
        FROM POP30310 p
        LEFT JOIN IV10200 iv ON p.ITEMNMBR = iv.ITEMNMBR 
            AND p.POPRCTNM = iv.RCPTNMBR
            AND p.LOCNCODE = iv.TRXLOCTN
        WHERE p.POPRCTNM = '{receipt}' AND p.ITEMNMBR = '{item}'
    """)
    
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"POP Cost: {row.POP_Cost}, IV Cost: {row.IV_Cost}, Receipt Match: {row.POPRCTNM} == {row.IV_Receipt}")
    else:
        print("No join results.")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
