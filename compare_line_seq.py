
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    receipt = 'RV0000039639' # The one we know exists in both
    
    print(f"Comparing Line/Seq for {receipt}...")
    
    # Get POP Lines
    print("--- POP30310 ---")
    cursor.execute(f"SELECT RCPTLNNM, ITEMNMBR, UNITCOST FROM POP30310 WHERE POPRCTNM='{receipt}'")
    pop_rows = cursor.fetchall()
    for row in pop_rows:
        print(f"POP Line: {row.RCPTLNNM}, Item: {row.ITEMNMBR.strip()}")

    # Get IV Layers
    print("\n--- IV10200 ---")
    cursor.execute(f"SELECT RCTSEQNM, ITEMNMBR, UNITCOST FROM IV10200 WHERE RCPTNMBR='{receipt}'")
    iv_rows = cursor.fetchall()
    for row in iv_rows:
        print(f"IV Seq: {row.RCTSEQNM}, Item: {row.ITEMNMBR.strip()}")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
