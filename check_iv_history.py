import pyodbc
from secrets_loader import build_connection_string

def check_iv_history():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # MOs that the user implied were mistakes
        mos = ['X2602-0018', 'C2602-0054']
        
        print(f"--- Checking IV30300 for MOs {mos} ---")
        
        # Check DOCNUMBR match
        print("\nChecking by DOCNUMBR:")
        placeholders = ",".join("?" for _ in mos)
        query = f"SELECT DOCTYPE, DOCNUMBR, ITEMNMBR, TRXQTY, DOCDATE FROM IV30300 WHERE DOCNUMBR IN ({placeholders})"
        cursor.execute(query, mos)
        rows = cursor.fetchall()
        if rows:
            for r in rows: print(f"  {list(r)}")
        else:
            print("  No direct matches in DOCNUMBR.")

        # Check if they are referenced in 'Reference' or similar? 
        # Usually Quick MOs create an adjustment.
        
        # New approach: Check 'MOP10213' (Manufacturing Order Receipt)
        print("\nChecking MOP10213 (MO Receipts):")
        query = f"SELECT MANUFACTUREORDER_I, RCPTNMBR, ITEMNMBR, QTYSOLD FROM MOP10213 WHERE MANUFACTUREORDER_I IN ({placeholders})"
        # MOP10213 might not have QTYSOLD, let's check basic cols or *
        query = f"SELECT MANUFACTUREORDER_I, RCPTNMBR, ITEMNMBR FROM MOP10213 WHERE MANUFACTUREORDER_I IN ({placeholders})"
        cursor.execute(query, mos)
        rows = cursor.fetchall()
        if rows:
            for r in rows: print(f"  {list(r)}")
        else:
            print("  No matches in MOP10213.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_iv_history()
