import pyodbc
from secrets_loader import build_connection_string

def discover_sheets():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Searching for 'X' and 'C' Sheets (Attempt 2) ---")
        
        # Check IV30200 (Inventory Header)
        print("\nChecking IV30200 (Header)...")
        # Check DOCNUMBR
        cursor.execute("SELECT TOP 5 DOCNUMBR, BACHNUMB, DOCDATE FROM IV30200 WHERE DOCNUMBR LIKE 'C%' AND DOCDATE > '2024-01-01'")
        print("  C-Docs (IV30200):")
        for r in cursor.fetchall(): print(f"    {list(r)}")
            
        cursor.execute("SELECT TOP 5 DOCNUMBR, BACHNUMB, DOCDATE FROM IV30200 WHERE DOCNUMBR LIKE 'X%' AND DOCDATE > '2024-01-01'")
        print("  X-Docs (IV30200):")
        for r in cursor.fetchall(): print(f"    {list(r)}")

        # Check BACHNUMB
        cursor.execute("SELECT TOP 5 DOCNUMBR, BACHNUMB, DOCDATE FROM IV30200 WHERE BACHNUMB LIKE 'C%' AND DOCDATE > '2024-01-01'")
        print("  C-Batches (IV30200):")
        for r in cursor.fetchall(): print(f"    {list(r)}")

        cursor.execute("SELECT TOP 5 DOCNUMBR, BACHNUMB, DOCDATE FROM IV30200 WHERE BACHNUMB LIKE 'X%' AND DOCDATE > '2024-01-01'")
        print("  X-Batches (IV30200):")
        for r in cursor.fetchall(): print(f"    {list(r)}")

        # Check WO010032 (MO Entry)
        print("\nChecking WO010032 (MO Entry)...")
        try:
            cursor.execute("SELECT TOP 5 MANUFACTUREORDER_I, DSCRIPTN, ITEMNMBR, ENDQTY_I FROM WO010032 WHERE MANUFACTUREORDER_I LIKE 'C%'")
            print("  C-MOs:")
            rows = cursor.fetchall()
            if rows:
                for r in rows: print(f"    {list(r)}")
            else:
                print("    (None)")
            
            cursor.execute("SELECT TOP 5 MANUFACTUREORDER_I, DSCRIPTN, ITEMNMBR, ENDQTY_I FROM WO010032 WHERE MANUFACTUREORDER_I LIKE 'X%'")
            print("  X-MOs:")
            rows = cursor.fetchall()
            if rows:
                for r in rows: print(f"    {list(r)}")
            else:
                print("    (None)")
        except Exception as e:
            print(f"Error checking WO010032: {e}")

    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    discover_sheets()
