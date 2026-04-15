import pyodbc
from secrets_loader import build_connection_string

def discover_production_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Checking Specific GP Manufacturing Tables ---")
        # Known GP Mfg Tables
        # MOP10000: MO Master?
        # WO010032: MO Receipt Header
        # MOP30100: MO History?
        
        candidates = [
            'MOP10000', 'MOP10213', 'WO010032', 'WO010033', 
            'IV30300', # Inventory Transaction History (might have MO receipts)
            'MOP30100'
        ]
        
        for table in candidates:
            try:
                print(f"\nChecking table: {table}")
                cursor.execute(f"SELECT TOP 3 * FROM {table}")
                rows = cursor.fetchall()
                if rows:
                    columns = [column[0] for column in cursor.description]
                    print(f"  Columns: {columns}")
                    for row in rows:
                        print(f"  Row: {list(row)}")
                        
                    # If IV30300, check for DOCTYPE related to Manufacturing
                    if table == 'IV30300':
                        print("  Checking for MO Receipts (DOCTYPE=1?) in IV30300...")
                        # DOCTYPE 1 is Adjustment, 2 is Variance, 3 is Transfer... 
                        # Need to verify which DOCTYPE represents MO Receipt. usually it's an Adjustment or distinct type.
                        # In standard GP, MO Receipt might show as DOCTYPE 1 (Adjustment) with specific source.
                        cursor.execute("SELECT TOP 5 DOCTYPE, DOCNUMBR, ITEMNMBR, TRXQTY FROM IV30300 WHERE DOCNUMBR LIKE 'MO%' OR DOCNUMBR LIKE 'WO%'")
                        mo_rows = cursor.fetchall()
                        for r in mo_rows:
                            print(f"    MO Transaction: {list(r)}")

                else:
                    print("  (Empty)")
            except Exception as e:
                print(f"  Error reading {table}: {e}")

    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    discover_production_tables()
