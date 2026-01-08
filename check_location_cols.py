import pyodbc
from secrets_loader import build_connection_string

def check_location_columns():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = ["IV00102", "IV30300"]
        
        for table in tables:
            print(f"\n--- Checking {table} for Location Columns ---")
            try:
                cursor.execute(f"SELECT TOP 1 * FROM {table}")
                columns = [column[0] for column in cursor.description]
                loc_cols = [c for c in columns if "LOC" in c or "SITE" in c]
                print(f"All Columns: {columns}")
                print(f"Location-related Columns: {loc_cols}")
                
                # specific check for 'MAIN' existence
                if "LOCNCODE" in columns:
                     cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE LOCNCODE = 'MAIN'")
                     count = cursor.fetchone()[0]
                     print(f"Rows with LOCNCODE='MAIN': {count}")
                elif "TRXLOCTN" in columns:
                     cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE TRXLOCTN = 'MAIN'")
                     count = cursor.fetchone()[0]
                     print(f"Rows with TRXLOCTN='MAIN': {count}")

            except Exception as e:
                print(f"Error checking {table}: {e}")

        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    check_location_columns()
