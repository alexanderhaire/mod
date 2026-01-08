import pyodbc
from secrets_loader import build_connection_string

def inspect_tascmfg():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        table_name = "TASCMFG"
        print(f"Inspecting columns for {table_name}...")
        
        cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
        columns = [column[0] for column in cursor.description]
        
        print(f"Columns: {', '.join(columns)}")
        
        # Check GOLDCA00
        print(f"\nChecking GOLDCA00 in {table_name}...")
        cursor.execute(f"SELECT * FROM {table_name} WHERE ITEMNMBR = 'GOLDCA00'")
        row = cursor.fetchone()
        if row:
            for i, col in enumerate(columns):
                print(f"{col}: {row[i]}")
        else:
             # Try to find what key is used if not ITEMNMBR
             print("GOLDCA00 not found or ITEMNMBR column missing. Checking first row:")
             cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
             row = cursor.fetchone()
             for i, col in enumerate(columns):
                 print(f"{col}: {row[i]}")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_tascmfg()
