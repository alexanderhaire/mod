import pyodbc
from secrets_loader import build_connection_string

def inspect_sop_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        for table in ["SOP30200", "SOP30300"]:
            print(f"\nInspecting {table}...")
            cursor.execute(f"SELECT TOP 1 * FROM {table}")
            columns = [column[0] for column in cursor.description]
            print(f"Columns: {columns}")
            
            if "DOCDATE" in columns:
                print(f"  -> Has DOCDATE (Likely Header)")
            if "ITEMNMBR" in columns:
                print(f"  -> Has ITEMNMBR (Likely Line)")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sop_tables()
