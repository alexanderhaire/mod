import pyodbc
from secrets_loader import build_connection_string

def check_iv_columns():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Checking IV00101 Columns ---")
        cursor.execute("SELECT TOP 1 * FROM IV00101")
        columns = [column[0] for column in cursor.description]
        print(columns)
        
        # Check for 'UOFM' or 'UOM'
        print("\nPossible UOM Columns:")
        for c in columns:
            if 'UOM' in c or 'UOFM' in c:
                print(f"  - {c}")

    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_iv_columns()
