import pyodbc
from secrets_loader import build_connection_string

def check_sop_columns():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Checking SOP10200 Columns ---")
        cursor.execute("SELECT TOP 1 * FROM SOP10200")
        columns = [column[0] for column in cursor.description]
        print(columns)
        
        # Check for 'REM' or 'QTY'
        print("\nPossible Remaining Qty Columns:")
        for c in columns:
            if 'REM' in c or 'QTY' in c:
                print(f"  - {c}")

    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_sop_columns()
