import pyodbc
from secrets_loader import build_connection_string

def inspect_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        tables = ["IV00101", "IV00102", "GL00105"]
        
        for table in tables:
            print(f"\n--- Checking {table} ---")
            try:
                cursor.execute(f"SELECT TOP 1 * FROM {table}")
                columns = [column[0] for column in cursor.description]
                print(f"Columns: {', '.join(columns)}")
            except Exception as e:
                print(f"Error checking {table}: {e}")

        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    inspect_tables()
