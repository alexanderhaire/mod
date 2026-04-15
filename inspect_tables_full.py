import pyodbc
from secrets_loader import build_connection_string

def inspect_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            
            print("--- IV00102 (Quantity Master) Columns ---")
            cursor.execute("SELECT TOP 1 * FROM IV00102")
            for col in cursor.description:
                print(col[0])

            print("\n--- IV00101 (Item Master) Columns ---")
            cursor.execute("SELECT TOP 1 * FROM IV00101")
            for col in cursor.description:
                print(col[0])
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_tables()
