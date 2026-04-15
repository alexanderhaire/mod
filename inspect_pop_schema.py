import pyodbc
from secrets_loader import build_connection_string

def inspect_pop_schema():
    print("--- DIAGNOSTIC: POP Schema Inspection ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = ['POP30300', 'POP30310']
        
        for t in tables:
            print(f"\nScanning {t}...")
            try:
                cursor.execute(f"SELECT TOP 1 * FROM {t}")
                cols = [column[0] for column in cursor.description]
                print(f"Columns: {cols}")
            except Exception as e:
                print(f"Error reading {t}: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    inspect_pop_schema()
