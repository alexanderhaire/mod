import pyodbc
from secrets_loader import build_connection_string

def get_schema():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    tables = ['MOP1000', 'PK010033']
    
    for table in tables:
        print(f"\n--- {table} ---")
        try:
            cursor.execute(f"SELECT TOP 1 * FROM {table}")
            columns = [column[0] for column in cursor.description]
            print(columns)
        except Exception as e:
            print(f"Error: {e}")
            
    conn.close()

if __name__ == "__main__":
    get_schema()
