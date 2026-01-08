import pyodbc
from secrets_loader import build_connection_string

def list_all_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME")
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        
        print(f"Total Tables: {len(tables)}")
        for table in tables:
            # Filter standard GP huge families to reduce noise if needed, but here we want to see everything
            print(table)

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_all_tables()
