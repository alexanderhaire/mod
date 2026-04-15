import pyodbc
from secrets_loader import build_connection_string
import sys

def find_tables(pattern):
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' AND TABLE_NAME LIKE '%{pattern}%' ORDER BY TABLE_NAME"
        cursor.execute(query)
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        
        print(f"Tables matching '{pattern}': {len(tables)}")
        for table in tables:
            print(table)

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_tables(sys.argv[1])
    else:
        print("Usage: python find_tables.py <pattern>")
