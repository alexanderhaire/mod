
import pyodbc
from secrets_loader import build_connection_string

def check_pm_tables():
    try:
        conn_str, _, _, _ = build_connection_string()
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE 'PM%' AND TABLE_SCHEMA = 'dbo'")
            rows = cursor.fetchall()
            print("Found PM tables:")
            for row in rows:
                print(row.TABLE_NAME)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pm_tables()
