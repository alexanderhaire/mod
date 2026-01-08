import pyodbc
from secrets_loader import build_connection_string

def check_history_table():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("\n--- Checking IV30300 (Inventory Transaction Amounts History) ---")
        try:
            # Check for table existence and columns
            cursor.execute("SELECT TOP 1 * FROM IV30300")
            columns = [column[0] for column in cursor.description]
            print(f"Columns: {', '.join(columns)}")
            
            # Check a sample content
            cursor.execute("SELECT TOP 3 DOCDATE, ITEMNMBR, TRXQTY FROM IV30300 ORDER BY DOCDATE DESC")
            rows = cursor.fetchall()
            print("\nSample Data (Recent):")
            for row in rows:
                print(row)
                
        except Exception as e:
            print(f"Error accessing IV30300: {e}")

        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    check_history_table()
